"""Chat engine shared by the CLI (run.py) and the web backend (web_backend.py).

One turn:
1. Retrieve Wikipedia QA reference context (RAG), if enabled.
2. Sample Best-of-N candidates and keep the one with the highest calibrated,
   length-normalized log-likelihood.
3. If the model chose to call the calculator, evaluate the expression, append the
   call and its result to the conversation as a tool exchange, and generate again.
   The model decides whether a calculation is needed; nothing forces a tool call.
"""

from __future__ import annotations

import json
import math
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    LogitsProcessorList,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)

from calibration import SimpleTemperatureScaler, TokenLogProbRecorder
from config import ChatConfig
from model_files import find_pending_downloads, report_pending_downloads
from tools import calculator
from tools.rag import RAGRowCountError, get_rag

PROJECT_DIR = Path(__file__).resolve().parent
TOOL_PATTERN = re.compile(r"<calculator>(.*?)</calculator>", re.DOTALL | re.IGNORECASE)
_OPEN_TOOL_TAG = re.compile(r"</?calculator>", re.IGNORECASE)

# Few-shot examples shown whenever the calculator is available. The last one
# demonstrates answering directly when a question only mentions a number.
FEW_SHOT_EXAMPLES: List[List[Dict[str, str]]] = [
    [
        {"role": "user", "content": "A jacket costs $80, has a 25% discount, and then 8% sales tax is applied to the discounted price. What is the final price?"},
        {"role": "assistant", "content": "<calculator>80 * (1 - 25 / 100) * (1 + 8 / 100)</calculator>"},
        {"role": "user", "content": "Calculator result for 80 * (1 - 25 / 100) * (1 + 8 / 100): 64.8\nAnswer the question using this result."},
        {"role": "assistant", "content": "After the 25% discount the jacket costs 60 dollars, and 8% tax adds 4.80 dollars, so the final price is **64.80 dollars**."},
    ],
    [
        {"role": "user", "content": "What is 17% of 240?"},
        {"role": "assistant", "content": "<calculator>240 * 17 / 100</calculator>"},
        {"role": "user", "content": "Calculator result for 240 * 17 / 100: 40.8\nAnswer the question using this result."},
        {"role": "assistant", "content": "17% of 240 is **40.8**."},
    ],
    [
        {"role": "user", "content": "What is GPT-3, and how many parameters does it have?"},
        {"role": "assistant", "content": "GPT-3 is a large language model released by OpenAI in 2020. It is a decoder-only transformer with about **175 billion parameters**."},
    ],
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def strip_think_tags(text: str) -> str:
    text = re.sub(r"<think\b[^>]*>.*?</think\s*>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<think\b[^>]*>.*$", "", text, flags=re.DOTALL | re.IGNORECASE)
    return re.sub(r"</?think\s*>", "", text, flags=re.IGNORECASE).strip()


class ChatEngine:
    def __init__(self, config: Optional[ChatConfig] = None, include_tts_in_report: bool = False):
        """Load the LLM and RAG retriever concurrently.

        include_tts_in_report: also list missing TTS files in the startup download
        report (the web backend downloads them alongside, the CLI does not use TTS).
        """
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available. This application requires a CUDA-enabled GPU.")

        self.config = config or ChatConfig()
        self.device = torch.device("cuda")
        self.calculator_log_path = PROJECT_DIR / "calculator_calls.jsonl"
        self._generation_lock = threading.Lock()
        self._stop_pattern = self._compile_stop_pattern(self.config.stop_sequences)

        if self.config.enable_calculator:
            calculator.warm_up()  # starts the worker process while the models load

        print(f"Using device: {self.device}")
        report_pending_downloads(find_pending_downloads(self.config, include_tts=include_tts_in_report))

        # The LLM and the RAG retriever (embedding model + dataset/index) are
        # independent, so they download and load at the same time. Downloads are
        # network-bound and release the GIL, so missing files arrive in parallel.
        started = time.perf_counter()
        with ThreadPoolExecutor(max_workers=2, thread_name_prefix="model-load") as pool:
            llm_future = pool.submit(self._load_llm)
            rag_future = pool.submit(self._load_rag) if self.config.enable_rag else None
            # The LLM is required: its exception propagates. RAG is optional, except
            # that a wrong dataset row count (RAGRowCountError) also stops startup.
            self.rag = rag_future.result() if rag_future is not None else None
            self.tokenizer, self.model = llm_future.result()
        self.scaler = SimpleTemperatureScaler(self.config)
        print(f"Models ready in {time.perf_counter() - started:.1f}s.")

    def _load_llm(self):
        model_path = self.config.local_model_path or self.config.model_name
        print(f"[LLM] Loading model from: {model_path}...")
        load_kwargs = {} if self.config.local_model_path else {"gguf_file": self.config.gguf_file}
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, **load_kwargs)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            dtype=torch.float16,
            attn_implementation=self.config.attention_implementation,
            **load_kwargs,
        ).to(self.device)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        print("[LLM] Model loaded.")
        return tokenizer, model

    def _load_rag(self):
        print("[RAG] Initializing retriever...")
        try:
            rag = get_rag(config=self.config.rag, device=self.device)
        except RAGRowCountError:
            raise  # a wrong row count is a configuration error, not a reason to run without RAG
        except Exception as exc:
            print(f"[RAG] Warning: Could not initialize RAG retriever: {exc}")
            return None
        print("[RAG] Retriever ready.")
        return rag

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    def build_messages(
        self,
        user_input: str,
        history: Sequence[Sequence[str]] = (),
        rag_context: Optional[str] = None,
        tool_exchange: Sequence[Dict[str, str]] = (),
    ) -> List[Dict[str, str]]:
        messages = []
        system_content = self.config.system_prompt
        if self.config.enable_calculator:
            system_content = f"{system_content} {self.config.calculator_prompt}".strip()
        if system_content:
            messages.append({"role": "system", "content": system_content})
        if self.config.enable_calculator:
            for example in FEW_SHOT_EXAMPLES:
                messages.extend(example)
        for user_message, assistant_message in history:
            messages.append({"role": "user", "content": user_message})
            messages.append({"role": "assistant", "content": assistant_message})
        user_content = f"{rag_context}\n\n{user_input}" if rag_context else user_input
        messages.append({"role": "user", "content": user_content})
        messages.extend(tool_exchange)
        return messages

    def build_prompt(
        self,
        user_input: str,
        history: Sequence[Sequence[str]] = (),
        rag_context: Optional[str] = None,
        tool_exchange: Sequence[Dict[str, str]] = (),
    ) -> str:
        """Render the chat template, dropping the oldest history turns if the prompt is too long."""
        history = list(history)
        budget = self.config.max_length - self.config.max_new_tokens
        while True:
            messages = self.build_messages(user_input, history, rag_context, tool_exchange)
            prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            if not history or len(self.tokenizer.encode(prompt)) <= budget:
                return prompt
            history = history[1:]

    @staticmethod
    def _compile_stop_pattern(stop_sequences: Sequence[str]) -> Optional[re.Pattern]:
        if not stop_sequences:
            return None
        alternatives = "|".join(re.escape(stop) for stop in stop_sequences)
        return re.compile(rf"^[ \t]*(?:{alternatives})", re.MULTILINE)

    def truncate_at_stop_sequences(self, text: str) -> str:
        """Cut the text where the model starts a fake new turn (a stop string at a line start)."""
        if self._stop_pattern is not None:
            for match in self._stop_pattern.finditer(text):
                if match.start() > 0:
                    return text[:match.start()].strip()
        return text.strip()

    # ------------------------------------------------------------------
    # Generation and Best-of-N selection
    # ------------------------------------------------------------------

    def _logits_processors(self, recorder: TokenLogProbRecorder) -> LogitsProcessorList:
        # transformers applies its own temperature/top-k/top-p warpers *after* any
        # custom processors, so we apply them here and put the recorder last. The
        # recorder then scores exactly the distribution the sampler draws from.
        processors = []
        if self.config.do_sample:
            if self.config.temperature != 1.0:
                processors.append(TemperatureLogitsWarper(self.config.temperature))
            if self.config.top_k and self.config.top_k > 0:
                processors.append(TopKLogitsWarper(self.config.top_k))
            if self.config.top_p is not None and self.config.top_p < 1.0:
                processors.append(TopPLogitsWarper(self.config.top_p))
        processors.append(recorder)
        return LogitsProcessorList(processors)

    def generate_best_of_n(self, prompt: str) -> Tuple[str, dict]:
        inputs = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        input_length = inputs.shape[1]
        candidate_count = self.config.best_of_n
        batched_inputs = inputs.repeat(candidate_count, 1)
        recorder = TokenLogProbRecorder(self.scaler)

        with torch.no_grad():
            sequences = self.model.generate(
                batched_inputs,
                attention_mask=torch.ones_like(batched_inputs),
                max_new_tokens=self.config.max_new_tokens,
                do_sample=self.config.do_sample,
                # Warpers are supplied in logits_processor; neutral values here stop
                # generate() from adding a second copy after the recorder.
                temperature=1.0,
                top_k=0,
                top_p=1.0,
                logits_processor=self._logits_processors(recorder),
                repetition_penalty=self.config.repetition_penalty,
                no_repeat_ngram_size=self.config.no_repeat_ngram_size,
                use_cache=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        token_log_probs = recorder.finish(sequences)

        candidates = []
        for index in range(candidate_count):
            generated_tokens = sequences[index, input_length:]
            text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            score = self.scaler.score_token_log_probs(
                token_log_probs[index], generated_tokens, self.tokenizer.pad_token_id
            )
            candidates.append({"index": index + 1, "text": self.truncate_at_stop_sequences(text), **score})

        winner = max(candidates, key=lambda candidate: candidate["score"])
        candidate_run = {
            "accepted_candidate": winner["index"],
            "candidate_count": candidate_count,
            "prompt_tokens": input_length,
            "candidates": [
                {
                    "index": candidate["index"],
                    "score": candidate["score"] if math.isfinite(candidate["score"]) else None,
                    "geom_mean_prob": candidate["geom_mean_prob"] if math.isfinite(candidate["geom_mean_prob"]) else None,
                    "token_count": candidate["token_count"],
                    "selected": candidate["index"] == winner["index"],
                }
                for candidate in candidates
            ],
        }
        return winner["text"], candidate_run

    # ------------------------------------------------------------------
    # Tools
    # ------------------------------------------------------------------

    def retrieve_context(self, question: str) -> Tuple[list, Optional[str]]:
        if self.rag is None:
            return [], None
        try:
            matches = self.rag.retrieve(
                question,
                top_k=self.config.rag.top_k,
                max_distance=self.config.rag.max_cosine_distance,
            )
        except Exception as exc:
            print(f"Warning: RAG retrieval failed: {exc}")
            return [], None
        return matches, (self.rag.format_context(matches) if matches else None)

    def log_calculator_call(self, question: str, expression: str, result=None, error=None, elapsed=0.0):
        """Append a local-only calculator audit record; never added to model messages."""
        record = {
            "timestamp_utc": utc_now(),
            "question": question,
            "expression": expression,
            "success": error is None,
            "result": result,
            "error": error,
            "elapsed_seconds": round(elapsed, 6),
        }
        try:
            with self.calculator_log_path.open("a", encoding="utf-8") as log_file:
                log_file.write(json.dumps(record, ensure_ascii=False) + "\n")
        except OSError as exc:
            print(f"Warning: Could not write calculator audit log: {exc}")

    def run_calculator(self, question: str, expression: str) -> dict:
        started = time.perf_counter()
        try:
            result = calculator.calculate(expression, timeout=self.config.calculator_timeout_seconds)
            error = None
        except Exception as exc:
            result, error = None, str(exc)
        elapsed = time.perf_counter() - started
        self.log_calculator_call(question, expression, result=result, error=error, elapsed=elapsed)
        call = {
            "tool": "calculator",
            "expression": expression,
            "success": error is None,
            "elapsed_ms": round(elapsed * 1000),
        }
        if error is None:
            call["result"] = str(result)
        else:
            call["error"] = error
        return call

    # ------------------------------------------------------------------
    # One full turn
    # ------------------------------------------------------------------

    def respond(self, question: str, history: Sequence[Sequence[str]] = ()) -> Tuple[str, dict]:
        """Answer one user message. Thread-safe: turns run one at a time on the GPU."""
        with self._generation_lock:
            return self._respond(question, list(history))

    def _respond(self, question: str, history: list) -> Tuple[str, dict]:
        started_at = utc_now()
        started = time.perf_counter()
        rag_matches, rag_context = self.retrieve_context(question)

        tool_exchange: List[Dict[str, str]] = []
        tool_calls: List[dict] = []
        generation_runs: List[dict] = []

        def generate(stage: str) -> str:
            prompt = self.build_prompt(question, history, rag_context, tool_exchange)
            text, run = self.generate_best_of_n(prompt)
            run["stage"] = stage
            generation_runs.append(run)
            return text

        response = generate("initial")
        rounds = self.config.max_calculator_rounds if self.config.enable_calculator else 0
        for attempt in range(rounds):
            match = TOOL_PATTERN.search(response)
            if match is None:
                break  # the model decided no calculation was needed
            expression = match.group(1).strip()
            call = self.run_calculator(question, expression)
            tool_calls.append(call)
            # Keep only the model's text up to its tool call; anything after it was
            # written without the result.
            tool_exchange.append({"role": "assistant", "content": response[:match.end()].strip()})
            if call["success"]:
                feedback = (
                    f"Calculator result for {expression}: {call['result']}\n"
                    "Answer the question using this result."
                )
                stage = "final_answer"
            else:
                feedback = (
                    f"Calculator error for {expression}: {call['error']}\n"
                    "If a numeric calculation is really needed, retry with one valid math "
                    "expression inside <calculator>...</calculator>. Otherwise answer the "
                    "question directly without the calculator."
                )
                stage = f"calculator_repair_{attempt + 1}"
            tool_exchange.append({"role": "user", "content": feedback})
            response = generate(stage)

        if TOOL_PATTERN.search(response) or _OPEN_TOOL_TAG.search(response):
            # The model asked for yet another calculation after the round limit.
            if tool_calls and not tool_calls[-1]["success"]:
                response = (
                    "I couldn't verify a numerical answer because the calculator request "
                    f"failed: {tool_calls[-1]['error']}"
                )
            else:
                response = _OPEN_TOOL_TAG.sub("", TOOL_PATTERN.sub("", response)).strip()
                if not response and tool_calls:
                    response = f"The result is {tool_calls[-1]['result']}."
        response = strip_think_tags(response)

        accepted_candidate = None
        if generation_runs:
            accepted = generation_runs[-1]
            selected = next((c for c in accepted["candidates"] if c["selected"]), None)
            if selected:
                accepted_candidate = {
                    "stage": accepted["stage"],
                    **{key: selected[key] for key in ("index", "score", "geom_mean_prob", "token_count")},
                }

        metadata = {
            "timestamp_utc": started_at,
            "duration_seconds": round(time.perf_counter() - started, 6),
            "model_name": self.config.local_model_path or self.config.model_name,
            "gguf_file": None if self.config.local_model_path else self.config.gguf_file,
            "attention_implementation": self.config.attention_implementation,
            "tool_used": bool(tool_calls),
            "tool_calls": tool_calls,
            "rag_used": bool(rag_matches),
            "rag_matches": rag_matches,
            "max_length": self.config.max_length,
            "max_new_tokens": self.config.max_new_tokens,
            "best_of_n": self.config.best_of_n,
            "accepted_candidate": accepted_candidate,
            "candidate_runs": generation_runs,
            "generation": {
                "temperature": self.config.temperature,
                "calibration_temperature": self.config.calibration_temperature,
                "top_k": self.config.top_k,
                "top_p": self.config.top_p,
                "do_sample": self.config.do_sample,
                "repetition_penalty": self.config.repetition_penalty,
                "no_repeat_ngram_size": self.config.no_repeat_ngram_size,
            },
        }
        return response, metadata
