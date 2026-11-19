from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class TTSConfig:
    """Configuration for the Kokoro TTS tool."""

    enabled: bool = True
    default_voice: str = "af_heart"
    speed: float = 1.0


@dataclass
class RAGConfig:
    """Configuration for the RAG retriever tool (single source of truth for RAG settings)."""

    max_cosine_distance: float = 0.75  # Only inject matches with D_cos <= this value
    top_k: int = 3  # Maximum number of reference QA pairs injected into the prompt
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    dataset_name: str = "FreedomIntelligence/RAG-Instruct"
    dataset_config: Optional[str] = None  # the dataset has only the default config
    dataset_split: str = "train"  # the only split (~40.5K rows)
    # Rows to index (the split has 40,541). Indexing every long answer is slow for
    # MiniLM, so only the first 12K are used. They are streamed from the Hub's
    # Parquet export; only the needed row groups are fetched and nothing is stored.
    max_rows: int = 12_000
    max_context_chars: int = 1_200  # Per retrieved answer injected into the prompt; RAG-Instruct answers are long
    cache_dir: Path = field(
        default_factory=lambda: Path(__file__).resolve().parent / "tools" / "rag_cache"
    )


@dataclass
class ChatConfig:
    """Configuration shared by the CLI and the web chat."""

    model_name: str = "Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF"
    gguf_file: str = "qwen2.5-coder-1.5b-instruct-q4_k_m.gguf"
    local_model_path: str = r""  # If set, uses local model path instead of downloading from HF
    attention_implementation: str = "sdpa"  # PyTorch fused attention when supported by CUDA hardware
    system_prompt: str = (
        "You are a careful assistant. Answer the user's actual request directly and stay "
        "on topic. Do not introduce unrelated calculations. Use inline LaTeX for any "
        "equations you include. When you provide code, format it as a fenced Markdown "
        "code block and include a language label when appropriate."
    )
    calculator_prompt: str = (
        "You have a calculator tool. Decide for yourself whether it is needed: use it only "
        "when answering requires computing a numeric result. Do not use it for questions "
        "that merely mention numbers, years, versions, or names such as GPT-3, COVID-19, "
        "or Windows 11. To use it, reply with only <calculator>expression</calculator> and "
        "wait for the result. The expression must be pure math: +, -, *, /, //, %, "
        "parentheses, ** or ^; constants pi, e, tau; functions sqrt, log, pow, max, min, "
        "abs, sin, cos, tan, exp, ceil, floor, round, sum, prod, mean; calculus functions "
        "diff, integrate, limit, approx. Trigonometric functions use radians. Variables are "
        "only allowed inside calculus functions. No prose, assignments, or Python code. "
        "Treat the returned result as authoritative and answer concisely with units."
    )
    enable_calculator: bool = True
    calculator_timeout_seconds: float = 2.0  # Compute time allowed per calculator expression
    max_calculator_rounds: int = 2  # Calculator calls allowed per response (includes repairs)
    max_length: int = 8192
    max_new_tokens: int = 2048
    temperature: float = 0.7  # Generation temperature for sampling
    calibration_temperature: float = 1.143  # Temperature for confidence calibration (T_gen * T_calib = 0.8)
    do_sample: bool = True
    top_k: int = 40
    top_p: float = 0.9
    repetition_penalty: float = 1.15
    no_repeat_ngram_size: int = 0  # 0 disables; values like 3 break code and math, which repeat n-grams
    best_of_n: int = 4  # Number of candidate responses to sample and score per turn
    show_candidate_scores: bool = True  # Display candidate scores and ranking in CLI output
    enable_rag: bool = True  # Automatically retrieve relevant Wikipedia QA context
    rag: RAGConfig = field(default_factory=RAGConfig)
    # Fallback stop strings, matched only at the start of a line. Qwen normally ends
    # with its <|im_end|> EOS token; these catch a model that starts a fake new turn.
    stop_sequences: List[str] = field(default_factory=lambda: [
        "User:",
        "Human:",
        "Assistant:",
        "Instruction:",
        "### Instruction:",
        "### Response:",
    ])
