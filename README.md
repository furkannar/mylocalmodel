# mylocalmodel

A CUDA-based chat application for Qwen instruct models, with both a command-line interface and a browser-based chat interface. Each turn samples multiple responses, ranks them by temperature-scaled, length-normalized token log-likelihood, and returns the highest-scoring response.

## Features

- Uses the Qwen chat template for system prompts and multi-turn conversation history.
- Generates a configurable Best-of-N batch of candidate responses.
- Scores candidates using the mean log-probability of generated tokens after temperature scaling. The geometric mean token probability is also shown as a percentage.
- Optionally prints scores for all candidates and marks the selected response.
- Can load a model from Hugging Face or from a local model directory.
- Drops the oldest conversation turns when the prompt exceeds the configured context budget.
- Includes a restricted calculator tool for arithmetic, numeric functions, and selected symbolic-calculus operations. **The model decides when to use it**; no keyword detector forces a tool call.
- **Persistent calculator worker process**, started once with the app and reused for every call, with a hard 2-second compute timeout per expression. A hung calculation kills and replaces the worker.
- **AST-based expression validation** with depth, node count, size, and exponent limits to mitigate unsafe or pathological inputs.
- **Repetition penalty** and optional n-gram blocking (`no_repeat_ngram_size`, off by default because it breaks code and math).
- **Separate generation and calibration temperatures** for independent control of sampling diversity and ranking confidence.
- **Memory-light candidate scoring**: per-token log-probabilities are recorded during generation instead of keeping the full `[N, steps, vocabulary]` score tensor on the GPU.
- **Parallel startup**: the LLM, the RAG retriever, and (in the web app) the TTS model download and load at the same time. Missing files are reported with their sizes before downloading, with a warning if the disk looks too full.
- One shared chat engine (`engine.py`) used by both the CLI and the web app, so both behave the same.
- Provides a browser chat with persistent conversation history, per-turn generation and tool metadata, and controls to reopen or delete conversations.
- Renders assistant Markdown, fenced code blocks, and inline or display LaTeX in the browser chat. Common unwrapped commands such as `\frac{1}{2}` are automatically given inline math delimiters; generated HTML is sanitized before display.
- **Automated RAG knowledge grounding**: Retrieves Wikipedia-grounded question-answer context from the first 12,000 rows of [`FreedomIntelligence/RAG-Instruct`](https://huggingface.co/datasets/FreedomIntelligence/RAG-Instruct), streamed (never fully downloaded), using `sentence-transformers/all-MiniLM-L6-v2` dense embeddings and cosine distance filtering ($D_{\cos} \le 0.75$), with the embedding index cached in `tools/rag_cache/`.
- Renders collapsible RAG reference cards in the browser chat showing retrieved Wikipedia QA pairs, question, answer, cosine distance, and similarity score.
- **Text-to-Speech (TTS)** integration using Kokoro ONNX for streaming audio playback in the web interface.
- Keeps conversation contents, calculator audit logs, and RAG cache indices in local files excluded from Git.

## Examples

### Calculator example

For a jacket that costs $80, has a 25% discount, and then 8% sales tax, the calculation is:

```text
<calculator>80 * (1 - 25 / 100) * (1 + 8 / 100)</calculator>
Calculator result: 64.8
```

The final price is **$64.80**. This is the calculator format used in the system's few-shot example.

## Requirements

- Python 3.9 or newer.
- A CUDA-enabled NVIDIA GPU and a PyTorch build with CUDA support. The application exits with an error if CUDA is unavailable.
- Enough GPU memory for the selected model and the batched candidates.

> [!IMPORTANT]
> The default `q4_k_m` GGUF file is loaded through Transformers and dequantized into a float16 PyTorch model for generation and scoring. Its quantized file size does not represent its runtime VRAM use; expect substantially higher GPU memory use than the GGUF file size. This is not llama.cpp inference and does not keep the model in its packed GGUF format.

Python dependencies are listed in `requirements.txt`:

- `torch`
- `transformers` (4.56 or newer)
- `sentencepiece`
- `tiktoken`
- `sympy` (symbolic calculator operations)
- `gguf` (loading GGUF weights through Transformers)
- `fastapi` and `uvicorn` (web interface)
- `datasets` (streaming the RAG dataset from the Hugging Face Hub)
- `sentence-transformers` (dense vector embeddings for semantic retrieval)
- `tqdm` (progress tracking for embedding calculation)
- `kokoro-onnx>=0.6.0` (text-to-speech; version 0.6.0+ required for speed parameter fix)
- `soundfile` (audio file handling)

Install a CUDA-compatible PyTorch build as described in the setup steps below.

## Setup and run

Clone the repository and enter its directory:

```bash
git clone https://github.com/TheOfficialFurkanNar/mylocalmodel.git
cd mylocalmodel
```

Create and activate a virtual environment:

```bash
python -m venv .venv
```

Activate the environment, install the CUDA-compatible PyTorch build using the command from the [official PyTorch installation selector](https://pytorch.org/get-started/locally/), then install the remaining project requirements. The web interface dependencies are included in `requirements.txt`.

```bash
# Windows PowerShell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# macOS / Linux
source .venv/bin/activate
pip install -r requirements.txt
```

Start the command-line chat:

```bash
python run.py
```

The default model is `Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF`, using its `q4_k_m` file. Transformers downloads the model from Hugging Face on the first run unless `local_model_path` is set in `config.py` to a local Transformers model directory. Type `quit` or `exit` to end the session.

To use the web chat with the conversation sidebar, start the API and UI with `python web_backend.py`, then open `http://127.0.0.1:8000` in your browser. The server listens on `127.0.0.1` only, so other machines on your network cannot read or delete your conversations. To expose it deliberately, set `LOCALQWEN_HOST=0.0.0.0` (and optionally `LOCALQWEN_PORT`) before starting it.

### First run and downloads

At startup the app checks which files are already on disk and prints what is missing, for example:

```text
Downloading 3 missing item(s) in parallel:
  - LLM Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf (~1.04 GB)
  - Embedding model sentence-transformers/all-MiniLM-L6-v2 (~0.10 GB)
  - RAG dataset FreedomIntelligence/RAG-Instruct, first 12000 rows (to build the index) (streamed, not stored on disk)
```

The LLM and the RAG retriever then download and load concurrently; in the web app the Kokoro TTS files do too, without delaying server startup. If a drive has less free space than the downloads need (plus 10%), a warning suggests freeing space or pointing `HF_HOME` at another drive. Files that are already present are never downloaded again. Hugging Face models go to the Hugging Face cache (`HF_HOME`, by default `~/.cache/huggingface`), TTS files to `tools/tts_cache/`, and the RAG index to `tools/rag_cache/`.

## Configuration

Edit the defaults in `config.py` (`ChatConfig`) before launching. Common settings include:

| Setting | Default                                   | Purpose |
| --- |-------------------------------------------| --- |
| `model_name` | `Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF`   | Hugging Face model repository |
| `gguf_file` | `qwen2.5-coder-1.5b-instruct-q4_k_m.gguf` | GGUF quantization file loaded from the repository |
| `local_model_path` | empty                                     | Local Transformers model directory; takes precedence over the Hub model and GGUF file |
| `attention_implementation` | `sdpa`                                    | PyTorch scaled dot product attention; uses a fused CUDA attention kernel when supported |
| `best_of_n` | `4`                                       | Number of candidates generated in one batch |
| `max_length` | `8192`                                    | Prompt context budget, including generation allowance |
| `max_new_tokens` | `2048`                                    | Maximum tokens generated per candidate |
| `temperature` | `0.7`                                     | Sampling temperature used during generation |
| `calibration_temperature` | `1.143`                                   | Temperature for scoring (T_gen × T_calib = 0.8 effective) |
| `top_k` / `top_p` | `40` / `0.9`                              | Sampling filters |
| `repetition_penalty` | `1.15`                                    | Penalty factor to reduce repetitive token generation |
| `no_repeat_ngram_size` | `0`                                       | Blocks repeated n-grams when > 0. Off by default: code and math legitimately repeat short token sequences |
| `show_candidate_scores` | `True`                                    | Whether to print candidate scores and token counts |
| `system_prompt` | concise general instructions              | Keeps answers on topic, uses inline LaTeX for equations, and asks for fenced code blocks |
| `enable_calculator` | `True`                                    | Offers the calculator tool to the model on every turn |
| `calculator_prompt` | calculator syntax and usage instructions  | Explains when (and when not) to use the calculator and its syntax |
| `calculator_timeout_seconds` | `2.0`                                     | Compute time allowed per calculator expression (worker startup is not counted) |
| `max_calculator_rounds` | `2`                                       | Calculator calls allowed per response, including repair attempts |
| `enable_rag` | `True`                                    | Whether to automatically retrieve Wikipedia reference QA pairs before generation |
| `rag` | `RAGConfig()`                             | RAG settings, see below |
| `stop_sequences` | `User:`, `Assistant:`, …                  | Fallback stop strings, only matched at the start of a line |

`RAGConfig` holds all retrieval settings:

| Setting | Default | Purpose |
| --- | --- | --- |
| `dataset_name` | `FreedomIntelligence/RAG-Instruct` | Hugging Face dataset with `question` and `answer` columns |
| `dataset_split` | `train` | Split to stream (RAG-Instruct only has `train`, 40,541 rows) |
| `max_rows` | `12000` | Number of rows streamed and indexed; startup raises `RuntimeError` if a different number of rows is loaded |
| `embedding_model_name` | `sentence-transformers/all-MiniLM-L6-v2` | Embedding model for the index and queries |
| `max_cosine_distance` | `0.75` | Only matches with $D_{\cos} \le$ this value are injected |
| `top_k` | `3` | Maximum number of matches injected into the prompt |
| `max_context_chars` | `1200` | Each injected answer is cut to this length (RAG-Instruct answers are long) |
| `cache_dir` | `tools/rag_cache` | Where the embedding index is cached |

Lower `best_of_n` or `max_new_tokens` to reduce GPU memory use and generation time. Increasing `best_of_n` samples more candidates, but does not guarantee a better answer; selection relies on the model's own token likelihoods. The `sdpa` attention setting uses PyTorch's scaled dot product attention, which can dispatch to its fused Flash Attention CUDA kernel when the GPU, dtype, and installed PyTorch build support it; otherwise PyTorch selects another supported SDPA kernel. This is distinct from Dao-AILab's standalone FlashAttention-2 package, whose maintainers describe Windows support as experimental.

## Calculator tool

The calculator is offered to the model on every turn, and **the model decides whether to use it**. Earlier versions used a keyword detector that forced a calculator call whenever a message contained a digit and a symbol such as `-`, so questions like "What is GPT-3?" or "How does COVID-19 spread?" were wrongly treated as math. That detector and the forced retry are gone.

When the selected response contains `<calculator>expression</calculator>`, the app evaluates the expression and adds the exchange to the prompt as real conversation turns: the model's tool call as an assistant message, then the result (or error) as a user message. The model then writes its answer. RAG context stays in the prompt during this step. The tag must contain one math expression only; assignments, `print` calls, and other Python statements are rejected. Names that are not math, such as `GPT` in `GPT-3`, are rejected too: variables are only allowed inside `diff`, `integrate`, and `limit`.

If evaluation fails, the model sees the error and can either retry with a valid expression or answer directly if no calculation is actually needed. Calculator requests are limited to `max_calculator_rounds` (default 2) per response. If the model still asks for a calculation after a failed final attempt, the app returns an explicit failure message instead of an unverified number. Float results are rounded to 12 significant digits, so the model sees `64.8` rather than `64.80000000000001`.

### Calculator prompting

The system prompt explains when to use the calculator (only when a numeric result must be computed, not when a question merely mentions numbers, years, versions, or names) and its syntax. Short few-shot examples show a calculator call, the returned result, and a concise answer, plus one example that answers directly without the tool. A valid expression can still encode the wrong interpretation of a word problem, so the calculator verifies arithmetic for the supplied expression but not that the expression matches the question.

**Known limitation:** the 1.5B model sometimes copies numbers from the few-shot examples into calculator calls for unrelated questions, and sometimes skips the calculator and does arithmetic itself. Check calculator calls in the web UI's tool cards or in `calculator_calls.jsonl`.

## Conversation history

The web app saves conversations locally in `conversation_history.json` and shows them in the left sidebar. Select a conversation to reopen it, start a new conversation, or use its × button to delete it after confirming. Each turn stores the user message, assistant response, UTC timestamp, total response duration, calculator usage and results, model and generation settings, the accepted Best-of-N candidate, and score metadata for each candidate run. Metadata is stored separately from message text and is not added to the model's conversation prompt. The conversation file is ignored by Git because it contains local chat content. The command-line interface keeps its conversation history in memory for the current run only.

## Calculator safeguards

The calculator runs in a **separate, persistent worker process** (`tools/calculator_worker.py`). It starts when the app starts (while the models load), imports SymPy once, and is reused for every call, so a calculation typically returns in milliseconds. Each expression gets a hard compute timeout (`calculator_timeout_seconds`, default 2 seconds). If it is exceeded, the worker is killed and a replacement starts immediately. The worker imports only the calculator module, not torch or the other tools. These are resource controls, not an operating-system security sandbox. Expression evaluation uses **AST parsing** with limits intended to mitigate unsafe or pathological inputs:

- Maximum expression length: 512 characters
- Maximum AST node count: 100 operations
- Maximum AST depth: 20 nesting levels
- Maximum exponent magnitude: 10,000, enforced for both `^`/`**` and `pow()`
- Maximum integer size: 100,000 bits, checked *before* a power is computed
- Maximum output length: 2,000 characters

These restrictions mitigate risk while supporting practical mathematical operations. The calculator does not use Python `eval`; it evaluates a restricted subset of operations through a custom AST walker.

Each calculator call is also appended to `calculator_calls.jsonl` in the project directory. Records include the UTC timestamp, original question, expression, success or failure, result or error, and elapsed time. This local audit log is not inserted into the model prompt or conversation history and is ignored by Git.

The calculator parses a restricted Python-like math syntax; it does not call Python `eval`. It supports `+`, `-`, `*`, `/`, `//`, `%`, parentheses, `^` as exponentiation, and the constants `pi`, `e`, and `tau`. Trigonometric functions use radians. Numeric functions include `sqrt`, `log`, `pow`, `max`, `min`, `abs`, `sin`, `cos`, `tan`, `exp`, `ceil`, `floor`, and `round`; aggregate functions include `sum`, `prod`, and `mean`.

Selected SymPy operations are also available:

| Operation | Example |
| --- | --- |
| Derivative | `diff(x^3, x)` |
| Indefinite integral | `integrate(sin(x), x)` |
| Definite integral | `integrate(x^2, x, 0, 1)` |
| Limit | `limit(sin(x)/x, x, 0)` |
| Numeric approximation | `approx(sqrt(2), 30)` |

Inputs are restricted by expression length, syntax depth, exponent size, and result size. Symbolic evaluation runs in a separate process that is terminated when the time limit expires. These safeguards reduce risk but do not make every symbolic problem easy or guarantee that a returned mathematical result is correct; review important results.

## Retrieval-augmented generation (RAG)

When `enable_rag` is `True` (the default), each user message is first used as a query against a local retrieval index built from [`FreedomIntelligence/RAG-Instruct`](https://huggingface.co/datasets/FreedomIntelligence/RAG-Instruct). This dataset of Wikipedia-grounded questions with GPT-4o-written answers is far broader than the small dataset used previously.

### Building the index (streamed, not downloaded)

The dataset has one split (`train`, 40,541 rows; the original file is a 296 MB JSON). Only the first `max_rows` (default 12,000) are used, because embedding every long answer would be slow for MiniLM. Instead of downloading the dataset, the app **streams** those rows:

- It reads the Hub's automatic Parquet export (`revision="refs/convert/parquet"`). That file is stored in row groups of 1,000 rows, so streaming 12,000 rows fetches only the first 12 of 41 row groups.
- Only the `question` and `answer` columns are read. The large `documents` column (about 10 Wikipedia passages per row) is never transferred.
- Nothing is written to the Hugging Face dataset cache. If the Parquet export is unavailable, the app falls back to streaming the original JSON file, which is transferred in full but still not stored.

The console prints how many rows were loaded, for example `Loaded 12000 rows from FreedomIntelligence/RAG-Instruct (requested 12000).`. If the number differs from `max_rows` (for example because `max_rows` is larger than the split), startup stops with a `RuntimeError` (`RAGRowCountError`). This is the only RAG failure that stops the app.

Each `Question: …\nAnswer: …` pair is embedded once with `sentence-transformers/all-MiniLM-L6-v2` (384-dimensional, L2-normalized; MiniLM reads the first 256 tokens, so the question and the start of the answer drive retrieval). The index is cached in `tools/rag_cache/rag_qa_index.pt`, about 20 MB for 12,000 rows. The cache records the dataset, split, `max_rows`, and embedding model. If any of these change, the index is rebuilt automatically. It is loaded with `torch.load(..., weights_only=True)`.

### Retrieval

At query time, the user message is embedded with the same model and compared to the index by cosine distance ($D_{\cos} = 1 - \text{cosine similarity}$). Matches with $D_{\cos} \le$ `max_cosine_distance` (default `0.75`) are kept and sorted by distance ascending. The top `top_k` (default `3`) are formatted as a `[Wikipedia QA Reference Context]` block and prepended to the user message inside the chat prompt. Each answer is cut to `max_context_chars` (default 1,200) at a word boundary, so a few long answers cannot crowd out the conversation history. Retrieved context is treated as reference material only. It is not inserted into conversation history, and the model is not explicitly instructed to cite or obey it. The 0.75 threshold was chosen for the previous, shorter dataset and may need tuning.

In the web interface, each assistant turn's metadata records whether RAG was used (`rag_used`) and the retrieved matches (question, answer, distance, similarity), rendered as collapsible reference cards. The RAG index cache and embedding model download are local-only artifacts and are excluded from Git.

RAG retrieval runs once per user turn, before the Best-of-N generation batch, so its latency is paid once regardless of `best_of_n`. If the embedding model or dataset cannot be loaded (for example offline on a first run), the application logs a warning and continues without RAG rather than failing.

## Text-to-Speech (TTS) tool

The web interface includes text-to-speech functionality using **Kokoro ONNX**, a lightweight ONNX-based TTS engine that runs entirely on CPU (no CUDA required). The TTS system streams audio as the response is generated, providing real-time speech playback in the browser.

### How TTS works in this application

The TTS system operates as follows:

1. **Text Processing**: When a response is generated, the text is split into sentences or natural phrase boundaries for more natural-sounding speech.

2. **ONNX Inference**: Kokoro ONNX uses a neural network model trained on speech data to generate audio waveforms from text. The model runs on CPU using ONNX Runtime, making it accessible without CUDA requirements.

3. **Streaming Architecture**: Instead of generating the entire audio file upfront, the system streams audio chunks as they are synthesized. This is implemented using Server-Sent Events (SSE), allowing the browser to begin playback almost immediately while synthesis continues in the background.

4. **Voice Selection**: Different voice embeddings (stored in `voices-v1.0.bin`) modify the base model's output to produce distinct speaking styles, accents, and gender characteristics.

5. **Audio Encoding**: Generated audio samples are encoded as 16-bit PCM WAV format at 24 kHz sample rate, providing high-quality speech suitable for web playback.

### Features

- **Streaming synthesis**: Audio is generated and streamed sentence-by-sentence using Server-Sent Events (SSE)
- **Multiple voices**: 11 built-in voices (7 American English, 4 British English) with different accents and genders
- **Automatic model download**: Missing model files are downloaded from GitHub releases when the web app starts, in parallel with the LLM and RAG loading, and cached locally
- **No external dependencies**: Uses ONNX Runtime (CPU-only), no espeak-ng or other system dependencies required
- **24 kHz audio**: High-quality 16-bit PCM WAV output at 24 kHz sample rate

### Available voices

| Voice ID | Name | Gender | Accent | Language |
| --- | --- | --- | --- | --- |
| `af_heart` | Heart | Female | US | en-us |
| `af_sky` | Sky | Female | US | en-us |
| `af_bella` | Bella | Female | US | en-us |
| `af_sarah` | Sarah | Female | US | en-us |
| `af_nicole` | Nicole | Female | US | en-us |
| `am_adam` | Adam | Male | US | en-us |
| `am_michael` | Michael | Male | US | en-us |
| `bf_emma` | Emma | Female | UK | en-gb |
| `bf_isabella` | Isabella | Female | UK | en-gb |
| `bm_george` | George | Male | UK | en-gb |
| `bm_lewis` | Lewis | Male | UK | en-gb |

### Configuration

TTS settings are configured in `config.py` (`TTSConfig`):

| Setting | Default | Purpose |
| --- | --- | --- |
| `enabled` | `True` | Enable/disable TTS functionality |
| `default_voice` | `af_heart` | Default voice for synthesis |
| `speed` | `1.0` | Speech speed (0.5 to 2.0) |

### API endpoints

- `GET /tts/voices` - Returns the list of available voices
- `POST /tts/stream` with JSON `{"text": "...", "voice": "af_heart"}` - Streams synthesized audio as SSE events. The web UI uses this endpoint, so long answers are not limited by URL length.
- `GET /tts/stream?text=<text>&voice=<voice>` - Same stream for short texts and manual testing

An unknown voice or empty text returns HTTP 400. The web interface reads the stream with `fetch` and plays the chunks in order using the Web Audio API. Users can select different voices from the UI.

### Model files

Kokoro TTS model files are stored in `tools/tts_cache/`:
- `kokoro-v1.0.onnx` - The ONNX model (~326 MB)
- `voices-v1.0.bin` - Voice embeddings (~28 MB)

These files are downloaded from GitHub releases when the web app starts (in parallel with each other and with the LLM and RAG loading) and cached for subsequent runs. Each download is written to a `.part` file and renamed only when complete, so an interrupted download is retried instead of being mistaken for a finished file. The web app also loads Kokoro in the background at startup, so the first read-aloud does not wait for it.

### Version requirement

The TTS functionality requires `kokoro-onnx>=0.6.0`. Earlier versions (0.4.7-0.5.0) have a bug where the speed parameter is incorrectly passed as `int32` instead of `float32`, causing ONNX Runtime errors with newer model exports.

## Project files

- `engine.py` — shared chat engine: parallel model loading, prompt construction, Best-of-N generation and selection, RAG retrieval, and the calculator flow. Used by both the CLI and the web app; turns run one at a time on the GPU.
- `run.py` — interactive command-line chat on top of the engine.
- `web_backend.py` — FastAPI chat API, local conversation persistence, and TTS streaming on top of the engine.
- `model_files.py` — startup check of which model files are missing, their sizes, and free disk space.
- `static/` — web chat UI with conversation history and per-turn calculator/candidate metadata.
- `config.py` — model, generation, scoring, calculator, RAG, TTS, and stop-sequence settings.
- `calibration.py` — temperature scaling, length-normalized candidate scoring, and the logits processor that records token log-probabilities during generation.
- `tools/calculator.py` — restricted arithmetic and SymPy calculus evaluator with AST-based validation, plus the persistent worker manager.
- `tools/calculator_worker.py` — the long-lived calculator worker process.
- `tools/rag.py` — streams the RAG dataset, builds and caches the embedding index, and retrieves matches.
- `tools/tts.py` — Kokoro ONNX text-to-speech engine with streaming synthesis and voice management.
- `requirements.txt` — Python dependencies.
- `LICENSE` — MIT License.

## How candidate selection works

For each sampled candidate, let the prompt be $x$, and let its generated token sequence be $y_1, \ldots, y_L$. At generation step $t$, the model produces a vocabulary-sized score vector $z_t \in \mathbb{R}^{|V|}$. In this implementation, $z_t$ is the score vector the sampler draws from, after generation-time processing: repetition penalty, then sampling temperature, then top-k/top-p filtering. A logits processor (`TokenLogProbRecorder` in `calibration.py`) runs last in that chain and records each sampled token's log-probability step by step, keeping only one `[N, vocabulary]` row in memory. (Transformers applies its own temperature/top-k/top-p after custom processors, so the engine applies those three itself to keep the recorder last.) This gives the same scores as keeping all per-step scores, which was verified to match exactly, without holding a multi-gigabyte tensor on the GPU. The candidate scorer applies a second temperature scaling, using the configured calibration temperature $T_{\mathrm{calib}} > 0$:

$$
p_t^{(T)}(v) = \mathrm{softmax}\!\left(\frac{z_t}{T_{\mathrm{calib}}}\right)_v
= \frac{\exp(z_{t,v}/T_{\mathrm{calib}})}
       {\sum_{u \in V}\exp(z_{t,u}/T_{\mathrm{calib}})},
\qquad v \in V.
$$

Here, $p_t^{(T)}(y_t)$ is the probability assigned at step $t$ to the token that the candidate actually generated. The code sums the log of those probabilities and divides by the number of unmasked generated tokens. If $M$ is the set of generated positions whose token ID is not the tokenizer's pad token ID, the implemented score is:

$$
S(x,y) = \frac{1}{|M|}\sum_{t \in M}\log p_t^{(T)}(y_t)
       = \frac{1}{|M|}\sum_{t \in M}
         \log\!\left(
           \frac{\exp(z_{t,y_t}/T_{\mathrm{calib}})}
                {\sum_{v \in V}\exp(z_{t,v}/T_{\mathrm{calib}})}
         \right).
$$

Among the $N$ candidates, the selected response is:

$$
y^* = \arg\max_{i=1,\ldots,N} S(x,y^{(i)}).
$$

The division by $|M|$ makes this a mean log-probability per scored token rather than a total log-probability, reducing the automatic preference for shorter sequences that a sum would create. Exponentiating the mean gives the geometric mean of the selected-token probabilities:

$$
G(x,y) = \exp(S(x,y))
        = \left(\prod_{t \in M}p_t^{(T)}(y_t)\right)^{1/|M|}.
$$

This value is what the CLI displays as `geom_mean_prob` (as a percentage). Higher $T_{\mathrm{calib}}$ softens the distribution and lower values make it peakier; it changes the scores and can change the ranking. Because $z_t$ already reflects generation-time processing, this score is specifically the likelihood under the processed generation scores after calibration scaling, not an independent evaluation using the model's unmodified logits. It is a model-based ranking signal, not a calibrated measure of factual correctness or answer quality. The score mask excludes every generated token whose ID equals `pad_token_id`. If a tokenizer uses the same ID for padding and EOS, the EOS token is therefore excluded as well. In the configured Qwen2.5-Coder-1.5B-Instruct tokenizer, padding is `<|endoftext|>` and the tokenizer's EOS is `<|im_end|>`, so the normal EOS token is distinct and remains part of the score; `<|endoftext|>` tokens are masked.

## Limitations and example session

Best-of-N selection ranks candidates using model token likelihoods, not factual verification. A high-scoring candidate can still contain errors, as in this recorded web-chat session from `conversation_history.json`. Candidate score output is omitted; the following is the user and assistant message text, with selected metadata noted.

![Web chat output for the Abraham Lincoln beard question](exampleoutput.png)

~~~text
You: Who suggested Abraham Lincoln grow a beard?

AI: Grace Bedell suggested Abraham Lincoln to grow a Beard.

You: What is the study of beetles called?

AI: The study of Beetles is called colepterology.

You: Why does this field of study matter

AI: Colepterologists study beetles because they help us understand their behavior, ecology, and evolution. They also use beetle research to develop new tools and techniques in various fields such as agriculture, medicine, and environmental science.

You: How does it contribute to our lives

AI: Beetle research contributes to our life by providing valuable information about the natural world and its inhabitants. It helps us better understand how different species interact with each other and how they adapt to changing environments. Additionally, beetle research can be used to develop sustainable agricultural practices that reduce pesticide usage and conserve biodiversity.
~~~

This session was recorded with the previous RAG dataset (`rag-datasets/rag-mini-wikipedia`). RAG supplied a near-exact Wikipedia QA match for the first two questions (cosine distances `0.1749` and `0.1622`), and both factual answers are essentially correct. The last two questions retrieved no context within the distance threshold and were answered from model parameters alone.

**Accuracy note:** The second answer misspells the field's name — the correct term is **coleopterology** (the model consistently writes "colepterology"/"Colepterologists" in later turns). The first answer is correct: 11-year-old Grace Bedell did suggest Lincoln grow a beard. This illustrates that RAG grounding and high candidate scores improve factual answers but do not guarantee correct spelling, terminology, or reasoning.

## Notes

- The implementation loads the full model onto the CUDA device; it does not currently configure quantization, CPU offloading, or automatic device mapping.
- EOS ends generation normally. Configured stop strings are a fallback: the decoded output is cut where one appears at the start of a line (for example a fake `User:` turn), so ordinary text such as "the user: …" is not truncated. Generation itself may continue beyond a stop string until EOS or the token limit.
- The web server handles chat requests in a worker thread, so a long generation does not block TTS streaming or `/health`. Errors are returned as HTTP 500 with a `detail` message, and requests that arrive while the model is still loading get HTTP 503.

## AI assistance disclosure

Some commits in this repository carry a `Co-Authored-By: Claude …` line. This section explains what that means and what it does not.

**Scope.** Claude (Anthropic's Claude Opus 5.5, used through the Claude Code CLI) helped with one round of changes, merged as PR #11 (commit `e4740c4`). The project's design and its earlier implementation (the Best-of-N sampler with calibrated scoring, the restricted calculator, RAG, Kokoro TTS, and the web UI) predate that work. See the Git history for details.

**What the maintainer did.** The work was directed and reviewed by the maintainer, not generated unattended:

- Asked for a review of the codebase and chose which findings to act on.
- Found and reported the problems that drove the changes: the calculator timeouts, and calculator calls on non-math questions such as "GPT-3".
- Made the design decisions:
  - let the model decide when to use the calculator, instead of a keyword detector;
  - load missing models concurrently;
  - replace the RAG dataset with `FreedomIntelligence/RAG-Instruct`;
  - cap it at 12,000 rows, with a hard error if the loaded row count differs;
  - stream the dataset rather than download it.
- Stopped runs that conflicted with local constraints (limited disk space).
- Tested the result locally, then reviewed and merged the pull request.

**What Claude did.** Within those decisions, Claude:

- Analyzed the code and found the root causes. For example, the calculator subprocess re-imported torch and sentence-transformers on every call.
- Wrote the implementation: the shared `engine.py`, the persistent calculator worker, memory-light scoring, parallel startup, streamed RAG loading, and the web fixes.
- Ran the verification checks described in the PR, such as calculator timing and scoring equivalence on the GPU.
- Updated this README.

**Responsibility.** All AI-assisted changes were reviewed before merging, and the maintainer remains responsible for the code. Known limitations, including those of the AI-assisted parts (see the calculator's *Known limitation* note), are documented above rather than hidden.

## License

This project is distributed under the MIT License. See `LICENSE` for details.
