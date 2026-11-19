"""
RAG (Retrieval-Augmented Generation) tool over Wikipedia-grounded QA pairs.

Uses (all configurable in config.RAGConfig):
- Dataset: FreedomIntelligence/RAG-Instruct, first `max_rows` rows of the "train" split
  (columns: question, answer, documents). Question + answer pairs are indexed; the
  `documents` passages are not stored, to keep the index small.
- Embedding model: sentence-transformers/all-MiniLM-L6-v2 on Hugging Face
- Metric: Cosine distance (1 - cosine_similarity), with maximum distance <= 0.75
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

try:
    from config import RAGConfig
except ImportError:
    from mylocalmodel.config import RAGConfig

MAX_COSINE_DISTANCE = RAGConfig().max_cosine_distance
CACHE_FORMAT_VERSION = 2
INDEX_FILENAME = "rag_qa_index.pt"


class RAGRowCountError(RuntimeError):
    """The dataset did not yield exactly `max_rows` rows. Fatal: the app must not start."""


class MiniWikiRAG:
    """
    RAG retriever over the configured QA dataset (default: FreedomIntelligence/RAG-Instruct)
    using sentence-transformers/all-MiniLM-L6-v2 embeddings and cosine distance filtering.
    """

    def __init__(
        self,
        config: Optional[RAGConfig] = None,
        device: Optional[torch.device | str] = None,
    ):
        self.config = config or RAGConfig()
        self.cache_dir = self.config.cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device

        self.records: List[Dict[str, Any]] = []
        self.embeddings: Optional[torch.Tensor] = None

        print(f"Loading embedding model: {self.config.embedding_model_name} on {self.device}...")
        self.model = SentenceTransformer(self.config.embedding_model_name, device=str(self.device))
        self._build_or_load_index()

    def _embed_texts(self, texts: List[str], batch_size: int = 64) -> torch.Tensor:
        """
        Embed a list of text strings into normalized L2 PyTorch tensor [N, 384].
        Uses tqdm to display batch progress when processing large datasets.
        """
        if len(texts) <= batch_size:
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=False,
                convert_to_tensor=True,
                normalize_embeddings=True,
                device=str(self.device),
            )
            return embeddings.to(self.device)

        batch_embeddings: List[torch.Tensor] = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Embedding passages"):
            batch = texts[i : i + batch_size]
            emb = self.model.encode(
                batch,
                batch_size=batch_size,
                show_progress_bar=False,
                convert_to_tensor=True,
                normalize_embeddings=True,
                device=str(self.device),
            )
            batch_embeddings.append(emb)

        return torch.cat(batch_embeddings, dim=0).to(self.device)

    def _open_stream(self):
        """
        Open the split as a stream (nothing is written to disk).

        Preferred source: the Hub's automatic Parquet export (revision
        refs/convert/parquet). Parquet is stored in row groups of ~1K rows, so a
        stream of the first max_rows rows only fetches the first row groups, and
        reading just the question/answer columns skips the large `documents`
        column. For RAG-Instruct, 12K rows is ~12 of 41 row groups instead of the
        296 MB JSON file. If the export is unavailable, fall back to streaming the
        original files (a single JSON file is still transferred in full, but is
        not stored on disk).
        """
        try:
            return load_dataset(
                self.config.dataset_name,
                self.config.dataset_config,
                split=self.config.dataset_split,
                revision="refs/convert/parquet",
                streaming=True,
                columns=["question", "answer"],
            ), "Parquet export (question/answer columns only)"
        except Exception as exc:
            print(f"Parquet export not available ({exc}); streaming the original files instead.")
            return load_dataset(
                self.config.dataset_name,
                self.config.dataset_config,
                split=self.config.dataset_split,
                streaming=True,
            ), "original files"

    def _load_raw_data(self) -> List[Dict[str, Any]]:
        """
        Stream the first `max_rows` rows of the configured split, the streaming
        equivalent of load_dataset(name, split="train[:max_rows]"), without
        downloading the whole dataset.

        Raises RuntimeError if the number of loaded rows differs from `max_rows`
        (for example when max_rows exceeds the rows the split actually has).
        """
        max_rows = self.config.max_rows
        if max_rows <= 0:
            raise RAGRowCountError(f"RAG max_rows must be positive, got {max_rows}.")
        stream, source = self._open_stream()
        print(
            f"Streaming the first {max_rows} rows of {self.config.dataset_name} "
            f"(split '{self.config.dataset_split}', {source})..."
        )
        rows = list(tqdm(stream.take(max_rows), total=max_rows, desc="Streaming rows"))

        loaded_rows = len(rows)
        print(f"Loaded {loaded_rows} rows from {self.config.dataset_name} (requested {max_rows}).")
        if loaded_rows != max_rows:
            raise RAGRowCountError(
                f"Expected {max_rows} rows from {self.config.dataset_name} split "
                f"'{self.config.dataset_split}', but loaded {loaded_rows}."
            )

        records = []
        for index, row in enumerate(rows):
            question = str(row.get("question") or "").strip()
            answer = str(row.get("answer") or "").strip()
            if question and answer:
                records.append({"id": row.get("id", index), "question": question, "answer": answer})
        skipped = loaded_rows - len(records)
        if skipped:
            print(f"Skipped {skipped} row(s) with an empty question or answer.")
        print(f"Indexing {len(records)} QA pairs.")
        return records

    def _build_or_load_index(self) -> None:
        """
        Load cached index if available; otherwise compute embeddings and persist to disk.
        """
        cache_file = self.cache_dir / INDEX_FILENAME
        expected = {
            "format_version": CACHE_FORMAT_VERSION,
            "dataset": self.config.dataset_name,
            "config": self.config.dataset_config,
            "split": self.config.dataset_split,
            "max_rows": self.config.max_rows,
            "model": self.config.embedding_model_name,
        }
        if cache_file.exists():
            try:
                # weights_only=True: the cache holds only tensors, lists, dicts and strings,
                # so refuse anything that would need arbitrary unpickling.
                data = torch.load(cache_file, map_location=self.device, weights_only=True)
                stale = {key: data.get(key) for key in expected if data.get(key) != expected[key]}
            except Exception as exc:
                data, stale = None, {"error": str(exc)}
            if data is not None and not stale:
                self.embeddings = data["embeddings"].to(self.device)
                self.records = data["records"]
                print(f"Loaded {len(self.records)} indexed entries from cache: {cache_file}")
                return
            print(f"RAG cache does not match current settings ({stale}); rebuilding index...")

        self.records = self._load_raw_data()
        texts_to_embed = [
            f"Question: {r['question']}\nAnswer: {r['answer']}" for r in self.records
        ]
        print(f"Encoding {len(texts_to_embed)} items with {self.config.embedding_model_name}...")
        self.embeddings = self._embed_texts(texts_to_embed)

        torch.save(
            {
                "embeddings": self.embeddings.cpu(),
                "records": self.records,
                **expected,
            },
            cache_file,
        )
        self.embeddings = self.embeddings.to(self.device)
        print(f"Saved RAG index to {cache_file}")

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        max_distance: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant QA pairs using cosine distance.

        Cosine Distance = 1.0 - Cosine Similarity.
        Filters matches where distance <= max_distance (default 0.75 from config).
        Returns results sorted by distance ascending (closest / highest similarity first).
        """
        if not query or not query.strip() or self.embeddings is None:
            return []

        cutoff_distance = max_distance if max_distance is not None else self.config.max_cosine_distance

        # Embed query [1, 384]
        query_vec = self._embed_texts([query.strip()])

        # Cosine similarity for normalized vectors is the dot product
        cosine_sim = torch.mv(self.embeddings, query_vec[0])  # [N]
        cosine_dist = 1.0 - cosine_sim  # [N]

        # Filter by maximum distance threshold
        valid_mask = cosine_dist <= cutoff_distance
        valid_indices = torch.where(valid_mask)[0]

        if len(valid_indices) == 0:
            return []

        # Sort filtered matches ascending by distance
        matched_distances = cosine_dist[valid_indices]
        sorted_order = torch.argsort(matched_distances)
        top_indices = valid_indices[sorted_order[:top_k]]

        results = []
        for idx in top_indices:
            idx_int = idx.item()
            record = self.records[idx_int]
            dist_val = cosine_dist[idx_int].item()
            sim_val = cosine_sim[idx_int].item()
            results.append({
                "id": record.get("id"),
                "question": record.get("question"),
                "answer": record.get("answer"),
                "distance": round(dist_val, 4),
                "similarity": round(sim_val, 4),
            })
        return results

    def format_context(self, matches: List[Dict[str, Any]]) -> str:
        """
        Format retrieved QA matches into a prompt context block. Each answer is cut to
        `max_context_chars` (at a word boundary) so a few long answers cannot crowd out
        the conversation history.
        """
        if not matches:
            return "No relevant Wikipedia QA context found within distance threshold."

        limit = self.config.max_context_chars
        lines = ["[Wikipedia QA Reference Context]:"]
        for i, match in enumerate(matches, 1):
            answer = match["answer"]
            if limit and len(answer) > limit:
                answer = answer[:limit].rsplit(" ", 1)[0] + " …"
            lines.append(
                f"{i}. Question: {match['question']}\n"
                f"   Answer: {answer}\n"
                f"   (Cosine Distance: {match['distance']:.4f}, Similarity: {match['similarity']:.4f})"
            )
        return "\n".join(lines)


_rag_singleton: Optional[MiniWikiRAG] = None


def get_rag(
    config: Optional[RAGConfig] = None,
    device: Optional[torch.device | str] = None,
) -> MiniWikiRAG:
    """Get or create singleton MiniWikiRAG instance."""
    global _rag_singleton
    if _rag_singleton is None:
        _rag_singleton = MiniWikiRAG(config=config, device=device)
    return _rag_singleton


def rag_retrieve(
    query: str,
    top_k: int = 5,
    max_distance: float = MAX_COSINE_DISTANCE,
    device: Optional[torch.device | str] = None,
) -> List[Dict[str, Any]]:
    """
    Convenience function to retrieve matching Wikipedia QA items.
    Filters by cosine distance <= max_distance (default 0.75).
    """
    rag = get_rag(device=device)
    return rag.retrieve(query, top_k=top_k, max_distance=max_distance)


def query_rag(
    query: str,
    top_k: int = 5,
    max_distance: float = MAX_COSINE_DISTANCE,
    device: Optional[torch.device | str] = None,
) -> str:
    """
    Retrieve matching context and format it as a prompt-ready string.
    """
    rag = get_rag(device=device)
    matches = rag.retrieve(query, top_k=top_k, max_distance=max_distance)
    return rag.format_context(matches)


if __name__ == "__main__":
    test_query = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "What is the capital of France?"
    )
    print(f"=== MiniWiki RAG Test ===")
    print(f"Query: {test_query}")
    rag_tool = get_rag()
    hits = rag_tool.retrieve(test_query, top_k=5, max_distance=MAX_COSINE_DISTANCE)
    print(f"\nRetrieved {len(hits)} match(es) with cosine distance <= {MAX_COSINE_DISTANCE}:")
    print(rag_tool.format_context(hits))
