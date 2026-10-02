#!/usr/bin/env python3
"""
Reusable few-shot example loading, selection, and prompt formatting for NER.

The module is intentionally independent from any LLM or DSPy runtime. It reads
the existing GBIE entity-recognition schema and exposes selectors that can be
shared by different recognizer backends.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
from dataclasses import dataclass
from typing import Protocol, Sequence

import numpy as np


DEFAULT_SEMANTIC_EMBEDDING_MODEL = (
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
)


@dataclass(frozen=True)
class NERFewShotExample:
    """One labeled NER demonstration."""

    record_id: str
    text: str
    entities_json: str


class EmbeddingEncoder(Protocol):
    """Minimal interface required by the semantic example selector."""

    @property
    def cache_key(self) -> str:
        """Stable identifier for the encoder configuration."""

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """Encode texts into a two-dimensional float array."""


class ExampleSelector(Protocol):
    """Interface shared by fixed and query-dependent example selectors."""

    @property
    def fingerprint(self) -> str:
        """Stable identifier used to separate incompatible checkpoints."""

    def select(
        self,
        query_text: str,
        k: int,
        query_id: str | None = None,
    ) -> list[NERFewShotExample]:
        """Select exactly k demonstrations for one query."""


def serialize_gold_entities(entities: Sequence[dict]) -> str:
    """
    Convert gold entity annotations to the JSON response expected by the LLM.

    Duplicate (surface form, label) pairs are removed while preserving their
    first occurrence.
    """
    seen: set[tuple[str, str]] = set()
    pairs: list[dict[str, str]] = []

    for entity in entities:
        term = str(entity.get("text_span", "")).strip()
        label = str(entity.get("label", "NA")).strip() or "NA"
        key = (term, label)
        if term and key not in seen:
            seen.add(key)
            pairs.append({"term": term, "label": label})

    return json.dumps(pairs, ensure_ascii=False)


def load_ner_few_shot_examples(
    path: str,
    text_field: str = "text",
    entities_field: str = "entities",
) -> list[NERFewShotExample]:
    """Load demonstrations from a GBIE-format JSON dictionary."""
    with open(path, "r", encoding="utf-8") as input_file:
        data = json.load(input_file)

    if not isinstance(data, dict):
        raise ValueError(
            f"Few-shot examples file must contain a JSON object; got "
            f"{type(data).__name__}."
        )

    examples: list[NERFewShotExample] = []
    for record_id, content in data.items():
        if not isinstance(content, dict):
            continue

        text = content.get(text_field)
        if not isinstance(text, str) or not text.strip():
            continue

        entities = content.get(entities_field, [])
        if not isinstance(entities, list):
            raise ValueError(
                f"Record '{record_id}' has a non-list '{entities_field}' field."
            )

        examples.append(
            NERFewShotExample(
                record_id=str(record_id),
                text=text,
                entities_json=serialize_gold_entities(entities),
            )
        )

    if not examples:
        raise ValueError(
            f"No usable few-shot examples were found in '{path}' using "
            f"text field '{text_field}'."
        )

    return examples


def _normalise_comparison_text(text: str) -> str:
    return " ".join(text.casefold().split())


def _examples_fingerprint(examples: Sequence[NERFewShotExample]) -> str:
    digest = hashlib.sha256()
    for example in examples:
        digest.update(example.record_id.encode("utf-8"))
        digest.update(b"\0")
        digest.update(example.text.encode("utf-8"))
        digest.update(b"\0")
        digest.update(example.entities_json.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _fingerprint_payload(payload: dict) -> str:
    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


class FixedExampleSelector:
    """
    Select the same deterministic ordered demonstrations for every query.

    When explicit IDs are omitted, the complete pool is shuffled once with the
    configured seed. Selecting different k values therefore uses deterministic
    prefixes of the same ordering.
    """

    def __init__(
        self,
        examples: Sequence[NERFewShotExample],
        fixed_example_ids: Sequence[str] | None = None,
        seed: int = 42,
        exclude_same_text: bool = True,
    ):
        if not examples:
            raise ValueError("FixedExampleSelector requires at least one example.")

        examples_by_id = {example.record_id: example for example in examples}
        if len(examples_by_id) != len(examples):
            raise ValueError("Few-shot example record IDs must be unique.")

        explicit_ids = [
            str(record_id) for record_id in (fixed_example_ids or [])
        ]
        if len(explicit_ids) != len(set(explicit_ids)):
            raise ValueError("fixed_example_ids must not contain duplicates.")

        if explicit_ids:
            missing_ids = [
                record_id
                for record_id in explicit_ids
                if record_id not in examples_by_id
            ]
            if missing_ids:
                raise ValueError(
                    "Unknown fixed few-shot example IDs: "
                    + ", ".join(missing_ids)
                )
            ordered_examples = [
                examples_by_id[record_id] for record_id in explicit_ids
            ]
        else:
            ordered_examples = list(examples)
            random.Random(seed).shuffle(ordered_examples)

        self.examples = ordered_examples
        self.exclude_same_text = exclude_same_text
        self._fingerprint = _fingerprint_payload(
            {
                "strategy": "fixed",
                "ordered_example_ids": [
                    example.record_id for example in ordered_examples
                ],
                "examples": _examples_fingerprint(ordered_examples),
                "exclude_same_text": exclude_same_text,
            }
        )

    @property
    def fingerprint(self) -> str:
        return self._fingerprint

    def select(
        self,
        query_text: str,
        k: int,
        query_id: str | None = None,
    ) -> list[NERFewShotExample]:
        if k < 0:
            raise ValueError("k must be greater than or equal to zero.")
        if k == 0:
            return []

        normalised_query = _normalise_comparison_text(query_text)
        candidates = [
            example
            for example in self.examples
            if example.record_id != query_id
            and (
                not self.exclude_same_text
                or _normalise_comparison_text(example.text) != normalised_query
            )
        ]

        if len(candidates) < k:
            raise ValueError(
                f"Requested {k} fixed few-shot examples, but only "
                f"{len(candidates)} eligible examples are available."
            )

        return candidates[:k]


class SentenceTransformerEmbeddingEncoder:
    """
    Lazy sentence-transformers encoder used by semantic selection.

    The optional dependency is imported only when embeddings are first needed,
    so fixed and zero-shot inference do not require sentence-transformers.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_SEMANTIC_EMBEDDING_MODEL,
        device: str | None = None,
        batch_size: int = 32,
    ):
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self._model = None

    @property
    def cache_key(self) -> str:
        return f"sentence-transformers:{self.model_name}:device={self.device or 'auto'}"

    def _load_model(self):
        if self._model is not None:
            return self._model

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "Semantic few-shot selection requires the optional "
                "'sentence-transformers' package. Install it with: "
                "pip install sentence-transformers"
            ) from exc

        self._model = SentenceTransformer(
            self.model_name,
            device=self.device,
        )
        return self._model

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        model = self._load_model()
        embeddings = model.encode(
            list(texts),
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=False,
            show_progress_bar=len(texts) > self.batch_size,
        )
        return np.asarray(embeddings, dtype=np.float32)


class SemanticSimilarityExampleSelector:
    """Select demonstrations by cosine similarity to the current input text."""

    def __init__(
        self,
        examples: Sequence[NERFewShotExample],
        encoder: EmbeddingEncoder,
        embeddings_cache_path: str | None = None,
        exclude_same_text: bool = True,
    ):
        if not examples:
            raise ValueError(
                "SemanticSimilarityExampleSelector requires at least one example."
            )

        self.examples = list(examples)
        self.encoder = encoder
        self.exclude_same_text = exclude_same_text
        self.embeddings_cache_path = self._normalise_cache_path(
            embeddings_cache_path
        )
        self._pool_fingerprint = _examples_fingerprint(self.examples)
        self._cache_fingerprint = _fingerprint_payload(
            {
                "examples": self._pool_fingerprint,
                "encoder": encoder.cache_key,
            }
        )
        self._fingerprint = _fingerprint_payload(
            {
                "strategy": "semantic",
                "cache": self._cache_fingerprint,
                "exclude_same_text": exclude_same_text,
            }
        )
        self._normalised_embeddings: np.ndarray | None = None

    @staticmethod
    def _normalise_cache_path(cache_path: str | None) -> str | None:
        if not cache_path:
            return None
        return cache_path if cache_path.endswith(".npz") else cache_path + ".npz"

    @property
    def fingerprint(self) -> str:
        return self._fingerprint

    @staticmethod
    def _normalise_embeddings(embeddings: np.ndarray) -> np.ndarray:
        embeddings = np.asarray(embeddings, dtype=np.float32)
        if embeddings.ndim != 2:
            raise ValueError(
                "Embedding encoder must return a two-dimensional array."
            )
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0.0, 1.0, norms)
        return embeddings / norms

    def _load_cached_embeddings(self) -> np.ndarray | None:
        cache_path = self.embeddings_cache_path
        if not cache_path or not os.path.exists(cache_path):
            return None

        try:
            with np.load(cache_path, allow_pickle=False) as cache:
                fingerprint = str(cache["fingerprint"].item())
                embeddings = cache["embeddings"]
        except (OSError, KeyError, ValueError):
            return None

        if fingerprint != self._cache_fingerprint:
            return None
        if embeddings.shape[0] != len(self.examples):
            return None
        return self._normalise_embeddings(embeddings)

    def _save_cached_embeddings(self, embeddings: np.ndarray) -> None:
        cache_path = self.embeddings_cache_path
        if not cache_path:
            return

        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        np.savez_compressed(
            cache_path,
            fingerprint=np.asarray(self._cache_fingerprint),
            embeddings=np.asarray(embeddings, dtype=np.float32),
        )

    def _get_pool_embeddings(self) -> np.ndarray:
        if self._normalised_embeddings is not None:
            return self._normalised_embeddings

        cached_embeddings = self._load_cached_embeddings()
        if cached_embeddings is not None:
            self._normalised_embeddings = cached_embeddings
            return cached_embeddings

        embeddings = self.encoder.encode(
            [example.text for example in self.examples]
        )
        if embeddings.shape[0] != len(self.examples):
            raise ValueError(
                "Embedding encoder returned a different number of vectors "
                "than input examples."
            )

        self._normalised_embeddings = self._normalise_embeddings(embeddings)
        self._save_cached_embeddings(self._normalised_embeddings)
        return self._normalised_embeddings

    def select(
        self,
        query_text: str,
        k: int,
        query_id: str | None = None,
    ) -> list[NERFewShotExample]:
        if k < 0:
            raise ValueError("k must be greater than or equal to zero.")
        if k == 0:
            return []

        pool_embeddings = self._get_pool_embeddings()
        query_embedding = self._normalise_embeddings(
            self.encoder.encode([query_text])
        )
        if query_embedding.shape[0] != 1:
            raise ValueError(
                "Embedding encoder must return exactly one vector for one query."
            )
        if query_embedding.shape[1] != pool_embeddings.shape[1]:
            raise ValueError(
                "Query and example embeddings have incompatible dimensions."
            )

        scores = pool_embeddings @ query_embedding[0]
        normalised_query = _normalise_comparison_text(query_text)
        ranked_candidates = []

        for index, example in enumerate(self.examples):
            if example.record_id == query_id:
                continue
            if (
                self.exclude_same_text
                and _normalise_comparison_text(example.text) == normalised_query
            ):
                continue
            ranked_candidates.append(
                (float(scores[index]), example.record_id, example)
            )

        if len(ranked_candidates) < k:
            raise ValueError(
                f"Requested {k} semantic few-shot examples, but only "
                f"{len(ranked_candidates)} eligible examples are available."
            )

        ranked_candidates.sort(key=lambda item: (-item[0], item[1]))
        return [item[2] for item in ranked_candidates[:k]]


class FewShotPromptFormatter:
    """Render demonstrations before the final query in the user prompt."""

    def __init__(self, language: str = "IT"):
        language = language.upper()
        if language not in {"IT", "EN"}:
            raise ValueError("Few-shot prompt language must be 'IT' or 'EN'.")
        self.language = language

    def format_examples(
        self,
        examples: Sequence[NERFewShotExample],
    ) -> str:
        if not examples:
            return ""

        if self.language == "IT":
            header = (
                "Esempi annotati. Segui lo stesso formato per il testo finale."
            )
            example_label = "Esempio"
            text_label = "Testo"
        else:
            header = (
                "Annotated examples. Follow the same format for the final text."
            )
            example_label = "Example"
            text_label = "Text"

        blocks = [header]
        for index, example in enumerate(examples, start=1):
            blocks.append(
                f"{example_label} {index}\n"
                f"{text_label}:\n{example.text}\n"
                f"Array JSON:\n{example.entities_json}"
            )

        return "\n\n".join(blocks)

    def augment_user_prompt(
        self,
        query_prompt: str,
        examples: Sequence[NERFewShotExample],
    ) -> str:
        example_block = self.format_examples(examples)
        if not example_block:
            return query_prompt
        return f"{example_block}\n\n{query_prompt}"


def build_example_selector(
    strategy: str,
    examples: Sequence[NERFewShotExample],
    fixed_example_ids: Sequence[str] | None = None,
    seed: int = 42,
    embedding_model: str = DEFAULT_SEMANTIC_EMBEDDING_MODEL,
    embeddings_cache_path: str | None = None,
    embedding_device: str | None = None,
    exclude_same_text: bool = True,
) -> ExampleSelector:
    """Build a fixed or semantic selector from configuration values."""
    strategy = strategy.lower()
    if strategy == "fixed":
        return FixedExampleSelector(
            examples=examples,
            fixed_example_ids=fixed_example_ids,
            seed=seed,
            exclude_same_text=exclude_same_text,
        )
    if strategy == "semantic":
        encoder = SentenceTransformerEmbeddingEncoder(
            model_name=embedding_model,
            device=embedding_device,
        )
        return SemanticSimilarityExampleSelector(
            examples=examples,
            encoder=encoder,
            embeddings_cache_path=embeddings_cache_path,
            exclude_same_text=exclude_same_text,
        )
    raise ValueError(
        f"Unsupported few-shot strategy '{strategy}'. Choose 'fixed' or 'semantic'."
    )


__all__ = [
    "DEFAULT_SEMANTIC_EMBEDDING_MODEL",
    "EmbeddingEncoder",
    "ExampleSelector",
    "FewShotPromptFormatter",
    "FixedExampleSelector",
    "NERFewShotExample",
    "SemanticSimilarityExampleSelector",
    "SentenceTransformerEmbeddingEncoder",
    "build_example_selector",
    "load_ner_few_shot_examples",
    "serialize_gold_entities",
]
