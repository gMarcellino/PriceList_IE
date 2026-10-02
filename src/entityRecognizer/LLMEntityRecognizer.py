#!/usr/bin/env python3
"""
LLMEntityRecognizer.py

LLM-based named entity recognition (NER) module that jointly extracts and classifies entities.
Supports both online providers (OpenAI, Azure OpenAI, Groq, and any other OpenAI-compatible endpoint) and local providers (LM Studio, Ollama, and HuggingFace Transformers running on the same machine).

Organized into four sections:
    - ResponseParser : converts raw LLM text output into structured entity lists with labels
    - LLMEntityRecognizer : zero-shot NER inference orchestrator
    - Helper functions : prompt loading and provider building (re-exported from LLMProvider)
    - CLI : argument parsing and top-level entry point

All LLM provider classes and the build_provider factory are re-used from LLMProvider and are not redefined here.

The key difference relative to LLMTermExtractor is the expected response format: 
instead of a plain JSON array of strings (term text only), the model is prompted to return a JSON array of {"term": ..., "label": ...} objects so that both extraction and classification are performed in a single LLM call.  
The ResponseParser handles this two-field format and falls back gracefully to the string-only format when the model does not comply.

Entry point: run with --help to see CLI options.
"""

import argparse
import hashlib
import json
import os
import re
import sys

from tqdm import tqdm

# Allow unsupported MPS ops to fall back to CPU instead of crashing
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
# Disable HuggingFace tokenizer parallelism to avoid fork-related warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Add workspace root to Python path for relative imports
workspace_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, workspace_root)

# Import utilities implemented in utils.py
from src.utils.utils import (
    canonicalize_entity_label,
    clean_llm_decoding_artifacts,
    get_device,
    load_json_data,
    maybe_empty_cache,
    print_device_info,
    safe_filename_component,
    save_json_data,
)

# Import reusable provider infrastructure from the canonical shared provider module.
from src.modules.LLMProvider import (
    BaseLLMProvider,
    PROVIDER_REGISTRY,
    build_provider,
    load_prompts,
    DEFAULT_TEMPERATURE,
    DEFAULT_MAX_NEW_TOKENS,
    LLM_CONFSCORE_PLACEHOLDER,
)

from src.entityRecognizer.FewShotExamples import (
    DEFAULT_SEMANTIC_EMBEDDING_MODEL,
    ExampleSelector,
    FewShotPromptFormatter,
    build_example_selector,
    load_ner_few_shot_examples,
)

# Import the NER evaluator
from src.entityRecognizer.HFEntityRecognizer import NERExtractionEvaluator  # noqa: F401  (re-exported for callers)


#############
# Constants #
#############

LLM_LOCATION_PLACEHOLDER = "location_placeholder"


##################
# ResponseParser #
##################

class ResponseParser:
    """
    Converts raw LLM text output into a list of entity dicts compatible with the format produced by HFEntityRecognizer.EntityRecognitionTrainer.perform_inference().

    Expected response format (primary):
        A JSON array of objects, each with "term" and "label" fields:
        [{"term": "Lactobacillus rhamnosus", "label": "bacteria"}, ...]

    Fallback formats (when the model does not produce the expected structure):
        1. JSON array of strings  -- treated as term-only output; label defaults to "NA".
        2. Bullet / dash / numbered list -- one "term: label" or plain term per line.
        3. Newline-separated pairs -- "term | label" or "term: label" on each line.
        4. Last resort -- whole response treated as a single term with label "NA".

    For each extracted (term, label) pair the parser performs a case-insensitive substring search over the source text to recover character offsets, producing one entity dict per occurrence.  
    Labels outside the configured edil ontology are remapped to "NA".
    """

    # Bullet / dash / numbered list item prefix patterns
    _BULLET_RE = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s*", re.MULTILINE)
    _THINKING_BLOCK_RE = re.compile(
        r"<(?P<tag>think|thinking)\b[^>]*>.*?</(?P=tag)>",
        flags=re.IGNORECASE | re.DOTALL,
    )

    @staticmethod
    def parse(response: str, source_text: str, location: str) -> list[dict]:
        """
        Main entry point. Parses 'response' into a list of entity dicts, anchoring each extracted (term, label) pair to its position(s) in 'source_text'.

        :param response: Raw string returned by the LLM.
        :param source_text: The original text that was presented to the model.
        :param location: Location label embedded in every entity dict. Current workflows use a placeholder value.
        :return: A list of entity dicts, each with keys: 'start_idx', 'end_idx', 'location', 'text_span', 'label', 'confscore'.
        """
        response = ResponseParser._prepare_response(response)
        term_label_pairs = ResponseParser._extract_term_label_pairs(response)
        entities = []

        for term, label in term_label_pairs:
            if not term.strip():
                continue
            # Normalise the label against the allowed set; fall back to "NA" on mismatch
            normalised_label = ResponseParser._normalise_label(label)
            spans = ResponseParser._find_spans(term, source_text)
            for start, end in spans:
                entities.append(
                    {
                        "start_idx": start,
                        # HFEntityRecognizer uses inclusive end; adjust accordingly
                        "end_idx": end - 1,
                        "location": location,
                        # Recover exact casing from source text
                        "text_span": source_text[start:end],
                        "label": normalised_label,
                        # LLMs do not expose per-span probabilities; use sentinel
                        "confscore": LLM_CONFSCORE_PLACEHOLDER,
                    }
                )

        return entities

    @staticmethod
    def _extract_term_label_pairs(response: str) -> list[tuple[str, str]]:
        """
        Attempts to extract a flat list of (term, label) pairs from the LLM response, trying each format in order of likelihood.

        Formats tried, in order:
            1. JSON array of {"term": ..., "label": ...} objects  -- primary format
            2. JSON array of strings                              -- term-only fallback (label="NA")
            3. Bullet / dash list with optional ": label" suffix
            4. Newline-separated "term | label" or "term: label" pairs
            5. Comma-separated "term: label" pairs
            6. Last resort: whole response as a single term (label="NA")

        :param response: Raw LLM output string.
        :return: A (possibly empty) list of (term, label) tuples.
        """
        response = ResponseParser._prepare_response(response)
        if not response:
            return []

        # 1. JSON array: parse every valid embedded array and prefer the last
        # compatible one. Reasoning models can emit bracketed lists before the
        # final answer, especially inside thinking traces.
        json_arrays = list(ResponseParser._iter_json_arrays(response))
        for parsed in reversed(json_arrays):
            pairs = ResponseParser._pairs_from_json_array(parsed)
            if pairs is not None:
                return pairs

        # 2. Bullet / dash / numbered list -- optional ": label" suffix
        lines = response.splitlines()
        bullet_pairs = []
        for line in lines:
            if ResponseParser._BULLET_RE.match(line):
                content = ResponseParser._BULLET_RE.sub("", line).strip()
                term, label = ResponseParser._split_term_label(content)
                if term:
                    bullet_pairs.append((term, label))
        if bullet_pairs:
            return bullet_pairs

        # 3. Newline-separated "term | label" or "term: label" pairs
        pair_lines = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            term, label = ResponseParser._split_term_label(line)
            if term:
                pair_lines.append((term, label))
        if len(pair_lines) > 1:
            return pair_lines

        # 4. Comma-separated "term: label" pairs
        if "," in response:
            pairs = []
            for segment in response.split(","):
                term, label = ResponseParser._split_term_label(segment.strip())
                if term:
                    pairs.append((term, label))
            if pairs:
                return pairs

        # 5. Last resort: treat the entire response as a single term, label unknown
        return [(response.strip(), "NA")]

    @staticmethod
    def _prepare_response(response: str) -> str:
        """
        Normalizes raw model output before parsing.
        """
        if not isinstance(response, str):
            return ""

        response = clean_llm_decoding_artifacts(response).strip()
        if not response:
            return ""

        response = ResponseParser._THINKING_BLOCK_RE.sub("", response).strip()
        for marker in ("</think>", "</thinking>"):
            marker_idx = response.lower().rfind(marker)
            if marker_idx != -1:
                response = response[marker_idx + len(marker):].strip()
        return response

    @staticmethod
    def _iter_json_arrays(response: str):
        """
        Yields every syntactically valid JSON array embedded in response.
        """
        decoder = json.JSONDecoder()
        for index, char in enumerate(response):
            if char != "[":
                continue
            try:
                parsed, _ = decoder.raw_decode(response[index:])
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, list):
                yield parsed

    @staticmethod
    def _pairs_from_json_array(parsed: list) -> list[tuple[str, str]] | None:
        """
        Converts a parsed JSON array into term-label pairs when it has a supported shape.
        Returns None for unsupported arrays and [] for a valid empty answer.
        """
        if not parsed:
            return []

        if all(isinstance(item, dict) for item in parsed):
            pairs = []
            for item in parsed:
                term = str(item.get("term", "")).strip()
                label = str(item.get("label", "NA")).strip()
                if term:
                    pairs.append((term, label))
            return pairs

        if all(isinstance(item, str) for item in parsed):
            return [(str(term).strip(), "NA") for term in parsed if term]

        return None

    @staticmethod
    def _split_term_label(text: str) -> tuple[str, str]:
        """
        Attempts to split a string of the form "term | label" or "term: label" into its constituent parts.  
        Returns (text, "NA") when no separator is found.

        :param text: A candidate term-label string.
        :return: A (term, label) tuple.
        """
        for sep in [" | ", ": ", " - "]:
            if sep in text:
                parts = text.split(sep, 1)
                return parts[0].strip(), parts[1].strip()
        return text.strip(), "NA"

    def _normalise_label(label: str) -> str:
        """
        Maps a raw label string from the LLM response to the canonical edil label set.
        Falls back to "NA" when no match is found.

        :param label: Raw label string from the LLM.
        :return: A canonical label string drawn from LABEL_LIST, or "NA".
        """
        return canonicalize_entity_label(label, default="NA")

    @staticmethod
    def _find_spans(term: str, text: str) -> list[tuple[int, int]]:
        """
        Returns all (start, end) character-offset pairs where 'term' appears in 'text',
        using a case-insensitive search.

        :param term: The term string to locate.
        :param text: The source text to search within.
        :return: A list of (start_inclusive, end_exclusive) character index tuples.
        """
        term = clean_llm_decoding_artifacts(term).strip()
        if not term:
            return []

        spans = []
        pattern = re.compile(re.escape(term), re.IGNORECASE)
        for match in pattern.finditer(text):
            spans.append((match.start(), match.end()))
        return spans


########################
# EntityMappingAdapter #
########################

class EntityMappingAdapter:
    """
    Small compatibility layer for mapping entities extracted from one text back to another text.

    This is primarily meant for the English-to-Italian workflow: run entity recognition on
    an English translated construction-sector record, then use a mapper (for example DspyEntityMapper or
    an alignment-based mapper) to recover offsets in the original Italian record.

    The recognizers keep the mapper optional so the direct extraction path remains unchanged.
    """

    @staticmethod
    def map_entities(
        entities: list[dict],
        original_text: str,
        extraction_text: str,
        entity_mapper,
        keep_unmapped: bool = False,
    ) -> list[dict]:
        """
        Maps HF-style entity dicts from extraction_text offsets to original_text offsets.

        :param entities: Entity dicts produced by ResponseParser.parse().
        :param original_text: The text where final offsets should point (e.g. the original Italian text).
        :param extraction_text: The text used for entity recognition (e.g. English translation).
        :param entity_mapper: Mapper object exposing map_entities(), map_entity(), or being directly callable.
        :param keep_unmapped: If True, retain unmapped entities with start_idx/end_idx=-1.
        :return: A list of HF-style entity dicts anchored in original_text.
        """
        if entity_mapper is None:
            return entities

        mapper_entities = [EntityMappingAdapter._to_mapper_schema(entity) for entity in entities]

        if hasattr(entity_mapper, "map_entities"):
            mapped = entity_mapper.map_entities(
                mapper_entities,
                italian_text=original_text,
                english_text=extraction_text,
            )
            mapped_entities = mapped[0] if isinstance(mapped, tuple) else mapped
        elif hasattr(entity_mapper, "map_entity"):
            mapped_entities = [
                entity_mapper.map_entity(entity, italian_text=original_text, english_text=extraction_text)
                for entity in mapper_entities
            ]
        elif callable(entity_mapper):
            mapped_entities = entity_mapper(mapper_entities, original_text, extraction_text)
        else:
            raise TypeError(
                "entity_mapper must expose map_entities(), map_entity(), or be callable"
            )

        converted = []
        for original_entity, mapped_entity in zip(entities, mapped_entities):
            hf_entity = EntityMappingAdapter._from_mapper_schema(original_entity, mapped_entity)
            if keep_unmapped or hf_entity["start_idx"] >= 0:
                converted.append(hf_entity)
        return converted

    @staticmethod
    def _to_mapper_schema(entity: dict) -> dict:
        """
        Converts the HFEntityRecognizer schema into the lightweight schema used by DspyEntityMapper.
        """
        mapper_entity = dict(entity)
        mapper_entity.setdefault("text", entity.get("text_span", ""))
        mapper_entity.setdefault("start", entity.get("start_idx", -1))
        mapper_entity.setdefault("end", entity.get("end_idx", -1) + 1)
        return mapper_entity

    @staticmethod
    def _from_mapper_schema(original_entity: dict, mapped_entity: dict) -> dict:
        """
        Converts a mapper result back into the HFEntityRecognizer entity schema.

        Mapper results are expected to use end-exclusive offsets when exposing "start" and
        "end", matching Python slicing and the current DspyEntityMapper implementation.
        """
        if "start_idx" in mapped_entity and "end_idx" in mapped_entity:
            hf_entity = dict(mapped_entity)
        else:
            start = int(mapped_entity.get("start", -1))
            end_exclusive = int(mapped_entity.get("end", 0))
            hf_entity = dict(original_entity)
            hf_entity.update(
                {
                    "start_idx": start,
                    "end_idx": end_exclusive - 1 if start >= 0 else -1,
                    "text_span": mapped_entity.get("text", "NOT_FOUND"),
                }
            )

        hf_entity.setdefault("label", original_entity.get("label", "NA"))
        hf_entity.setdefault("location", original_entity.get("location", LLM_LOCATION_PLACEHOLDER))
        hf_entity.setdefault("confscore", original_entity.get("confscore", LLM_CONFSCORE_PLACEHOLDER))

        if "original_text" in mapped_entity:
            hf_entity["original_text_span"] = mapped_entity["original_text"]
        if "mapping_confidence" in mapped_entity:
            hf_entity["mapping_confidence"] = mapped_entity["mapping_confidence"]
        if "mapping_method" in mapped_entity:
            hf_entity["mapping_method"] = mapped_entity["mapping_method"]

        return hf_entity


#####################
# LLMEntityRecognizer #
#####################

class LLMEntityRecognizer:
    """
    Zero-shot or few-shot LLM-based NER pipeline that jointly extracts and classifies construction-sector entities.

    Given a dataset in the GBIE format (dict mapping paper IDs to content dicts with a text field), this class:
        1. Formats each text with the user-supplied prompts.
        2. Sends each text through the configured LLM provider.
        3. Parses the raw response into structured entity dicts with semantic labels.
        4. Returns results in the same format as HFEntityRecognizer.perform_inference().

    Responses are optionally cached to a JSONL checkpoint file so that long inference runs can be interrupted and resumed without re-processing already completed examples.

    The NER prompt is expected to elicit a JSON array of {"term": ..., "label": ...} objects.
    The ResponseParser handles deviations from this format gracefully.
    """

    def __init__(
        self,
        provider: BaseLLMProvider,
        system_prompt: str,
        user_prompt_template: str,
        temperature: float = DEFAULT_TEMPERATURE,
        checkpoint_path: str | None = None,
        entity_mapper=None,
        keep_unmapped_entities: bool = False,
        system_as_user: bool = False,
        few_shot_k: int = 0,
        example_selector: ExampleSelector | None = None,
        few_shot_language: str = "IT",
    ):
        """
        Initializes the recognizer with an LLM provider and prompt configuration.

        :param provider: A configured BaseLLMProvider instance.
        :param system_prompt: The system-role message sent to the model for every inference call.
        :param user_prompt_template: A Python format string whose {text} placeholder will be filled with the source text on each call.
        :param temperature: Sampling temperature forwarded to the provider.
        :param checkpoint_path: Optional path to a JSONL file used for resumable inference checkpointing.  Already-processed example IDs are skipped on restart.
        :param entity_mapper: Optional mapper used to project entities extracted from one text back to another text.
        :param keep_unmapped_entities: If True, retain mapper failures with start_idx/end_idx=-1.
        :param system_as_user: If True, prepend the system prompt to the user message and omit the system role.
        :param few_shot_k: Number of labeled demonstrations to include before each query. Zero disables few-shot prompting.
        :param example_selector: Fixed or query-dependent selector used when few_shot_k is greater than zero.
        :param few_shot_language: Language used for the demonstration block labels ('IT' or 'EN').
        """
        if few_shot_k < 0:
            raise ValueError("few_shot_k must be greater than or equal to zero.")
        if few_shot_k > 0 and example_selector is None:
            raise ValueError(
                "example_selector is required when few_shot_k is greater than zero."
            )

        self.provider = provider
        self.system_prompt = system_prompt
        self.user_prompt_template = user_prompt_template
        self.temperature = temperature
        self.checkpoint_path = checkpoint_path
        self.entity_mapper = entity_mapper
        self.keep_unmapped_entities = keep_unmapped_entities
        self.system_as_user = system_as_user
        self.few_shot_k = few_shot_k
        self.example_selector = example_selector
        self.few_shot_formatter = FewShotPromptFormatter(few_shot_language)
        self._checkpoint_fingerprint = self._build_checkpoint_fingerprint()

    # -- Public interface --

    def perform_inference(
        self,
        data: dict,
        text_field: str = "text",
        original_text_field: str | None = None,
    ) -> dict:
        """
        Runs zero-shot NER over an entire dataset.

        :param data: A dict mapping paper IDs to content dicts (GBIE format).
        :param text_field: Key containing the text sent to the LLM.
        :param original_text_field: Optional key containing the text where mapped offsets should point.
        :return: A copy of 'data' with an "entities" list populated for every paper.  Each entity follows the HFEntityRecognizer schema: {start_idx, end_idx, location, text_span, label, confscore}.
        """
        # Load already-processed IDs from an existing checkpoint so that a re-run after
        # interruption skips completed examples.
        processed = self._load_checkpoint()

        result = {}

        with self._open_checkpoint() as ckpt_file:
            for id, content in tqdm(data.items(), total=len(data), desc="LLM NER Inference"):
                # Resume: reconstruct result dict from checkpoint
                if id in processed:
                    result[id] = processed[id]
                    continue

                text = content.get(text_field)
                entity_predictions = []
                if not text:
                    continue
                raw_response = self._call_llm(text, record_id=id)
                entities = ResponseParser.parse(raw_response, text, LLM_LOCATION_PLACEHOLDER)
                if original_text_field is not None and self.entity_mapper is not None:
                    original_text = content.get(original_text_field)
                    if original_text:
                        entities = EntityMappingAdapter.map_entities(
                            entities,
                            original_text=original_text,
                            extraction_text=text,
                            entity_mapper=self.entity_mapper,
                            keep_unmapped=self.keep_unmapped_entities,
                        )
                entity_predictions.extend(entities)

                output_content = dict(content)
                output_content["entities"] = entity_predictions
                result[id] = output_content

                # Persist immediately so progress is not lost on crash
                if ckpt_file is not None:
                    checkpoint_entry = {"id": id, "content": output_content}
                    if self._checkpoint_fingerprint is not None:
                        checkpoint_entry["inference_fingerprint"] = (
                            self._checkpoint_fingerprint
                        )
                    ckpt_file.write(json.dumps(checkpoint_entry) + "\n")
                    ckpt_file.flush()

        return result

    def run_raw_inference(self, data: dict, text_field: str = "text") -> dict:
        """
        Runs inference and returns the raw LLM responses instead of parsed entities.
        Useful for debugging prompt quality or post-processing with a custom parser.

        :param data: A dict mapping paper IDs to content dicts (GBIE format).
        :param text_field: Key containing the text sent to the LLM.
        :return: A dict mapping paper IDs to {"raw_response": str}.
        """
        processed_raw = self._load_raw_checkpoint()
        result = {}

        with self._open_checkpoint(suffix=".raw") as ckpt_file:
            for id, content in tqdm(data.items(), total=len(data), desc="LLM NER Raw Inference"):
                if id in processed_raw:
                    result[id] = processed_raw[id]
                    continue

                text = content.get(text_field)
                if not text:
                    continue
                raw_response = self._call_llm(text, record_id=id)
                entry = {"raw_response": raw_response}
                result[id] = entry

                if ckpt_file is not None:
                    checkpoint_entry = {"id": id, **entry}
                    if self._checkpoint_fingerprint is not None:
                        checkpoint_entry["inference_fingerprint"] = (
                            self._checkpoint_fingerprint
                        )
                    ckpt_file.write(json.dumps(checkpoint_entry) + "\n")
                    ckpt_file.flush()

        return result

    ####################
    # Internal helpers #
    ####################

    def _call_llm(self, text: str, record_id: str | None = None) -> str:
        """
        Formats the user prompt template with the given text and calls the provider.

        :param text: Source text to embed into the user_prompt_template.
        :param record_id: Optional current record ID, used to prevent self-selection.
        :return: Raw string response from the LLM.
        """
        user_prompt = self.user_prompt_template.format(text=text)
        if self.few_shot_k > 0:
            examples = self.example_selector.select(
                query_text=text,
                k=self.few_shot_k,
                query_id=record_id,
            )
            user_prompt = self.few_shot_formatter.augment_user_prompt(
                query_prompt=user_prompt,
                examples=examples,
            )
        system_prompt = self.system_prompt
        if self.system_as_user:
            if self.system_prompt:
                user_prompt = f"{self.system_prompt}\n\n{user_prompt}"
            system_prompt = None
        return self.provider.chat_completion(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=self.temperature,
        )

    def _build_checkpoint_fingerprint(self) -> str | None:
        """
        Builds a fingerprint that prevents zero-shot and incompatible few-shot
        runs from reusing each other's checkpoint entries.
        """
        if self.few_shot_k == 0:
            return None

        payload = (
            f"k={self.few_shot_k}\n"
            f"selector={self.example_selector.fingerprint}\n"
            f"language={self.few_shot_formatter.language}"
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _checkpoint_entry_matches(self, entry: dict) -> bool:
        entry_fingerprint = entry.get("inference_fingerprint")
        if self._checkpoint_fingerprint is None:
            return entry_fingerprint is None
        return entry_fingerprint == self._checkpoint_fingerprint

    def _load_checkpoint(self) -> dict:
        """
        Reads a structured JSONL checkpoint and returns a dict mapping paper IDs to their already-processed output content dicts.

        :return: Dict mapping paper_id -> output_content, or {} if no checkpoint exists.
        """
        if not self.checkpoint_path or not os.path.exists(self.checkpoint_path):
            return {}
        processed = {}
        with open(self.checkpoint_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    if not self._checkpoint_entry_matches(entry):
                        continue
                    processed[entry["id"]] = entry["content"]
                except (json.JSONDecodeError, KeyError):
                    pass  # Skip malformed lines silently
        return processed

    def _load_raw_checkpoint(self) -> dict:
        """
        Reads a raw JSONL checkpoint and returns a dict mapping paper IDs to their response strings.

        :return: Dict mapping paper_id -> {"raw_response": str}.
        """
        raw_path = (self.checkpoint_path + ".raw") if self.checkpoint_path else None
        if not raw_path or not os.path.exists(raw_path):
            return {}
        processed = {}
        with open(raw_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    if not self._checkpoint_entry_matches(entry):
                        continue
                    paper_id = entry.pop("id")
                    entry.pop("inference_fingerprint", None)
                    processed[paper_id] = entry
                except (json.JSONDecodeError, KeyError):
                    pass
        return processed

    def _open_checkpoint(self, suffix: str = ""):
        """
        Opens the checkpoint file for appending, or returns a null context if no checkpoint path was configured.

        :param suffix: Optional suffix appended to self.checkpoint_path (e.g. ".raw").
        :return: A file object opened in append mode, or a _NullContext instance.
        """
        if self.checkpoint_path is None:
            return _NullContext()
        path = self.checkpoint_path + suffix
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        return open(path, "a", encoding="utf-8")


class _NullContext:
    """Minimal no-op context manager used when checkpointing is disabled."""

    def __enter__(self):
        return None

    def __exit__(self, *_):
        pass

def generate_output_filename(args: argparse.Namespace) -> str:
    """
    Generate a standardized output filename based on configuration parameters.

    :param args: Parsed CLI arguments.
    :return: Output filename string.
    """
    stem = (
        f"entities_{args.language}_{args.provider}_"
        f"{safe_filename_component(args.model)}"
    )
    few_shot_k = getattr(args, "few_shot_k", 0)
    if few_shot_k > 0:
        strategy = safe_filename_component(
            getattr(args, "few_shot_strategy", "fixed")
        )
        stem += f"_fewshot-{strategy}-k{few_shot_k}"
    return stem + ".json"
    
#######
# CLI #
#######

def parse_args() -> argparse.Namespace:
    """
    Defines and parses command-line arguments for zero-shot LLM NER inference.

    :return: An argparse.Namespace object with all parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Zero-shot or few-shot LLM-based NER (joint entity extraction + classification) for construction-sector technical records.  "
            "Supports OpenAI, Azure, Groq, LM Studio, Ollama, and HuggingFace backends."
        )
    )

    # -- Config file (optional) --
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to a YAML config file.  When provided, CLI arguments are ignored in favor of config values.",
    )

    # -- Provider --
    parser.add_argument(
        "--provider",
        type=str,
        required=False,
        choices=sorted(PROVIDER_REGISTRY.keys()),
        help="LLM backend to use (e.g. 'openai', 'lmstudio', 'huggingface')",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=False,
        help=(
            "Model identifier as expected by the provider "
            "(e.g. 'gpt-4o', 'medgemma-27b-text-it', 'llama3')"
        ),
    )

    # -- Provider credentials / endpoints --
    parser.add_argument(
        "--api_key",
        type=str,
        default=None,
        help="API key for the provider.  Defaults to the relevant environment variable.",
    )
    parser.add_argument(
        "--base_url",
        type=str,
        default=None,
        help=(
            "Override the default API base URL (OpenAI-compatible providers only).  "
            "Well-known local server defaults are applied automatically for 'lmstudio' "
            "and 'ollama'."
        ),
    )
    parser.add_argument(
        "--azure_endpoint",
        type=str,
        default=None,
        help="Azure resource endpoint URL (azure provider only).",
    )
    parser.add_argument(
        "--azure_api_version",
        type=str,
        default="2024-02-01",
        help="Azure API version string (azure provider only, default: 2024-02-01).",
    )

    # -- Data paths --
    parser.add_argument(
        "--inference_data_path",
        type=str,
        required=False,
        help="Path to the JSON file to run inference on (GBIE format).",
    )
    parser.add_argument(
        "--text_field",
        type=str,
        default="text",
        help="Key inside each content dict containing the text sent to the LLM (default: 'text').",
    )
    parser.add_argument(
        "--inference_output_path",
        type=str,
        required=False,
        help="Path where inference results JSON will be written.",
    )
    parser.add_argument(
        "--prompts_path",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "prompts_IT.json"),
        help="Path to the prompts JSON file (default: prompts_IT.json in the same directory).",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help=(
            "Optional path to a JSONL checkpoint file.  Enables resumable inference: "
            "already-processed IDs are skipped on restart."
        ),
    )

    # -- Prompt selection --
    parser.add_argument(
        "--system_prompt_key",
        type=str,
        default="base",
        help="Key selecting the system prompt variant from the prompts file.",
    )
    parser.add_argument(
        "--user_prompt_key",
        type=str,
        default="base",
        help="Key selecting the user prompt variant from the prompts file.",
    )
    parser.add_argument(
        "--language",
        type=str,
        default="IT",
        choices=["IT", "EN"],
        help="Language label used in the generated output filename (default: IT).",
    )

    # -- Few-shot prompting --
    parser.add_argument(
        "--few_shot_k",
        type=int,
        default=0,
        help="Number of labeled examples to include before each query (default: 0).",
    )
    parser.add_argument(
        "--few_shot_strategy",
        type=str,
        default="fixed",
        choices=["fixed", "semantic"],
        help="Example selection strategy used when few_shot_k > 0 (default: fixed).",
    )
    parser.add_argument(
        "--few_shot_examples_path",
        type=str,
        default=None,
        help="Path to a labeled GBIE-format JSON file used as the example pool.",
    )
    parser.add_argument(
        "--few_shot_text_field",
        type=str,
        default=None,
        help=(
            "Text field in the few-shot examples file. "
            "Defaults to the inference text_field."
        ),
    )
    parser.add_argument(
        "--few_shot_fixed_example_ids",
        nargs="*",
        default=None,
        help=(
            "Optional ordered record IDs for the fixed strategy. "
            "When omitted, a seeded deterministic ordering is used."
        ),
    )
    parser.add_argument(
        "--few_shot_seed",
        type=int,
        default=42,
        help="Seed used to construct the deterministic fixed example order.",
    )
    parser.add_argument(
        "--few_shot_embedding_model",
        type=str,
        default=DEFAULT_SEMANTIC_EMBEDDING_MODEL,
        help="Sentence-transformers model used by semantic selection.",
    )
    parser.add_argument(
        "--few_shot_embeddings_cache_path",
        type=str,
        default=None,
        help="Optional .npz path used to cache example-pool embeddings.",
    )
    parser.add_argument(
        "--few_shot_embedding_device",
        type=str,
        default=None,
        help="Optional sentence-transformers device override (e.g. cpu, cuda, mps).",
    )
    parser.add_argument(
        "--few_shot_exclude_same_text",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Exclude examples whose normalized text equals the current query.",
    )

    # -- Generation hyper-parameters --
    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
        help=f"Sampling temperature (default: {DEFAULT_TEMPERATURE}; 0.0 = greedy).",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=DEFAULT_MAX_NEW_TOKENS,
        help=f"Maximum tokens the model may generate per call (default: {DEFAULT_MAX_NEW_TOKENS}).",
    )

    # -- Optional evaluation --
    parser.add_argument(
        "--eval_data_path",
        type=str,
        default=None,
        help=(
            "Optional path to a ground-truth JSON file.  When provided, strict (span+label) "
            "and lenient (span-only) P/R/F1 are printed after inference."
        ),
    )

    # -- HuggingFace-only options --
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help=(
            "Target device for HuggingFace local inference ('cuda', 'mps', 'cpu').  "
            "Auto-detected when not specified."
        ),
    )
    parser.add_argument(
        "--torch_dtype",
        type=str,
        default="auto",
        choices=["auto", "float16", "bfloat16", "float32"],
        help=(
            "Weight dtype for HuggingFace local inference. With 'auto', "
            "Gemma 3 uses BF16 on supported MPS/CUDA devices."
        ),
    )

    # -- Raw output mode --
    parser.add_argument(
        "--raw",
        action="store_true",
        help=(
            "Save raw LLM responses instead of parsed entities.  "
            "Useful for debugging prompt quality."
        ),
    )
    parser.add_argument(
        "--system_as_user",
        action="store_true",
        help="Prepend the system prompt to the user message instead of sending a system-role message.",
    )

    args = parser.parse_args()
    
    if args.config is not None:
        # Load config from YAML file and override CLI args
        import yaml

        with open(args.config, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
        if not isinstance(config, dict):
            parser.error("the YAML config root must be a mapping")

        config = dict(config)
        few_shot_config = config.pop("few_shot", None)
        for key, value in config.items():
            setattr(args, key, value)
        if few_shot_config is not None:
            if not isinstance(few_shot_config, dict):
                parser.error("the 'few_shot' YAML section must be a mapping")
            for key, value in few_shot_config.items():
                attribute = f"few_shot_{key}"
                if not hasattr(args, attribute):
                    parser.error(
                        f"unknown few-shot config option: few_shot.{key}"
                    )
                setattr(args, attribute, value)

    # Validate that all required arguments are present (from CLI or config)
    required_args = ["provider", "model", "inference_data_path", "inference_output_path"]
    missing = [arg for arg in required_args if not getattr(args, arg, None)]
    if missing:
        parser.error(f"the following arguments are required: {', '.join('--' + arg for arg in missing)}")
    if args.few_shot_k < 0:
        parser.error("--few_shot_k must be greater than or equal to zero")
    if args.few_shot_k > 0 and not args.few_shot_examples_path:
        parser.error(
            "--few_shot_examples_path is required when --few_shot_k is greater than zero"
        )
    if args.few_shot_strategy not in {"fixed", "semantic"}:
        parser.error("--few_shot_strategy must be 'fixed' or 'semantic'")

    return args

def print_args(args: argparse.Namespace) -> None:
    """
    Prints the parsed CLI arguments in a human-readable format.

    :param args: Parsed CLI arguments.
    :return: None
    """
    print("=== LLM Entity Recognizer Configuration ===")
    for key, value in vars(args).items():
        print(f"{key}: {value}")
    print("==========================================")


###############
# Entry point #
###############

def run_inference(args: argparse.Namespace) -> None:
    """
    Orchestrates provider initialization, inference, optional evaluation, and result persistence.

    :param args: Parsed CLI arguments.
    :return: None
    """
    device = get_device()
    print_device_info(device)

    # -- Build provider --
    provider = build_provider(
        provider_name=args.provider,
        model=args.model,
        api_key=args.api_key,
        base_url=args.base_url,
        azure_endpoint=args.azure_endpoint,
        api_version=args.azure_api_version,
        max_new_tokens=args.max_new_tokens,
        device=args.device,
        torch_dtype=getattr(args, "torch_dtype", "auto"),
    )

    # -- Load prompts --
    system_prompt, user_prompt_template = load_prompts(
        prompts_path=args.prompts_path,
        system_key=args.system_prompt_key,
        user_key=args.user_prompt_key,
    )

    # -- Build optional few-shot example selector --
    example_selector = None
    if args.few_shot_k > 0:
        example_text_field = args.few_shot_text_field or args.text_field
        examples = load_ner_few_shot_examples(
            path=args.few_shot_examples_path,
            text_field=example_text_field,
        )
        example_selector = build_example_selector(
            strategy=args.few_shot_strategy,
            examples=examples,
            fixed_example_ids=args.few_shot_fixed_example_ids,
            seed=args.few_shot_seed,
            embedding_model=args.few_shot_embedding_model,
            embeddings_cache_path=args.few_shot_embeddings_cache_path,
            embedding_device=args.few_shot_embedding_device,
            exclude_same_text=args.few_shot_exclude_same_text,
        )
        print(
            f"Loaded {len(examples)} few-shot candidates; "
            f"strategy={args.few_shot_strategy}, k={args.few_shot_k}"
        )

    # -- Build recognizer --
    recognizer = LLMEntityRecognizer(
        provider=provider,
        system_prompt=system_prompt,
        user_prompt_template=user_prompt_template,
        temperature=args.temperature,
        checkpoint_path=args.checkpoint_path,
        system_as_user=args.system_as_user,
        few_shot_k=args.few_shot_k,
        example_selector=example_selector,
        few_shot_language=args.language,
    )

    # -- Load inference data --
    data = load_json_data(args.inference_data_path)

    # -- Run inference --
    if args.raw:
        results = recognizer.run_raw_inference(data, text_field=args.text_field)
    else:
        results = recognizer.perform_inference(data, text_field=args.text_field)

    output_filename = generate_output_filename(args)
    raw_dir = os.path.join(os.path.dirname(args.inference_output_path), "raw")
    os.makedirs(raw_dir, exist_ok=True)
    raw_output_filename = os.path.join(raw_dir, output_filename)
    save_json_data(results, raw_output_filename)
    print(f"Inference results saved to {raw_output_filename}")

    # -- Optional evaluation --
    if args.eval_data_path and not args.raw:
        ground_truth = load_json_data(args.eval_data_path)
        evaluator = NERExtractionEvaluator()
        metrics = evaluator.evaluate(results, ground_truth)
        print(
            f"Strict   (span+label)  P: {metrics['strict_precision']:.4f} | "
            f"R: {metrics['strict_recall']:.4f} | "
            f"F1: {metrics['strict_f1']:.4f}"
        )
        print(
            f"Lenient  (span-only)   P: {metrics['span_precision']:.4f} | "
            f"R: {metrics['span_recall']:.4f} | "
            f"F1: {metrics['span_f1']:.4f}"
        )

    maybe_empty_cache(device)


if __name__ == "__main__":
    args = parse_args()
    print_args(args)
    run_inference(args)
