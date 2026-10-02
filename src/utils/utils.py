#%%
from __future__ import annotations

import json
import os
import random
import numpy as np
import re
import requests
import uuid

try:
    import pandas as pd
except ModuleNotFoundError:
    pd = None

try:
    import torch
except ModuleNotFoundError:
    torch = None

###################################
# Constants Shared Across Modules #
###################################

# Full set of section labels used throughout the sectionExtractor module
SECTION_LABEL_LIST = [
    'A: Allergie', 
    'AI: Azioni Eseguite', 
    'APP: Anamnesi Patologica Prossima', 
    'APR: Anamnesi Patologica Remota', 
    'DS: Dati Strumentali', 
    'EO: Esame Obiettivo', 
    'T: Terapia'
]

LLM_BYTE_ARTIFACT_PATTERN = re.compile(r"\[UNK_BYTE_0x([0-9a-fA-F]+)[^\]]*\]")

# Full set of entity category labels used throughout entity extraction,
# recognition, classification, and linking modules.
ENTITY_LABEL_LIST = [
    "Geometria nome",
    "Geometria_valore",
    "Proprietà_valore",
    "Proprietà nome",
    "Prodotto edilizio",
    "Elemento",
    "Materiale",
    "Metodo",
    "Voce di costo",
]

# Compatibility alias used by term classification modules.
LABEL_LIST = ENTITY_LABEL_LIST

# Common aliases accepted when reading legacy annotations or model outputs.
# Canonical output always uses the strings in ENTITY_LABEL_LIST.
ENTITY_LABEL_ALIASES = {
    "geometria_nome": "Geometria nome",
    "geometria valore": "Geometria_valore",
    "proprietà valore": "Proprietà_valore",
    "proprieta_valore": "Proprietà_valore",
    "proprieta valore": "Proprietà_valore",
    "proprieta nome": "Proprietà nome",
    "proprietà_nome": "Proprietà nome",
    "proprieta_nome": "Proprietà nome",
    "prodotto_edilizio": "Prodotto edilizio",
    "prodottoedilizio": "Prodotto edilizio",
    "voce_di_costo": "Voce di costo",
    "vocedicosto": "Voce di costo",
}


def canonicalize_entity_label(label: str, default: str = "NA") -> str:
    """
    Return the canonical edil entity label for a raw annotation/model label.

    Matching is case-insensitive and accepts a few common underscore/space and
    accent variants. Unknown labels return ``default``.
    """
    if not isinstance(label, str):
        return default

    normalized = label.strip().lower()
    valid_labels = {entity_label.lower(): entity_label for entity_label in ENTITY_LABEL_LIST}
    if normalized in valid_labels:
        return valid_labels[normalized]
    return ENTITY_LABEL_ALIASES.get(normalized, default)

# Full NER BIO label set for the joint extraction + classification task.
# "O" is always first; B- and I- variants are generated for every non-NA entity class drawn from LABEL_LIST, preserving the original ordering.
NER_BIO_LABELS = ["O"] + [
    f"{prefix}-{label}"
    for prefix in ["B", "I"]
    for label in ENTITY_LABEL_LIST
    if label != "NA"
]

# Entity type list for GLiNER -- all ENTITY_LABEL_LIST entries except "NA", lower-cased to match the convention enforced by GLiNER during both fine-tuning and inference.
# The ordering mirrors ENTITY_LABEL_LIST so that the two constants stay in sync.
GLINER_ENTITY_TYPES = [label.lower() for label in ENTITY_LABEL_LIST if label != "NA"]


# Full set of relation predicate labels used throughout the relationExtractor module.
# "NA" is always the first entry and acts as the negative / no-relation class.
RELATION_LABEL_LIST = [
    "NA",
    "haMateriale",
    "haProdotto",
    "haPropNome",
    "haGeomNome",
    "haMetodo",
    "haGeomValore",
    "haPropValore",
    "haPrezzario",
]

# Edil ontology relation schema.  The canonical labels below match the labels used
# by the GBIE JSON files; canonicalize_entity_label() accepts ontology-style aliases
# such as "ProdottoEdilizio", "Geometria_nome", and "VoceDiCosto" at module
# boundaries.
RELATION_SCHEMA: dict[str, dict[str, set[str]]] = {
    "haMateriale": {
        "domain": {"Elemento", "Prodotto edilizio"},
        "range": {"Materiale"},
    },
    "haProdotto": {
        "domain": {"Elemento", "Prodotto edilizio"},
        "range": {"Prodotto edilizio"},
    },
    "haPropNome": {
        "domain": {"Materiale", "Prodotto edilizio", "Elemento"},
        "range": {"Proprietà nome"},
    },
    "haGeomNome": {
        "domain": {"Materiale", "Prodotto edilizio", "Elemento"},
        "range": {"Geometria nome"},
    },
    "haMetodo": {
        "domain": {"Materiale", "Prodotto edilizio", "Elemento"},
        "range": {"Metodo"},
    },
    "haGeomValore": {
        "domain": {"Geometria nome", "Materiale", "Prodotto edilizio", "Elemento"},
        "range": {"Geometria_valore"},
    },
    "haPropValore": {
        "domain": {"Proprietà nome", "Materiale", "Prodotto edilizio", "Elemento"},
        "range": {"Proprietà_valore"},
    },
    "haPrezzario": {
        "domain": {"Elemento", "Prodotto edilizio", "Materiale"},
        "range": {"Voce di costo"},
    },
}

# Schema-constrained valid predicates for each ordered
# (subject_entity_label, object_entity_label) pair.
VALID_RELATIONS: dict[tuple[str, str], set[str]] = {}
for _predicate, _constraints in RELATION_SCHEMA.items():
    for _domain_label in _constraints["domain"]:
        for _range_label in _constraints["range"]:
            VALID_RELATIONS.setdefault(
                (_domain_label, _range_label), set()
            ).add(_predicate)


def get_valid_relation_predicates(
    subject_label: str,
    object_label: str,
) -> set[str]:
    """
    Return the predicates allowed for an ordered entity-type pair.

    Entity labels are canonicalized so callers may use either the display labels
    stored in the GBIE files or ontology-style aliases.
    """
    canonical_subject = canonicalize_entity_label(
        subject_label, default=subject_label if isinstance(subject_label, str) else ""
    )
    canonical_object = canonicalize_entity_label(
        object_label, default=object_label if isinstance(object_label, str) else ""
    )
    return VALID_RELATIONS.get((canonical_subject, canonical_object), set())


def is_valid_relation(
    predicate: str,
    subject_label: str,
    object_label: str,
) -> bool:
    """Return whether a predicate satisfies the edil domain/range constraints."""
    return predicate in get_valid_relation_predicates(subject_label, object_label)


##########################
# Data Loading Functions #
##########################

def load_json_data(file_path: str) -> dict:
    """
    Load JSON data from a file and return it as a dictionary.

    :param file_path: The path to the JSON file.
    :return: A dictionary containing the JSON data.
    """

    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_csv_data(file_path: str) -> pd.DataFrame:
    """
    Load CSV data from a file and return it as a pandas DataFrame.

    :param file_path: The path to the CSV file.
    :return: A pandas DataFrame containing the CSV data.
    """

    return pd.read_csv(file_path)

def save_json_data(data: dict, file_path: str, encoding: str = "utf-8", indent: int = 4) -> None:
    """
    Save a dictionary as JSON data to a file. If the directory does not exist, it will be created.

    :param data: The dictionary to save as JSON.
    :param file_path: The path to the JSON file where the data will be saved.
    :param encoding: The encoding to use for the file.
    :param indent: The indentation for the JSON data.
    :return: None
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w", encoding=encoding) as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)

def load_merge_json_data(file_paths: list) -> dict:
    """
    Load and merge multiple JSON files into a single dictionary.

    :param file_paths: A list of paths to the JSON files to be merged.
    :return: A dictionary containing the merged JSON data.
    """

    merged_data = {}
    for file_path in file_paths:
        data = load_json_data(file_path)
        merged_data.update(data)
    return merged_data

def load_merge_csv_data(file_paths: list) -> pd.DataFrame:
    """
    Load and merge multiple CSV files into a single pandas DataFrame.

    :param file_paths: A list of paths to the CSV files to be merged.
    :return: A pandas DataFrame containing the merged CSV data.
    """

    data_frames = [load_csv_data(file_path) for file_path in file_paths]
    return pd.concat(data_frames, ignore_index=True)


def safe_filename_component(value: str) -> str:
    """
    Convert provider/model identifiers and local paths into one filename segment.
    """
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")


def clean_llm_decoding_artifacts(text: str) -> str:
    """
    Remove byte-fallback artifacts emitted by some GGUF / SentencePiece backends.

    Some local servers can surface tokenizer byte fallback tokens as strings like
    "[UNK_BYTE_0xe29681...]" in generated text.  The byte sequence e2 96 81 is
    SentencePiece's whitespace marker, so it must become a real space rather
    than an empty string; otherwise section offsets drift from the source text.
    """
    if not isinstance(text, str):
        return text

    def replace_artifact(match: re.Match) -> str:
        try:
            decoded = bytes.fromhex(match.group(1)).decode("utf-8")
        except (UnicodeDecodeError, ValueError):
            return ""
        return " " if decoded == "\u2581" else decoded

    return LLM_BYTE_ARTIFACT_PATTERN.sub(replace_artifact, text)


def _normalize_for_span_alignment(text: str) -> tuple[str, list[int]]:
    """
    Collapse whitespace and ignore spaces adjacent to common punctuation.

    The returned index map links each normalized character back to its position
    in the original text, so matches can still produce source-text offsets.
    """
    normalized_chars: list[str] = []
    index_map: list[int] = []
    pending_space_idx: int | None = None

    for idx, char in enumerate(text):
        if char.isspace():
            if normalized_chars and pending_space_idx is None:
                pending_space_idx = idx
            continue

        if char in ",;:.":
            pending_space_idx = None
        elif pending_space_idx is not None and normalized_chars[-1] not in "([:":
            normalized_chars.append(" ")
            index_map.append(pending_space_idx)
            pending_space_idx = None
        else:
            pending_space_idx = None

        normalized_chars.append(char)
        index_map.append(idx)

    return "".join(normalized_chars), index_map


def find_text_span_in_source(source_text: str, predicted_text: str, search_start: int = 0) -> tuple[int, int, str] | None:
    """
    Find a predicted span in the original source text after decoding artifacts.

    Returns inclusive start/end offsets plus the exact source slice, or None when
    the model changed the text enough that exact alignment is not possible.
    """
    cleaned_text = clean_llm_decoding_artifacts(predicted_text).strip()
    if not cleaned_text:
        return None

    start = source_text.find(cleaned_text, max(search_start, 0))
    if start == -1:
        start = source_text.find(cleaned_text)
    if start == -1:
        normalized_source, source_index_map = _normalize_for_span_alignment(source_text)
        normalized_prediction, _ = _normalize_for_span_alignment(cleaned_text)
        normalized_search_start = 0
        if 0 < search_start < len(source_text):
            normalized_search_start = next(
                (
                    normalized_idx
                    for normalized_idx, source_idx in enumerate(source_index_map)
                    if source_idx >= search_start
                ),
                0,
            )

        normalized_start = normalized_source.find(normalized_prediction, normalized_search_start)
        if normalized_start == -1:
            normalized_start = normalized_source.find(normalized_prediction)
        if normalized_start != -1:
            normalized_end = normalized_start + len(normalized_prediction) - 1
            start = source_index_map[normalized_start]
            end = source_index_map[normalized_end]
            return start, end, source_text[start:end + 1]
        return None

    end = start + len(cleaned_text) - 1
    return start, end, source_text[start:end + 1]


###############################
# Device Management Functions #
###############################

def get_device() -> torch.device:
    """
    Get the available device (GPU or CPU) for PyTorch.

    :return: A torch.device object representing the available device.
    """

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if _xpu_is_available():
        return torch.device("xpu")
    return torch.device("cpu")

def _xpu_is_available() -> bool:
    """
    Check whether PyTorch can use an Intel XPU device, such as an Intel Arc GPU.

    :return: True when the torch.xpu backend is present and reports at least one available device.
    """

    try:
        return hasattr(torch, "xpu") and torch.xpu.is_available()
    except Exception:
        return False

def move_to_device(data, device: torch.device):
    """
    Move data to the specified device.

    :param data: The data to be moved (can be a tensor, list, or dictionary).
    :param device: The target device to move the data to.
    :return: The data moved to the specified device.
    """

    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, list):
        return [move_to_device(item, device) for item in data]
    elif isinstance(data, dict):
        return {key: move_to_device(value, device) for key, value in data.items()}
    else:
        raise TypeError(f"Unsupported data type: {type(data)}")
    
def print_device_info(device: torch.device) -> None:
    """
    Print information about the specified device.

    :param device: The device to print information about.
    :return: None
    """

    print(f"Device: {device}")
    if device.type == "cuda":
        try:
            print(f"CUDA device: {torch.cuda.get_device_name(0)}")
            print(f"CUDA device count: {torch.cuda.device_count()}")
        except Exception:
            pass
    elif device.type == "mps":
        print("MPS backend is active")
        try:
            print(f"MPS recommended max memory: {torch.mps.recommended_max_memory()}")
            print(f"MPS current allocated memory: {torch.mps.current_allocated_memory()}")
            print(f"MPS driver allocated memory: {torch.mps.driver_allocated_memory()}")
        except Exception:
            pass
    elif device.type == "xpu":
        print("XPU backend is active")
        try:
            print(f"XPU device: {torch.xpu.get_device_name(0)}")
            print(f"XPU device count: {torch.xpu.device_count()}")
        except Exception:
            pass
        try:
            free_memory, total_memory = torch.xpu.mem_get_info(0)
            print(f"XPU free memory: {free_memory}")
            print(f"XPU total memory: {total_memory}")
        except Exception:
            pass
        try:
            print(f"XPU allocated memory: {torch.xpu.memory_allocated(0)}")
            print(f"XPU reserved memory: {torch.xpu.memory_reserved(0)}")
        except Exception:
            pass
    
def maybe_empty_cache(device: torch.device) -> None:
    """
    Empty the GPU cache if the device is a GPU.

    :param device: The device to check for GPU and potentially empty the cache.
    :return: None
    """

    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        try:
            torch.mps.empty_cache()
        except Exception:
            pass
    elif device.type == "xpu":
        try:
            torch.xpu.empty_cache()
        except Exception:
            pass
    
    
#####################
# Seeding Functions #
#####################

def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if hasattr(torch, "mps") and torch.backends.mps.is_available():
        try:
            torch.mps.manual_seed(seed)
        except Exception:
            pass

    if _xpu_is_available():
        try:
            torch.xpu.manual_seed(seed)
            torch.xpu.manual_seed_all(seed)
        except Exception:
            pass

######################################
# SectionExtractor Utility Functions #
######################################

def parse_predicted_sections_to_gbie_format(data, metadata):
    """
    Convert the predicted sections from SectionExtractor modules output format to the GBIE format expected by the evaluation script.

    :param data: dict mapping pmid to predicted sections, where each section has "start", "end", and "text" fields.
    :param metadata: dict mapping pmid to article metadata, including "metadata" and "text" fields.
    :return: dict in GBIE format with "metadata", "text", and "sections" (list of dicts with "start_idx", "end_idx", "text_span", and "label") for each pmid.
    """
    ret = {}

    mapping = {e.split(":")[0]: e for e in SECTION_LABEL_LIST}

    for pmid, content in data.items():
        source_text = metadata[pmid]["text"]
        search_start = 0
        ret[pmid] = {
            "metadata": metadata[pmid]["metadata"],
            "text": source_text,
            "sections": []
        }
        for label, details in content["sections"].items():
            if label not in mapping:
                raise NameError(f"Illegal section label {label} for article {pmid}")
            aligned_span = find_text_span_in_source(
                source_text,
                details["text"],
                search_start=search_start,
            )
            if aligned_span is not None:
                start_idx, end_idx, text_span = aligned_span
                search_start = end_idx + 1
            else:
                start_idx = details["start"]
                end_idx = details["end"]
                text_span = clean_llm_decoding_artifacts(details["text"]).strip()
            ret[pmid]["sections"].append({
                "start_idx": start_idx,
                "end_idx": end_idx,
                "text_span": text_span,
                "label": mapping[label]
            })

    return ret


def realign_section_predictions_to_source_text(
    data: dict,
    metadata: dict,
    text_field: str = "text",
    metadata_field: str = "metadata",
    sections_field: str = "sections",
    strict: bool = True,
) -> dict:
    """
    Re-align legacy section predictions to the original source text.

    Older LLM/DSPy parsed prediction files were produced by computing section
    offsets and spans against the model output after delimiter removal.  This
    function rebuilds those parsed predictions so that every section span is
    anchored to the source record text from metadata/ground truth data.

    It accepts both already-parsed GBIE-style predictions:

        {"sections": [{"start_idx": ..., "end_idx": ..., "text_span": ..., "label": ...}]}

    and raw SectionExtractor-style predictions:

        {"sections": {"APP": {"start": ..., "end": ..., "text": ...}}}

    :param data: Prediction dictionary keyed by record id.
    :param metadata: Metadata/ground-truth dictionary keyed by record id, with source text.
    :param text_field: Field containing the source text in metadata and output.
    :param metadata_field: Field containing record metadata.
    :param sections_field: Field containing section predictions.
    :param strict: Whether to raise on malformed or missing records.
    :return: New prediction dictionary with source-backed section offsets/texts.
    """
    processed = {}

    for record_id, content in data.items():
        if not isinstance(content, dict):
            if strict:
                raise TypeError(f"Prediction record {record_id} is not a dictionary.")
            continue

        metadata_record = metadata.get(record_id)
        if not isinstance(metadata_record, dict):
            if strict:
                raise KeyError(f"Record {record_id} is missing from metadata.")
            metadata_record = {}

        source_text = metadata_record.get(text_field)
        if not isinstance(source_text, str):
            if strict:
                raise KeyError(f"Record {record_id} does not contain metadata text field '{text_field}'.")
            source_text = content.get(text_field, "")

        output_content = dict(content)
        output_content[text_field] = source_text
        if metadata_field in metadata_record:
            output_content[metadata_field] = metadata_record[metadata_field]

        output_content[sections_field] = _realign_record_sections_to_source_text(
            record_id=record_id,
            sections=content.get(sections_field, []),
            source_text=source_text,
            strict=strict,
        )
        processed[record_id] = output_content

    return processed


def post_process_section_predictions_with_source_text(
    predictions_path: str,
    metadata_path: str,
    output_path: str | None = None,
    text_field: str = "text",
    metadata_field: str = "metadata",
    sections_field: str = "sections",
    strict: bool = True,
    indent: int = 4,
) -> str:
    """
    Repair a legacy section prediction file and save the processed JSON.

    :param predictions_path: Path to the prediction JSON file to repair.
    :param metadata_path: Path to ground-truth/metadata JSON containing source texts.
    :param output_path: Optional destination path. Defaults to the input directory
        with filename '<prediction-stem>-processed.json'.
    :param text_field: Field containing the source text in metadata and output.
    :param metadata_field: Field containing record metadata.
    :param sections_field: Field containing section predictions.
    :param strict: Whether to raise on malformed or missing records.
    :param indent: JSON indentation for the saved file.
    :return: Path to the processed prediction file.
    """
    predictions = load_json_data(predictions_path)
    metadata = load_json_data(metadata_path)

    processed = realign_section_predictions_to_source_text(
        data=predictions,
        metadata=metadata,
        text_field=text_field,
        metadata_field=metadata_field,
        sections_field=sections_field,
        strict=strict,
    )

    if output_path is None:
        predictions_dir = os.path.dirname(predictions_path) or "."
        prediction_stem = os.path.splitext(os.path.basename(predictions_path))[0]
        output_path = os.path.join(predictions_dir, f"{prediction_stem}-processed.json")
    elif not os.path.dirname(output_path):
        output_path = os.path.join(".", output_path)

    save_json_data(processed, output_path, indent=indent)
    return output_path


def _realign_record_sections_to_source_text(
    record_id: str,
    sections,
    source_text: str,
    strict: bool,
) -> list[dict]:
    """
    Re-align one record's section list/dict to source text.
    """
    if isinstance(sections, dict):
        section_items = [
            _raw_section_to_gbie_section(label=label, details=details)
            for label, details in sections.items()
        ]
    elif isinstance(sections, list):
        section_items = [dict(section) for section in sections if isinstance(section, dict)]
        if strict and len(section_items) != len(sections):
            raise TypeError(f"Record {record_id} contains a non-dictionary section.")
    else:
        if strict:
            raise TypeError(f"Record {record_id} has unsupported sections type {type(sections)}.")
        section_items = []

    aligned_sections = []
    search_start = 0

    for section in section_items:
        label = section.get("label")
        tag = _section_label_to_tag(label)
        if tag is None:
            if strict:
                raise NameError(f"Illegal section label {label} for record {record_id}.")
            continue

        predicted_text = section.get("text_span", section.get("text", ""))
        aligned_span = find_text_span_in_source(
            source_text=source_text,
            predicted_text=predicted_text,
            search_start=search_start,
        )

        if aligned_span is not None:
            start_idx, end_idx, text_span = aligned_span
            search_start = end_idx + 1
        else:
            fallback_span = _fallback_source_span_from_section(source_text, section)
            if fallback_span is None:
                if strict:
                    raise ValueError(f"Could not align section {label} for record {record_id}.")
                continue
            start_idx, end_idx, text_span = fallback_span

        aligned_section = dict(section)
        aligned_section.pop("start", None)
        aligned_section.pop("end", None)
        aligned_section.pop("text", None)
        aligned_section["start_idx"] = start_idx
        aligned_section["end_idx"] = end_idx
        aligned_section["text_span"] = text_span
        aligned_section["label"] = _section_tag_to_label(tag)
        aligned_sections.append(aligned_section)

    return aligned_sections


def _raw_section_to_gbie_section(label: str, details) -> dict:
    """
    Convert one raw SectionExtractor section entry to parsed section shape.
    """
    section = dict(details) if isinstance(details, dict) else {"text": str(details)}
    section["label"] = _section_tag_to_label(label) or label

    if "start_idx" not in section and "start" in section:
        section["start_idx"] = section["start"]
    if "end_idx" not in section and "end" in section:
        section["end_idx"] = section["end"]
    if "text_span" not in section and "text" in section:
        section["text_span"] = section["text"]

    return section


def _fallback_source_span_from_section(source_text: str, section: dict) -> tuple[int, int, str] | None:
    """
    Fall back to the section's existing offsets when exact source alignment fails.
    """
    if not source_text:
        return None

    try:
        start_idx = int(section.get("start_idx", section.get("start")))
        end_idx = int(section.get("end_idx", section.get("end")))
    except (TypeError, ValueError):
        return None

    text_length = len(source_text)
    start_idx = max(0, min(start_idx, text_length - 1))
    end_idx = max(0, min(end_idx, text_length - 1))
    if end_idx < start_idx:
        return None

    start_idx, end_idx = _trim_span_whitespace(source_text, start_idx, end_idx)
    if end_idx < start_idx:
        return None

    return start_idx, end_idx, source_text[start_idx:end_idx + 1]


def post_process_predicted_sections(
    data: dict,
    text_field: str = "text",
    sections_field: str = "sections",
    marker_window: int = 10,
    min_section_chars: int = 2,
    merge_same_label_gap: int = 8,
    snap_start_to_marker: bool = True,
    snap_end_to_next_marker: bool = True,
    merge_same_label_sections: bool = True,
    resolve_overlaps: bool = True,
) -> dict:
    """
    Post-process parsed/GBIE-format section predictions.

    This utility is model-agnostic and can be applied to predictions produced by
    HFSectionExtractor, LLMSectionExtractor, DspySectionExtractor, or any other
    module that returns:

        {
            "<record_id>": {
                "text": "...",
                "sections": [
                    {
                        "start_idx": int,
                        "end_idx": int,
                        "text_span": str,
                        "label": "APP: ...",
                        ...
                    }
                ]
            }
        }

    The post-processing is intentionally conservative:
        - clamp spans to the source text
        - trim surrounding whitespace
        - snap section starts to nearby explicit markers (APP:, APR:, T:, EO:, DS:, AI:, A:)
        - optionally cut a section before the next explicit marker
        - discard tiny fragments
        - optionally merge adjacent same-label sections
        - optionally remove overlaps by trimming the previous span before the next one

    :param data: Parsed/GBIE-format predictions.
    :param text_field: Field containing the source text for each record.
    :param sections_field: Field containing the section list for each record.
    :param marker_window: Maximum character distance for snapping starts to nearby markers.
    :param min_section_chars: Minimum stripped section length to keep.
    :param merge_same_label_gap: Maximum gap between same-label sections to merge.
    :param snap_start_to_marker: Whether to snap starts to nearby same-label section markers.
    :param snap_end_to_next_marker: Whether to trim a span before the next explicit section marker.
    :param merge_same_label_sections: Whether to merge adjacent same-label sections.
    :param resolve_overlaps: Whether to trim overlapping different-label sections.
    :return: A new parsed/GBIE-format prediction dict with post-processed sections.
    """
    processed = {}

    for record_id, content in data.items():
        source_text = content.get(text_field, content.get("text", ""))
        output_content = dict(content)
        markers = _find_section_markers(source_text)

        sections = []
        for section in content.get(sections_field, []):
            normalized = _normalise_predicted_section(
                section=section,
                source_text=source_text,
                markers=markers,
                marker_window=marker_window,
                min_section_chars=min_section_chars,
                snap_start_to_marker=snap_start_to_marker,
                snap_end_to_next_marker=snap_end_to_next_marker,
            )
            if normalized is not None:
                sections.append(normalized)

        sections.sort(key=lambda item: (item["start_idx"], item["end_idx"], item["label"]))

        if merge_same_label_sections:
            sections = _merge_adjacent_same_label_sections(
                sections=sections,
                source_text=source_text,
                max_gap=merge_same_label_gap,
                min_section_chars=min_section_chars,
            )

        if resolve_overlaps:
            sections = _trim_overlapping_sections(
                sections=sections,
                source_text=source_text,
                min_section_chars=min_section_chars,
            )

        output_content[sections_field] = sections
        processed[record_id] = output_content

    return processed


def _normalise_predicted_section(
    section: dict,
    source_text: str,
    markers: list[dict],
    marker_window: int,
    min_section_chars: int,
    snap_start_to_marker: bool,
    snap_end_to_next_marker: bool,
) -> dict | None:
    """
    Normalise one predicted section span against its source text.

    :param section: Predicted section dict.
    :param source_text: Source text for the current record.
    :param markers: Explicit section markers found in source_text.
    :param marker_window: Maximum character distance for start snapping.
    :param min_section_chars: Minimum stripped section length to keep.
    :param snap_start_to_marker: Whether to snap starts to nearby same-label markers.
    :param snap_end_to_next_marker: Whether to trim before the next explicit marker.
    :return: Normalised section dict, or None when the section should be discarded.
    """
    if not source_text:
        return None

    try:
        start_idx = int(section["start_idx"])
        end_idx = int(section["end_idx"])
    except (KeyError, TypeError, ValueError):
        return None

    label = section.get("label")
    tag = _section_label_to_tag(label)
    if tag is None:
        return None

    text_length = len(source_text)
    start_idx = max(0, min(start_idx, text_length - 1))
    end_idx = max(0, min(end_idx, text_length - 1))
    if end_idx < start_idx:
        return None

    start_idx, end_idx = _trim_span_whitespace(source_text, start_idx, end_idx)

    if snap_start_to_marker:
        marker = _nearest_same_label_marker(
            markers=markers,
            tag=tag,
            start_idx=start_idx,
            marker_window=marker_window,
        )
        if marker is not None:
            start_idx = marker["start"]

    if snap_end_to_next_marker:
        next_marker = _next_section_marker(markers, start_idx)
        if next_marker is not None and next_marker["start"] <= end_idx:
            end_idx = next_marker["start"] - 1

    if end_idx < start_idx:
        return None

    start_idx, end_idx = _trim_span_whitespace(source_text, start_idx, end_idx)
    if end_idx < start_idx:
        return None

    text_span = source_text[start_idx:end_idx + 1]
    if len(text_span.strip()) < min_section_chars:
        return None

    normalized = dict(section)
    normalized["start_idx"] = start_idx
    normalized["end_idx"] = end_idx
    normalized["text_span"] = text_span
    normalized["label"] = _section_tag_to_label(tag)
    return normalized


def _find_section_markers(text: str) -> list[dict]:
    """
    Find explicit section markers in source text.

    :param text: Source text.
    :return: List of marker dicts with tag, label, start, and end fields.
    """
    if not text:
        return []

    tags = sorted((label.split(":")[0] for label in SECTION_LABEL_LIST), key=len, reverse=True)
    pattern = re.compile(
        rf"(?<![A-Za-z0-9])({'|'.join(re.escape(tag) for tag in tags)})\s*:",
        flags=re.IGNORECASE,
    )

    markers = []
    for match in pattern.finditer(text):
        tag = match.group(1).upper()
        label = _section_tag_to_label(tag)
        if label is None:
            continue
        markers.append(
            {
                "tag": tag,
                "label": label,
                "start": match.start(),
                "end": match.end() - 1,
            }
        )
    return markers


def _section_tag_to_label(tag: str | None) -> str | None:
    """
    Convert a short section tag to the canonical full label.

    :param tag: Short section tag, e.g. APP.
    :return: Full section label, or None when unknown.
    """
    if tag is None:
        return None
    mapping = {label.split(":")[0]: label for label in SECTION_LABEL_LIST}
    return mapping.get(str(tag).upper())


def _section_label_to_tag(label: str | None) -> str | None:
    """
    Convert a full section label or short tag to its short tag.

    :param label: Full label or short tag.
    :return: Short section tag, or None when unknown.
    """
    if label is None:
        return None
    label = str(label)
    mapping = {full_label.split(":")[0]: full_label for full_label in SECTION_LABEL_LIST}
    if label in SECTION_LABEL_LIST:
        return label.split(":")[0]
    tag = label.split(":", 1)[0].upper()
    if tag in mapping:
        return tag
    return None


def _trim_span_whitespace(text: str, start_idx: int, end_idx: int) -> tuple[int, int]:
    """
    Trim surrounding whitespace from an inclusive character span.

    :param text: Source text.
    :param start_idx: Inclusive start index.
    :param end_idx: Inclusive end index.
    :return: Trimmed (start_idx, end_idx).
    """
    while start_idx <= end_idx and text[start_idx].isspace():
        start_idx += 1
    while end_idx >= start_idx and text[end_idx].isspace():
        end_idx -= 1
    return start_idx, end_idx


def _nearest_same_label_marker(markers: list[dict], tag: str, start_idx: int, marker_window: int) -> dict | None:
    """
    Find the closest same-label marker around a predicted section start.

    :param markers: Explicit section markers found in source text.
    :param tag: Short section tag for the predicted section.
    :param start_idx: Predicted section start.
    :param marker_window: Maximum character distance for snapping.
    :return: Closest marker dict, or None.
    """
    candidates = [
        marker
        for marker in markers
        if marker["tag"] == tag and abs(marker["start"] - start_idx) <= marker_window
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda marker: abs(marker["start"] - start_idx))


def _next_section_marker(markers: list[dict], start_idx: int) -> dict | None:
    """
    Find the next explicit section marker after a span start.

    :param markers: Explicit section markers found in source text.
    :param start_idx: Section start index.
    :return: Next marker dict, or None.
    """
    candidates = [marker for marker in markers if marker["start"] > start_idx]
    if not candidates:
        return None
    return min(candidates, key=lambda marker: marker["start"])


def _merge_adjacent_same_label_sections(sections: list[dict], source_text: str, max_gap: int, min_section_chars: int) -> list[dict]:
    """
    Merge adjacent same-label sections separated by a small gap.

    :param sections: Sorted section predictions.
    :param source_text: Source text for rebuilding text_span.
    :param max_gap: Maximum gap allowed between same-label spans.
    :param min_section_chars: Minimum stripped section length to keep.
    :return: Merged section list.
    """
    merged = []

    for section in sections:
        if not merged:
            merged.append(dict(section))
            continue

        current = merged[-1]
        gap = section["start_idx"] - current["end_idx"] - 1
        if section["label"] == current["label"] and gap <= max_gap:
            current["end_idx"] = max(current["end_idx"], section["end_idx"])
            current["text_span"] = source_text[current["start_idx"]:current["end_idx"] + 1]
            if "confscore" in current and "confscore" in section:
                current["confscore"] = min(float(current["confscore"]), float(section["confscore"]))
        else:
            merged.append(dict(section))

    return [
        section for section in merged
        if len(section["text_span"].strip()) >= min_section_chars
    ]


def _trim_overlapping_sections(sections: list[dict], source_text: str, min_section_chars: int) -> list[dict]:
    """
    Trim overlaps by ending each previous section before the next section starts.

    :param sections: Sorted section predictions.
    :param source_text: Source text for rebuilding text_span.
    :param min_section_chars: Minimum stripped section length to keep.
    :return: Non-overlapping section list.
    """
    if not sections:
        return []

    trimmed = []
    for section in sections:
        section = dict(section)
        if trimmed and section["start_idx"] <= trimmed[-1]["end_idx"]:
            previous = trimmed[-1]
            previous["end_idx"] = section["start_idx"] - 1
            if previous["end_idx"] >= previous["start_idx"]:
                previous["start_idx"], previous["end_idx"] = _trim_span_whitespace(
                    source_text,
                    previous["start_idx"],
                    previous["end_idx"],
                )
                previous["text_span"] = source_text[previous["start_idx"]:previous["end_idx"] + 1]
            if previous["end_idx"] < previous["start_idx"] or len(previous["text_span"].strip()) < min_section_chars:
                trimmed.pop()

        if len(section["text_span"].strip()) >= min_section_chars:
            trimmed.append(section)

    return trimmed


######################################
# EntityRecognizer Utility Functions #
######################################

def link_drug_entity_to_aifa(entity: dict) -> dict:
    """
    Link drug entities to AIFA (Agenzia Italiana del Farmaco) database.
    """
    if 'text_span' not in entity:
        raise ValueError("Entity dict must contain a 'text_span' field for AIFA linking.")
    if 'label' not in entity or entity['label'].lower() != 'drug':
        print("Entity dict must contain a 'label' field with value 'drug' for AIFA linking.")
        entity['link'] = []
        return entity

    def query_aifa(drug_name: str) -> dict:
        """
        Queries the AIFA API for drug information based on the specified drug name.
        
        Parameters:
            drug_name (str): The name of the drug to search for.
        
        Returns:
            dict: A dictionary containing the JSON response data.
        """
        # API endpoint as determined from the HAR data
        base_url = "https://api.aifa.gov.it/aifa-bdf-eif-be/1.0.0/formadosaggio/ricerca"
        
        # Query parameters: spell correction is enabled and we request the first page (page 0)
        params = {
            "query": drug_name,
            "spellingCorrection": "true",
            "page": "0"
        }
        
        # Headers to simulate the browser request (as seen in the HAR)
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:137.0) Gecko/20100101 Firefox/137.0",
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": "en-GB,en;q=0.5",
            "Origin": "https://medicinali.aifa.gov.it",
            "Referer": "https://medicinali.aifa.gov.it/"
        }
        
        # Optionally add a Correlation-Id header (the HAR shows one being sent)
        headers["Correlation-Id"] = str(uuid.uuid4())
        
        try:
            # Send the GET request with parameters and headers
            response = requests.get(base_url, params=params, headers=headers)
            response.raise_for_status()  # Raises an HTTPError if the status is 4xx or 5xx
            
            # Parse the JSON response into a dictionary and return it
            data = response.json()
            return data

        except requests.RequestException as e:
            print(f"Request failed for drug {drug_name}:", e)
            return {}
        except ValueError as ve:
            # This will catch JSON decoding errors
            print(f"JSON decode error for drug {drug_name}:", ve)
            return {}

    def retrieve_drug_info(drug_name: str) -> dict:
        """
        Retrieves and processes drug information from the AIFA API based on the provided drug name.
        """

        result = query_aifa(drug_name)
        
        if not result:
            print(f"No data returned for drug {drug_name}")
            return {}, {}

        if result['status'] != 200:
            print(f"Error in response for drug {drug_name}:", result['status'])
            return {}, result
        
        if 'data' not in result:
            print(f"No data found in response for drug {drug_name}")
            return {}, result
        
        if 'content' not in result['data']:
            print(f"No drugs found for drug {drug_name}")
            return {}, result
        
        drugs = result['data']['content']
        if len(drugs) == 0:
            print(f"No drugs found for the given drug {drug_name} ")
            return {}, result
        
        descrizioneFormaDosaggio = set()
        principiAttiviIt = set()
        codice_atc = set()
        descrizione_atc = set()
        codice_medicinale = set()
        denominazione_medicinale = set()

        for d in drugs:
            if 'descrizioneFormaDosaggio' in d:
                descrizioneFormaDosaggio.add(d['descrizioneFormaDosaggio'])
            if 'principiAttiviIt' in d:
                for item in d['principiAttiviIt']:
                    principiAttiviIt.add(item)
            if 'codiceAtc' in d:
                for item in d['codiceAtc']:
                    codice_atc.add(item)
            if 'descrizioneAtc' in d:
                for item in d['descrizioneAtc']:
                    descrizione_atc.add(item)
            if 'medicinale' in d:
                if 'codiceMedicinale' in d['medicinale']:
                    codice_medicinale.add(d['medicinale']['codiceMedicinale'])
                if 'denominazioneMedicinale' in d['medicinale']:
                    denominazione_medicinale.add(d['medicinale']['denominazioneMedicinale'])

        # Convert sets to lists for JSON serialization
        # Create a dictionary to hold the processed drug information
        drug_info = {
            "descrizioneFormaDosaggio": list(descrizioneFormaDosaggio),
            "principiAttiviIt": list(principiAttiviIt),
            "codice_atc": list(codice_atc),
            "descrizione_atc": list(descrizione_atc),
            "codice_medicinale": list(codice_medicinale),
            "denominazione_medicinale": list(denominazione_medicinale)
        }
        return drug_info, result
    
    drug_info, result = retrieve_drug_info(entity['text_span'])
    codice_atc, descrizione_atc = drug_info.get('codice_atc', []), drug_info.get('descrizione_atc', [])
    if not codice_atc and not descrizione_atc:
        print(f"No drug information found in AIFA for entity: {entity}")
        entity['link'] = []
        return entity

    entity['link'] = [
        codice_atc[0],
        descrizione_atc[0],
        f"https://www.codifa.it/ATC/{codice_atc[0]}",
        1.0 # Assuming a default score of 1.0 for AIFA links
    ]
    return entity


########################################
# GutBrainIE Data Processing Functions #
########################################

def build_doc_id_to_aacs_id_mapping(content: dict, content_field: str, metadata: pd.DataFrame, metadata_field: str, metadata_text_field: str = "DEA_ANMNS", document_id_field: str = "document_id", mention_text_field: str = "mention_text") -> dict:
    """
    Build a mapping from document IDs to accession IDs.

    :param content: A dictionary containing the content of a paper in Metatron format.
    :param content_field: The field name in content containing the list of annotations (e.g., "tags", "concepts").
    :param metadata: A DataFrame containing metadata for the papers.
    :param metadata_field: The field name in the metadata DataFrame corresponding to the accession IDs.
    :param metadata_text_field: The field name in the metadata DataFrame containing the text to search in (default: "DEA_ANMNS").
    :param document_id_field: The field name in each annotation containing the document ID (default: "document_id").
    :param mention_text_field: The field name in each annotation containing the mention text (default: "mention_text").
    :return: A dictionary mapping document IDs to accession IDs.
    """
    # Build a mapping of document_id to list of mention_texts
    id_to_mentions = {}
    for entry in content[content_field]:
        doc_id = entry[document_id_field]
        mention_text = entry[mention_text_field]
        if doc_id not in id_to_mentions:
            id_to_mentions[doc_id] = []
        id_to_mentions[doc_id].append(mention_text)
    
    # Initialize mapping for each document_id
    doc_id_to_aacs = {}
    
    # For each row in metadata, try to match document_ids
    for index, row in metadata.iterrows():
        aacs_id = row[metadata_field]
        text = row[metadata_text_field]
        
        # Check each document_id
        for doc_id, mention_texts in id_to_mentions.items():
            # Check if all mention_texts are in the metadata text
            match = True
            for mention_text in mention_texts:
                if text.find(mention_text) == -1:
                    match = False
                    break
            
            # If all mention_texts are found, map document_id to aacs_id
            if match:
                doc_id_to_aacs[doc_id] = aacs_id
    
    return doc_id_to_aacs

def metatron_sections_to_gbie_format(content: dict, doc_ids_to_aacs_ids: dict, dict_field: str="tags", metadata: pd.DataFrame = None) -> dict:
    """
    Convert a Metatron-format content dictionary for annotated sections to the GutBrainIE format.
    Metatron-format content dictionaries have a "<dict_field>" field containing a list of tag dictionaries, each with "start", "end", "tag", and "mention_text" keys. 
    This function transforms that list into a "sections" field containing a list of section dictionaries, each with "start_idx", "end_idx", "text_span", and "label" keys, where "label" is derived from "tag" and "text_span" is derived from "mention_text". 
    The function also maps document IDs to accession IDs using the provided mapping.

    :param content: A dictionary containing the content of a paper in Metatron format, including a "<dict_field>" field with section annotations. 
    :param doc_ids_to_aacs_ids: A dictionary mapping document IDs to accession IDs.
    :param dict_field: The field name containing the section annotations.
    :param metadata: A DataFrame containing metadata for the papers. If provided, this will be used to enrich the section annotations with additional metadata information.
    :return: A dictionary containing the content in GutBrainIE format.
    """
    ret = {}
    for annotation in content[dict_field]:
        document_id = annotation["document_id"]
        start = annotation["start"]
        end = annotation["stop"]
        location = annotation["mention_location"]
        text = annotation["mention_text"]
        label = annotation["tag"]
        aacs_id = doc_ids_to_aacs_ids[document_id]
        if aacs_id not in ret:
            ret[aacs_id] = {}
        if metadata is not None:
            ret[aacs_id]["metadata"] = {
                "ID_patient": str(metadata.loc[metadata["ID_AACS"] == aacs_id].iloc[0]["IDrandom_pz_GR"])
            }
            ret[aacs_id]["text"] = metadata.loc[metadata["ID_AACS"] == aacs_id].iloc[0]["DEA_ANMNS"]
        if "sections" not in ret[aacs_id]:
            ret[aacs_id]["sections"] = []
        ret[aacs_id]["sections"].append({
            "start_idx": start,
            "end_idx": end,
            "location": location,
            "text_span": text,
            "label": label
        })
    return ret

def _parse_concept_uri(concept_uri: str) -> str:
    """
    Parse a concept URI to extract the ontology prefix.

    :param concept_uri: The concept URI to parse.
    :return: The extracted ontology prefix, or the original URI if the URI does not match any defined pattern.
    """
    if re.search(r'DB\d{5}', concept_uri):
        return f"http://go.drugbank.com/drugs/{concept_uri}"
    elif concept_uri == "66951008":
        return "http://snomed.info/id/66951008"
    else:
        return concept_uri
    
def metatron_entities_to_gbie_format(content: dict, doc_ids_to_aacs_ids: dict, dict_field: str="concepts", metadata: pd.DataFrame = None) -> dict:
    """
    Convert a Metatron-format content dictionary for annotated entities to the GutBrainIE format.
    Metatron-format content dictionaries have a "<dict_field>" field containing a list of concept dictionaries, each with "start", "end", "concept_uri", "concept_name", "area", and "mention_text" keys.
    This function transforms that list into an entity list with "start_idx", "end_idx", "text_span", "label", and "uri" keys.
    The function also maps document IDs to accession IDs using the provided mapping.

    :param content: A dictionary containing the content of a paper in Metatron format, including a "<dict_field>" field with entity annotations.
    :param doc_ids_to_aacs_ids: A dictionary mapping document IDs to accession IDs.
    :param dict_field: The field name containing the entity annotations.
    :param metadata: A DataFrame containing metadata for the papers. If provided, this will be used to enrich the entity annotations with additional metadata information.
    :return: A dictionary containing the content in GutBrainIE format.
    """
    ret = {}
    for annotation in content[dict_field]:
        document_id = annotation["document_id"]
        start = annotation["start"]
        end = annotation["stop"]
        location = annotation["mention_location"]
        text = annotation["mention_text"]
        label = annotation["area"]
        concept_uri = _parse_concept_uri(annotation["concept_url"])
        concept_name = annotation["concept_name"]
        aacs_id = doc_ids_to_aacs_ids[document_id]
        if aacs_id not in ret:
            ret[aacs_id] = {}
        if metadata is not None:
            ret[aacs_id]["metadata"] = {
                "ID_patient": str(metadata.loc[metadata["ID_AACS"] == aacs_id].iloc[0]["IDrandom_pz_GR"])
            }
            ret[aacs_id]["text"] = metadata.loc[metadata["ID_AACS"] == aacs_id].iloc[0]["DEA_ANMNS"]
        if "entities" not in ret[aacs_id]:
            ret[aacs_id]["entities"] = []
        ret[aacs_id]["entities"].append({
            "start_idx": start,
            "end_idx": end,
            "text_span": text,
            "label": label,
            "concept_name": concept_name,
            "uri": [concept_uri, concept_name],
            "location": location
        })
    return ret

def metatron_tags_to_gbie_format(content: dict, doc_ids_to_aacs_ids: dict, dict_field: str="tags", metadata: pd.DataFrame = None) -> dict:
    """
    Convert a Metatron-format content dictionary for annotated tags to the GutBrainIE format.
    Metatron-format content dictionaries have a "<dict_field>" field containing a list of concept dictionaries, each with "start", "end", "concept_uri", "concept_name", "area", and "mention_text" keys.
    This function transforms that list into an entity list with "start_idx", "end_idx", "text_span", "label", and "uri" keys.
    The function also maps document IDs to accession IDs using the provided mapping.

    :param content: A dictionary containing the content of a paper in Metatron format, including a "<dict_field>" field with entity annotations.
    :param doc_ids_to_aacs_ids: A dictionary mapping document IDs to accession IDs.
    :param dict_field: The field name containing the entity annotations.
    :param metadata: A DataFrame containing metadata for the papers. If provided, this will be used to enrich the entity annotations with additional metadata information.
    :return: A dictionary containing the content in GutBrainIE format.
    """
    ret = {}
    for annotation in content[dict_field]:
        document_id = annotation["document_id"]
        start = annotation["start"]
        end = annotation["stop"]
        location = annotation["mention_location"]
        text = annotation["mention_text"]
        label = annotation["tag"]
        aacs_id = doc_ids_to_aacs_ids[document_id]
        if aacs_id not in ret:
            ret[aacs_id] = {}
        if metadata is not None:
            ret[aacs_id]["metadata"] = {
                "ID_patient": str(metadata.loc[metadata["ID_AACS"] == aacs_id].iloc[0]["IDrandom_pz_GR"])
            }
            ret[aacs_id]["text"] = metadata.loc[metadata["ID_AACS"] == aacs_id].iloc[0]["DEA_ANMNS"]
        if "entities" not in ret[aacs_id]:
            ret[aacs_id]["entities"] = []
        ret[aacs_id]["entities"].append({
            "start_idx": start,
            "end_idx": end,
            "text_span": text,
            "label": label,
            "location": location
        })
    return ret
        
def metatron_to_gbie_format(sections_content: dict, entities_content: dict, doc_ids_to_aacs_ids: dict, sections_dict_field: str="tags", entities_dict_field: str="concepts", metadata: pd.DataFrame = None) -> dict:
    """
    Convert a Metatron-format content dictionary for annotated sections and entities to the GutBrainIE format.
    This function combines the transformations performed by metatron_sections_to_gbie_format and metatron_entities_to_gbie_format to produce a unified output in GutBrainIE format.

    :param sections_content: A dictionary containing the section annotations in Metatron format.
    :param entities_content: A dictionary containing the entity annotations in Metatron format.
    :param doc_ids_to_aacs_ids: A dictionary mapping document IDs to accession IDs.
    :param doc_ids_to_aacs_ids: A dictionary mapping document IDs to accession IDs.
    :param sections_dict_field: The field name containing the section annotations.
    :param entities_dict_field: The field name containing the entity annotations.
    :param metadata: A DataFrame containing metadata for the papers. If provided, this will be used to enrich the annotations with additional metadata information.
    :return: A dictionary containing the content in GutBrainIE format, with both sections and entities.
    """
    sections_ret = metatron_sections_to_gbie_format(sections_content, doc_ids_to_aacs_ids, dict_field=sections_dict_field, metadata=metadata)
    entities_ret = metatron_entities_to_gbie_format(entities_content, doc_ids_to_aacs_ids, dict_field=entities_dict_field, metadata=metadata)
    ret = {}
    for aacs_id in set(sections_ret.keys()).union(entities_ret.keys()):
        ret[aacs_id] = {}
        if aacs_id in sections_ret:
            ret[aacs_id]["sections"] = sections_ret[aacs_id].get("sections", [])
            if "metadata" in sections_ret[aacs_id]:
                ret[aacs_id]["metadata"] = sections_ret[aacs_id]["metadata"]
            if "text" in sections_ret[aacs_id]:
                ret[aacs_id]["text"] = sections_ret[aacs_id]["text"]
        if aacs_id in entities_ret:
            ret[aacs_id]["entities"] = entities_ret[aacs_id].get("entities", [])
            if "metadata" in entities_ret[aacs_id] and "metadata" not in ret[aacs_id]:
                ret[aacs_id]["metadata"] = entities_ret[aacs_id]["metadata"]
            if "text" in entities_ret[aacs_id] and "text" not in ret[aacs_id]:
                ret[aacs_id]["text"] = entities_ret[aacs_id]["text"]
    return ret

def parse_metatron():
    import json
    import pandas as pd

    #sections = json.load(open("../../data/annotations/metatron/sections.json", "r", encoding="utf-8"))
    entities = json.load(open("../../data/annotations/metatron/entities_secondPhase.json", "r", encoding="utf-8"))
    metadata = pd.read_csv("../../data/raw/DEESCALATE_entrambiPS_soloAnamnesi.txt", sep="\t")

    #doc_id_to_aacs_id_sections = build_doc_id_to_aacs_id_mapping(
    #    content=sections,
    #    content_field="tags",
    #    metadata=metadata,
    #    metadata_field="ID_AACS",
    #    metadata_text_field="DEA_ANMNS",
    #    document_id_field="document_id",
    #    mention_text_field="mention_text"
    #)

    doc_id_to_aacs_id_entities = build_doc_id_to_aacs_id_mapping(
        content=entities,
        content_field="concepts",
        metadata=metadata,
        metadata_field="ID_AACS",
        metadata_text_field="DEA_ANMNS",
        document_id_field="document_id",
        mention_text_field="mention_text"
    )

    #doc_id_to_aacs_id = {**doc_id_to_aacs_id_sections, **doc_id_to_aacs_id_entities}
    doc_id_to_aacs_id = {**doc_id_to_aacs_id_entities}
    print(f"Built mapping for {len(doc_id_to_aacs_id)} document IDs to accession IDs.")

    #doc_id_to_aacs_id = json.load(open("../../data/annotations/metatron/doc_id_to_aacs_id.json", "r", encoding="utf-8"))

    #sections_gbie_format = dict(sorted(metatron_sections_to_gbie_format(
    #    content=sections,
    #    doc_ids_to_aacs_ids=doc_id_to_aacs_id,
    #    dict_field="tags",
    #    metadata=metadata
    #).items()))
    #for k, v in sections_gbie_format.items():
    #    print(f"AACS ID: {k}")
    #    print(f"Metadata: {v.get('metadata', {})}")
    #    print(f"Text: {v.get('text', '')[:100]}...")
    #    print(f"Sections: {v.get('sections', [])[:2]}")  # Print only the first 2 sections for brevity
    #    print()
    #    break

    entities_gbie_format = dict(sorted(metatron_entities_to_gbie_format(
        content=entities,
        doc_ids_to_aacs_ids=doc_id_to_aacs_id,
        dict_field="concepts",
        metadata=metadata
    ).items()))
    for k, v in entities_gbie_format.items():
        print(f"AACS ID: {k}")
        print(f"Metadata: {v.get('metadata', {})}")
        print(f"Text: {v.get('text', '')[:100]}...")
        print(f"Entities: {v.get('entities', [])[:2]}")  # Print only the first 2 entities for brevity
        print()
        break

    #with open("../../data/annotations/parsed/sections_gbie_format.json", "w", encoding="utf-8") as f:
    #    json.dump(sections_gbie_format, f, ensure_ascii=False, indent=4)
    #with open("../../data/annotations/parsed/entities_gbie_format.json", "w", encoding="utf-8") as f:
    with open("../../data/annotations/parsed/entities_secondPhase.json", "w", encoding="utf-8") as f:
        json.dump(entities_gbie_format, f, ensure_ascii=False, indent=4)

def parse_sections():
    filepath = "predictions/HFSectionExtractor/parsed/sections_IT_HF_medBIT-train-real-synthetic.json"
    predictions = load_json_data(filepath)

    post_processed = post_process_predicted_sections(
        predictions,
        marker_window=10,
        min_section_chars=5,
        merge_same_label_gap=8,
    )

    save_json_data(post_processed, filepath.replace(".json", "_postprocessed.json"))

def post_process_all_sections():
    import os
    paths = [
        "C:/Users/marti/Desktop/DEESCALATE/branch/predictions/DspySectionExtractorTestSet/parsed",
        "C:/Users/marti/Desktop/DEESCALATE/branch/predictions/DspySectionExtractor/parsed",
        "C:/Users/marti/Desktop/DEESCALATE/branch/predictions/HFSectionExtractorTestSet/parsed", 
        "C:/Users/marti/Desktop/DEESCALATE/branch/predictions/LLMSectionExtractor/parsed",
        "C:/Users/marti/Desktop/DEESCALATE/branch/predictions/LLMSectionExtractorTestSet/parsed"
    ]

    for path in paths:
        for filename in os.listdir(path):
            if filename.endswith("postprocessed.json"):
                os.remove(os.path.join(path, filename))
                print(f"Removed existing post-processed file: {filename}")

    for path in paths:
        for filename in os.listdir(path):
            if filename.endswith(".json") and "postprocessed" not in filename:
                print(f"Post-processing {filename}...")
                output_path = path.replace("parsed", "postprocessed")
                if not os.path.exists(output_path):
                    os.makedirs(output_path)
                post_process_section_predictions_with_source_text(
                    predictions_path=os.path.join(path, filename),
                    metadata_path="C:/Users/marti/Desktop/DEESCALATE/branch/data/annotations/parsed/sections_metadata.json",
                    output_path=os.path.join(output_path, filename)
                )
