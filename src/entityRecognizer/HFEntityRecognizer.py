#!/usr/bin/env python3
"""
HFEntityRecognizer.py

Transformer-based named entity recognition (NER) module that jointly extracts and classifies entities using entity-type-specific BIO tagging.
Organized into five classes:
    - EntityRecognitionData : Dataset wrapping for NER
    - EntityRecognitionTrainer : training loop and inference pipeline for NER
    - NERExtractionEvaluator : span+label precision / recall / F1 evaluation
    - NERBIOTokenizer : entity-type-specific BIO preprocessing (imported from Tokenizers module)

EntityRecognitionData inherits from BaseTermDataset and EntityRecognitionTrainer inherits from BaseTrainer (both defined in HFTermExtractor), keeping data-loading and training consistent across tasks.

NER extends term extraction by replacing the single B-term / I-term tag pair with entity-type-specific variants (for example B-Materiale, I-Prodotto edilizio, B-Geometria_valore), allowing the model to simultaneously locate entity spans and assign semantic categories in a single forward pass.
Inference produces entity dicts with the actual label field populated rather than the generic "term" placeholder used by TermExtractionTrainer.

Entry point: run with --help to see CLI options.
"""

import argparse
import os
from contextlib import nullcontext
import sys

import torch
from sklearn.metrics import precision_recall_fscore_support
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForTokenClassification, AutoTokenizer

# Allow unsupported MPS ops to fall back to CPU instead of crashing
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
# Disable HuggingFace tokenizer parallelism to avoid fork-related warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Add workspace root to Python path for relative imports
workspace_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, workspace_root)

# Import utilities implemented in utils.py
from src.utils.utils import (
    ENTITY_LABEL_LIST as LABEL_LIST,
    get_device,
    load_json_data,
    load_merge_json_data,
    maybe_empty_cache,
    move_to_device,
    print_device_info,
    save_json_data,
    seed_everything,
)

# Import shared base classes from the term extraction module
from src.termExtractor.HFTermExtractor import BaseTermDataset, BaseTrainer

# Import the NER BIO tokenizer
from src.termExtractor.Tokenizers import NERBIOTokenizer  # noqa: F401  (re-exported for callers)


#############
# Constants #
#############

# NER_BIO_LABELS is re-exported here for convenience so callers do not need to import from utils
from src.utils.utils import NER_BIO_LABELS  # noqa: F401  (re-exported for callers)

# Import reusable provider infrastructure from LLMTermExtractor
from src.termExtractor.LLMTermExtractor import (
    LLM_LOCATION_PLACEHOLDER,
)


########################
# EntityRecognitionData #
########################

class EntityRecognitionData(BaseTermDataset):
    """
    Wraps a list of pre-tokenized NER BIO samples as a PyTorch Dataset.

    Each sample in 'data' must be a dictionary with the keys:
        - 'input_ids'      (torch.LongTensor)
        - 'attention_mask' (torch.LongTensor)
        - 'labels'         (torch.LongTensor, token-level NER BIO tag IDs)

    These are produced by NERBIOTokenizer.process_files().
    Uses the default BaseTermDataset.collate_fn since all three values are pre-shaped tensors of identical length.
    """


###########################
# EntityRecognitionTrainer #
###########################

class EntityRecognitionTrainer(BaseTrainer):
    """
    Handles both the training loop and the inference pipeline for the transformer-based NER model.

    Training:
        - Fine-tunes an AutoModelForTokenClassification with entity-type-specific BIO labels using AdamW.
        - Optionally evaluates on a development set after every epoch and saves the best checkpoint based on strict F1 score (span + label match).
        - Supports automatic mixed precision (AMP) on CUDA.

    Inference:
        - Runs an entity-type-aware BIO NER pipeline over title / abstract text.
        - Merges subword predictions into full entity spans, preserving the predicted semantic label.
        - Corrects casing by copying spans directly from the source text.

    The key difference relative to TermExtractionTrainer is that each predicted entity carries its actual semantic category in the 'label' field rather than the generic "term" placeholder.
    """

    def __init__(
        self,
        model,
        tokenizer,
        device: torch.device,
        max_length: int = 512,
        merge_subword_predictions_to_words: bool = False,
    ):
        """
        Initializes the trainer with a model, tokenizer, target device, and sequence length cap.

        :param model: An AutoModelForTokenClassification instance configured with NER_BIO_LABELS.
        :param tokenizer: A fast HuggingFace tokenizer instance.
        :param device: The torch.device to run computations on.
        :param max_length: Maximum token sequence length (default 512).
        :param merge_subword_predictions_to_words: When True, collapse subword predictions to
            tokenizer word-level predictions before BIO entity merging.
        """
        super().__init__(model, tokenizer, device, max_length)

        # Build reverse label mapping from the model config
        self.id2label = model.config.id2label
        self.merge_subword_predictions_to_words = merge_subword_predictions_to_words

    # -- BaseTrainer abstract method implementations --

    def _forward_pass(self, batch: dict) -> torch.Tensor:
        """
        Runs a forward pass through AutoModelForTokenClassification and returns the token-level cross-entropy loss.

        :param batch: A dict with 'input_ids', 'attention_mask', and 'labels' tensors.
        :return: A scalar loss tensor.
        """
        outputs = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        return outputs.loss

    def _evaluate_on_dev(self, dev_data: dict, evaluator) -> dict:
        """
        Runs full inference over the raw development data and computes strict precision, recall, and F1 (span + label match) via the provided evaluator.

        :param dev_data: Raw development data dict (paper ID -> content).
        :param evaluator: A NERExtractionEvaluator instance.
        :return: A dict with 'score' (strict F1) and 'log_str' (P/R/F1 summary).
        """
        dev_predictions = self.perform_inference(dev_data)
        metrics = evaluator.evaluate(dev_predictions, dev_data)
        return {
            "score": metrics["strict_f1"],
            "log_str": (
                f"P: {metrics['strict_precision']:.4f} | "
                f"R: {metrics['strict_recall']:.4f} | "
                f"F1: {metrics['strict_f1']:.4f} "
                f"(span-only F1: {metrics['span_f1']:.4f}; "
                f"pred/gold: {metrics['num_pred_entities']}/{metrics['num_gold_entities']})"
            ),
        }

    def _save_checkpoint(self, output_dir: str) -> None:
        """
        Saves the model and tokenizer to 'output_dir'.

        :param output_dir: Directory path for the checkpoint.
        :return: None
        """
        os.makedirs(output_dir, exist_ok=True)
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)

    # -- Inference --

    def perform_inference(self, data: dict) -> dict:
        """
        Runs the NER inference pipeline over an entire dataset, returning a copy of the input data with 'entities' fields populated.  
        Each predicted entity carries the actual semantic label in its 'label' field.

        :param data: A dict mapping paper IDs to content dicts (GBIE format).
        :return: A dict with the same structure as 'data' but with predicted entities.
        """
        result = {}
        self.model.eval()

        for id, content in tqdm(data.items(), total=len(data), desc="Inference"):
            text = content.get('text')
            if not text:
                continue
            entity_predictions = []

            # Process title and abstract independently to preserve location labels
            token_preds = self._ner_pipeline(text)
            merged = self._merge_entities(token_preds, LLM_LOCATION_PLACEHOLDER)
            adjusted = self._adjust_casing(merged, text)
            entity_predictions.extend(adjusted)

            # Shallow copy of the content and overwrite entities field
            output_content = dict(content)
            output_content["entities"] = entity_predictions
            result[id] = output_content

        return result

    def _ner_pipeline(self, text: str) -> list:
        """
        Tokenizes a single text string, runs a forward pass through the model, and returns per-token predictions with confidence scores.

        :param text: Raw input text (title or abstract).
        :return: A list of dicts with keys 'entity', 'word', 'start', 'end', and 'confscore' -- only non-O predictions are included.
        """
        encoding = self.tokenizer(
            text,
            return_tensors="pt",
            return_offsets_mapping=True,
            truncation=True,
            max_length=self.max_length,
        )

        # Move inputs to device; keep offset mapping on CPU for span extraction
        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)

        # Strip [CLS] and [SEP] special tokens (index 0 and -1)
        offsets = encoding["offset_mapping"][0][1:-1]
        tokens = self.tokenizer.convert_ids_to_tokens(encoding["input_ids"][0])[1:-1]
        word_ids = self._get_word_ids(encoding, text, offsets)

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            # Softmax over label dimension; remove batch dimension and special tokens
            probabilities = torch.softmax(outputs.logits, dim=-1)[0][1:-1]
            predictions = torch.argmax(probabilities, dim=-1).tolist()

        token_predictions = [
            {
                "entity": self.id2label[prediction],
                "word": token,
                "start": int(start),
                "end": int(end),
                "confscore": float(probabilities[i][prediction].item()),
                "word_id": word_id,
            }
            for i, (prediction, token, (start, end), word_id) in enumerate(
                zip(predictions, tokens, offsets, word_ids)
            )
        ]

        if self.merge_subword_predictions_to_words:
            return self._collapse_subword_predictions_to_words(token_predictions, text)

        return [
            self._strip_word_id(token_prediction)
            for token_prediction in token_predictions
            if token_prediction["entity"] != "O"
        ]

    @staticmethod
    def _get_word_ids(encoding, text: str, offsets) -> list:
        """
        Returns one word identifier per non-special token, falling back to whitespace grouping
        if the tokenizer encoding does not expose word_ids().
        """
        try:
            return encoding.word_ids(batch_index=0)[1:-1]
        except (AttributeError, TypeError, ValueError):
            return EntityRecognitionTrainer._infer_word_ids_from_offsets(text, offsets)

    @staticmethod
    def _infer_word_ids_from_offsets(text: str, offsets) -> list:
        """
        Best-effort fallback for fast tokenizer word IDs: contiguous non-whitespace spans are
        treated as one word.
        """
        word_ids = []
        current_word_id = -1
        previous_end = None

        for start, end in offsets:
            start = int(start)
            end = int(end)
            if start == end:
                word_ids.append(None)
                continue

            starts_new_word = (
                previous_end is None
                or any(char.isspace() for char in text[previous_end:start])
            )
            if starts_new_word:
                current_word_id += 1

            word_ids.append(current_word_id)
            previous_end = end

        return word_ids

    @staticmethod
    def _strip_word_id(token_prediction: dict) -> dict:
        """
        Removes the tokenizer-alignment helper field from a prediction dict.
        """
        cleaned = dict(token_prediction)
        cleaned.pop("word_id", None)
        return cleaned

    def _collapse_subword_predictions_to_words(self, token_predictions: list, text: str) -> list:
        """
        Collapses token/subword predictions to one prediction per tokenizer word.

        The word receives the label predicted for its first token. If that first token is
        labelled O, the entire word is omitted from entity merging, even if later subword
        pieces received entity labels.
        """
        collapsed = []
        current_word = None

        def flush_current_word() -> None:
            if current_word is None or current_word["entity"] == "O":
                return
            current_word["word"] = text[current_word["start"] : current_word["end"]]
            collapsed.append(self._strip_word_id(current_word))

        for token_prediction in token_predictions:
            word_id = token_prediction.get("word_id")
            if word_id is None or token_prediction["start"] == token_prediction["end"]:
                continue

            if current_word is None or word_id != current_word["word_id"]:
                flush_current_word()
                current_word = dict(token_prediction)
                continue

            current_word["end"] = max(current_word["end"], token_prediction["end"])

        flush_current_word()
        return collapsed

    def _merge_entities(self, token_predictions: list, location: str) -> list:
        """
        Merges consecutive entity-type-specific BIO-tagged subword predictions into whole entity spans.  
        The entity label is extracted from the B-tag of each new span.  
        Confidence scores are averaged across constituent tokens.

        An I-{type} token that follows a B/I tag of a different type is treated as the start of a new entity (i.e. it is interpreted as an implicit B-{type}) to handle imperfect model outputs gracefully.

        :param token_predictions: List of per-token prediction dicts from _ner_pipeline.
        :param location: Section label -- 'title' or 'abstract'.
        :return: A list of merged entity dicts with keys 'start_idx', 'end_idx', 'location', 'text_span', 'label', and 'confscore'.
        """
        merged = []
        current_entity = None  # Tracks the entity being assembled

        for token_pred in token_predictions:
            # Tags have the form "B-Materiale", "I-Prodotto edilizio", etc.
            # Split on the first "-" only to correctly handle multi-word labels such as "anatomical location" that themselves contain no hyphens in the label part.
            prefix, entity_label = token_pred["entity"].split("-", 1)
            # Remove subword marker so pieces join cleanly
            word = token_pred["word"].replace("##", "")

            # Decide whether to start a new entity:
            #   - Explicit B-tag always starts a new span.
            #   - I-tag with a different label than the current open span also starts a new span.
            start_new = (
                prefix == "B"
                or current_entity is None
                or entity_label != current_entity["label"]
            )

            if start_new:
                # Finalize the previous entity before opening a new one
                if current_entity:
                    current_entity["confscore"] = round(
                        sum(current_entity["_scores"]) / len(current_entity["_scores"]), 6
                    )
                    del current_entity["_scores"]
                    merged.append(current_entity)

                # Open a fresh entity with the label from this tag
                current_entity = {
                    "start_idx": token_pred["start"],
                    "end_idx": token_pred["end"] - 1,
                    "location": location,
                    "text_span": word,
                    "label": entity_label,
                    "_scores": [token_pred["confscore"]],  # temp list, removed later
                }
            else:
                # I-tag with matching type: extend the current entity
                if token_pred["start"] == current_entity["end_idx"] + 1:
                    # Adjacent token -- no space needed
                    current_entity["text_span"] += word
                else:
                    # Gap between tokens -- insert a space
                    current_entity["text_span"] += " " + word
                current_entity["end_idx"] = token_pred["end"] - 1
                current_entity["_scores"].append(token_pred["confscore"])

        # Finalize the last open entity
        if current_entity:
            current_entity["confscore"] = round(
                sum(current_entity["_scores"]) / len(current_entity["_scores"]), 6
            )
            del current_entity["_scores"]
            merged.append(current_entity)

        return merged

    def _adjust_casing(self, entity_predictions: list, source_text: str) -> list:
        """
        Replaces each entity's 'text_span' with the exact substring from the original source text to restore correct casing lost during tokenization.

        :param entity_predictions: List of merged entity dicts.
        :param source_text: The original untokenized text.
        :return: The same list with corrected 'text_span' values (mutated in-place).
        """
        for entity in entity_predictions:
            entity["text_span"] = source_text[entity["start_idx"] : entity["end_idx"] + 1]
        return entity_predictions


########################
# NERExtractionEvaluator #
########################

class NERExtractionEvaluator:
    """
    Evaluates NER quality by comparing predicted entities against ground-truth annotations at two levels of strictness:
        - Strict (span + label): an entity is a true positive only when both the character offsets AND the semantic label match the gold annotation.
        - Span-only (lenient): an entity is a true positive when only the character offsets match, regardless of the predicted label. This mirrors TermExtractionEvaluator and allows comparison with the extraction-only pipeline.

    Precision, recall, and F1 are computed with binary averaging for both modes and returned in a single metrics dict.
    """

    def evaluate(self, predictions: dict, ground_truth: dict) -> dict:
        """
        Computes strict (span + label) and lenient (span-only) P/R/F1 across all papers.

        :param predictions:  Dict mapping paper IDs to content with predicted entities.
        :param ground_truth: Dict mapping paper IDs to content with gold entities.
        :return: A dict with keys 'strict_precision', 'strict_recall', 'strict_f1', 'span_precision', 'span_recall', 'span_f1' (all floats 0–1).
        """
        strict_gt_sets = self._build_entity_sets(ground_truth, include_label=True)
        strict_pred_sets = self._build_entity_sets(predictions,  include_label=True)
        span_gt_sets = self._build_entity_sets(ground_truth, include_label=False)
        span_pred_sets = self._build_entity_sets(predictions,  include_label=False)

        strict_p, strict_r, strict_f1 = self._compute_prf(strict_gt_sets, strict_pred_sets)
        span_p, span_r, span_f1 = self._compute_prf(span_gt_sets,   span_pred_sets)
        num_gold_entities = sum(len(entity_set) for entity_set in span_gt_sets.values())
        num_pred_entities = sum(len(entity_set) for entity_set in span_pred_sets.values())

        return {
            "strict_precision": strict_p,
            "strict_recall": strict_r,
            "strict_f1": strict_f1,
            "span_precision": span_p,
            "span_recall": span_r,
            "span_f1": span_f1,
            "num_gold_entities": num_gold_entities,
            "num_pred_entities": num_pred_entities,
        }

    @staticmethod
    def _compute_prf(gt_sets: dict, pred_sets: dict) -> tuple[float, float, float]:
        """
        Computes binary precision, recall, and F1 over the union of all paper IDs.

        :param gt_sets: Dict mapping paper IDs to sets of comparable entity tuples.
        :param pred_sets: Dict mapping paper IDs to sets of comparable entity tuples.
        :return: (precision, recall, f1) as floats in [0, 1].
        """
        y_true = []
        y_pred = []

        all_ids = set(gt_sets.keys()) | set(pred_sets.keys())
        for paper_id in all_ids:
            gt_entities   = gt_sets.get(paper_id, set())
            pred_entities = pred_sets.get(paper_id, set())
            all_entities  = gt_entities | pred_entities

            for entity in all_entities:
                y_true.append(1 if entity in gt_entities   else 0)
                y_pred.append(1 if entity in pred_entities else 0)

        if not y_true:
            return 0.0, 0.0, 0.0

        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="binary", zero_division=0
        )
        return float(precision), float(recall), float(f1)

    @staticmethod
    def _build_entity_sets(data: dict, include_label: bool) -> dict:
        """
        Converts entity lists to sets of comparable tuples for fast lookup.

        :param data: Dict mapping paper IDs to content dicts with 'entities' lists.
        :param include_label: When True, each tuple is (start_idx, end_idx, label); when False, each tuple is (start_idx, end_idx).
        :return: Dict mapping paper IDs to sets of tuples.
        """
        entity_sets = {}
        for paper_id, content in data.items():
            if include_label:
                entity_sets[paper_id] = {
                    (
                        entity["start_idx"],
                        entity["end_idx"],
                        entity["label"],
                    )
                    for entity in content.get("entities", [])
                }
            else:
                entity_sets[paper_id] = {
                    (
                        entity["start_idx"],
                        entity["end_idx"],
                    )
                    for entity in content.get("entities", [])
                }
        return entity_sets


####################
# Metadata helpers #
####################

def save_metadata(output_dir: str, args: argparse.Namespace) -> None:
    """
    Persists label mapping and training arguments alongside the model checkpoint.

    :param output_dir: Directory where the metadata JSON files will be written.
    :param args: Parsed argument namespace from argparse.
    :return: None
    """
    os.makedirs(output_dir, exist_ok=True)

    # Save the full NER BIO label list so inference can reconstruct the id2label mapping
    save_json_data({"ner_bio_labels": NER_BIO_LABELS}, os.path.join(output_dir, "label_mapping.json"))

    # Save all training hyper-parameters for reproducibility
    save_json_data(vars(args), os.path.join(output_dir, "training_args.json"))


#############################
# Top-level entry functions #
#############################

def run_training(args: argparse.Namespace) -> None:
    """
    Orchestrates data loading, model initialization, training, and (optional) development evaluation for a full NER training run.

    :param args: Parsed CLI arguments.
    :return: None
    """
    seed_everything(args.seed)

    device = get_device()
    print_device_info(device)

    # Build label↔id mappings from the global NER BIO label list
    label2id = {label: idx for idx, label in enumerate(NER_BIO_LABELS)}
    id2label = {idx: label for label, idx in label2id.items()}

    # -- Load raw data --
    train_data = load_merge_json_data(args.train_data_paths)
    dev_data = load_json_data(args.dev_data_path) if args.dev_data_path else None

    # -- Initialise tokenizer and model --
    hf_tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    model = AutoModelForTokenClassification.from_pretrained(
        args.model_name,
        num_labels=len(NER_BIO_LABELS),
        id2label=id2label,
        label2id=label2id,
    )

    # -- Preprocess training data --
    ner_tokenizer = NERBIOTokenizer(
        data=train_data,
        tokenizer=hf_tokenizer,
        label2id=label2id,
        max_length=args.max_length,
    )
    train_samples = ner_tokenizer.process_files()
    train_dataset = EntityRecognitionData(train_samples)
    train_loader = train_dataset.build_dataloader(
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        device=device,
    )

    print(f"Training samples: {len(train_samples)}")

    # -- Build trainer and evaluator --
    trainer = EntityRecognitionTrainer(
        model=model,
        tokenizer=hf_tokenizer,
        device=device,
        max_length=args.max_length,
    )
    evaluator = NERExtractionEvaluator() if dev_data is not None else None

    # -- Train --
    trainer.train(
        train_loader=train_loader,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        output_dir=args.output_dir,
        dev_data=dev_data,
        use_amp=args.use_amp,
        evaluator=evaluator,
    )

    # Persist label mapping and hyper-parameters alongside the checkpoint
    save_metadata(args.output_dir, args)

    maybe_empty_cache(device)


def run_inference(args: argparse.Namespace) -> None:
    """
    Loads a saved checkpoint and runs NER inference over a JSON dataset, writing results to
    the specified output path.

    :param args: Parsed CLI arguments.
    :return: None
    """
    device = get_device()
    print_device_info(device)

    # Load hyper-parameters saved during training (e.g. max_length)
    training_args = load_json_data(os.path.join(args.model_path, "training_args.json"))
    # CLI override takes priority over the saved value
    max_length = args.max_length if args.max_length else training_args.get("max_length", 512)

    # -- Load model and tokenizer from checkpoint --
    hf_tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
    model = AutoModelForTokenClassification.from_pretrained(args.model_path)
    model.to(device)
    model.eval()

    # -- Run inference --
    data = load_json_data(args.inference_data_path)
    trainer = EntityRecognitionTrainer(
        model=model,
        tokenizer=hf_tokenizer,
        device=device,
        max_length=max_length,
        merge_subword_predictions_to_words=getattr(args, "merge_subword_predictions_to_words", False),
    )
    results = trainer.perform_inference(data)

    output_filename = generate_output_filename(args)
    output_path = os.path.join(args.inference_output_path, output_filename)
    save_json_data(results, output_path)
    print(f"Inference results saved to {output_path}")

    maybe_empty_cache(device)

def generate_output_filename(args: argparse.Namespace) -> str:
    """
    Generate a standardized output filename based on configuration parameters.

    :param args: Parsed CLI arguments.
    :return: Output filename string.
    """
    model_name = args.model_name.split("/")[-1]  # Use the last part of the model name for brevity
    batch_size = str(args.batch_size)
    num_epochs = str(args.num_epochs)
    learning_rate = str(args.learning_rate).replace(".", "p")  # Replace dot for filename compatibility
    return f"entities_{model_name}_batch={batch_size}_epochs={num_epochs}_lr={learning_rate}.json"

#######
# CLI #
#######

def parse_args() -> argparse.Namespace:
    """
    Defines and parses command-line arguments for both training and inference modes.

    :return: An argparse.Namespace object with all parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Train or run inference for a transformer-based construction-sector NER model (joint extraction + classification)"
    )

    # -- Config file (optional) --
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to a YAML config file.  When provided, CLI arguments are ignored in favor of config values.",
    )

    # -- Mode --
    parser.add_argument(
        "--inference_only",
        action="store_true",
        help="Run inference only instead of training",
    )
    parser.add_argument(
        "--merge_subword_predictions_to_words",
        action="store_true",
        help=(
            "Inference only: collapse subword token predictions to word-level predictions "
            "before BIO entity merging, using the first token label for each word"
        ),
    )

    # -- Data paths --
    parser.add_argument(
        "--train_data_paths",
        type=str,
        nargs="+",
        default=None,
        help="One or more paths to training JSON files (merged automatically)",
    )
    parser.add_argument(
        "--dev_data_path",
        type=str,
        default=None,
        help="Path to development set JSON file",
    )
    parser.add_argument(
        "--inference_data_path",
        type=str,
        default=None,
        help="Path to JSON file used during inference",
    )
    parser.add_argument(
        "--inference_output_path",
        type=str,
        default=None,
        help="Path where inference results JSON will be written",
    )

    # -- Model paths --
    parser.add_argument(
        "--model_name",
        type=str,
        default="dbmdz/bert-base-italian-xxl-cased",
        help="HuggingFace model name or local path used to initialise the model",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory where the trained model checkpoint will be saved",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to a saved model checkpoint for inference",
    )

    # -- Training hyper-parameters --
    parser.add_argument("--batch_size", type=int, default=8, help="Training batch size")
    parser.add_argument("--num_epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="AdamW learning rate")
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="Maximum token sequence length (for both training and inference)",
    )
    parser.add_argument(
        "--separate_title_abstract",
        action="store_true",
        help="Process title and abstract as separate sequences (default: concatenated)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Global random seed")
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="DataLoader worker processes (use 0 or 2 on macOS)",
    )
    parser.add_argument(
        "--use_amp",
        action="store_true",
        help="Enable CUDA automatic mixed precision training",
    )

    args = parser.parse_args()

    if args.config is not None:
        # Load config from YAML file and override CLI args
        import yaml

        with open(args.config, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        for key, value in config.items():
            setattr(args, key, value)

    # -- Validation --
    if args.inference_only:
        if not args.model_path:
            parser.error("--model_path is required when --inference_only is used")
        if not args.inference_data_path:
            parser.error("--inference_data_path is required when --inference_only is used")
        if not args.inference_output_path:
            parser.error("--inference_output_path is required when --inference_only is used")
    else:
        if args.merge_subword_predictions_to_words:
            parser.error("--merge_subword_predictions_to_words is only valid with --inference_only")
        if not args.train_data_paths:
            parser.error("--train_data_paths is required for training")
        if not args.output_dir:
            parser.error("--output_dir is required for training")

    return args


###############
# Entry point #
###############

if __name__ == "__main__":
    args = parse_args()
    if args.inference_only:
        run_inference(args)
    else:
        run_training(args)
