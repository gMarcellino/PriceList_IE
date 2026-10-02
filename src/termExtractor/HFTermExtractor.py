#!/usr/bin/env python3
"""
HFTermExtractor.py

Transformer-based term extraction module using BIO tagging.
Organized into six classes:
    - BaseTermDataset : shared Dataset base for all term-task datasets
    - TermExtractionData : Dataset wrapping for term extraction
    - BaseTrainer : shared training loop infrastructure for all term-task trainers
    - TermExtractionTrainer : training loop and inference pipeline for term extraction
    - BIOTokenizer : BIO-tag preprocessing (imported from Tokenizers module)
    - TermExtractionEvaluator : precision / recall / F1 evaluation (imported from TermExtractionEvaluator module)

BaseTermDataset and BaseTrainer are designed to be reused by other task modules (e.g. HFTermClassifier) to keep training and data-loading consistent across tasks.

Entry point: run with --help to see CLI options.
"""

import argparse
import os
from abc import ABC, abstractmethod
from contextlib import nullcontext

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForTokenClassification, AutoTokenizer

# Allow unsupported MPS ops to fall back to CPU instead of crashing
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
# Disable HuggingFace tokenizer parallelism to avoid fork-related warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Import utilities implemented in utils.py
from src.utils.utils import (
    get_device,
    load_json_data,
    load_merge_json_data,
    maybe_empty_cache,
    move_to_device,
    print_device_info,
    save_json_data,
    seed_everything,
)

# Import the BIO tokenizer
from src.termExtractor.Tokenizers import BIOTokenizer  # noqa: F401  (re-exported for callers)

# Import the evaluator
from src.termExtractor.TermExtractionEvaluator import TermExtractionEvaluator  # noqa: F401  (re-exported for callers)


#############
# Constants #
#############

# BIO label set used throughout the module
BIO_LABELS = ["O", "B-term", "I-term"]


###################
# BaseTermDataset #
###################

class BaseTermDataset(Dataset, ABC):
    """
    Abstract base class for all term-task PyTorch Datasets.

    Provides the shared __len__, __getitem__, and build_dataloader implementations. 
    Subclasses must override collate_fn to handle their task-specific label representation (e.g. stacked tensors for token-level tasks vs. scalar integers for sequence-level tasks).

    Subclasses:
        - TermExtractionData (src.termExtractor.HFTermExtractor)
        - TermClassificationData (src.termClassifier.HFTermClassifier)
    """

    def __init__(self, data: list):
        """
        Initializes the dataset with a list of pre-processed samples.

        :param data: A list of sample dictionaries produced by a BaseTokenizer subclass. Each sample must contain at minimum 'input_ids', 'attention_mask', and 'labels'.
        """
        self.data = data

    def __len__(self) -> int:
        """
        Returns the total number of samples in the dataset.

        :return: Integer length of the dataset.
        """
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        """
        Retrieves a single sample by index.

        :param idx: Index of the sample to retrieve.
        :return: A sample dictionary with 'input_ids', 'attention_mask', and 'labels'.
        """
        return self.data[idx]

    @staticmethod
    def collate_fn(batch: list) -> dict:
        """
        Default collate function -- stacks all three keys as long tensors.
        Subclasses should override this if labels require a different representation (e.g. scalar integers rather than pre-shaped tensors).

        :param batch: A list of sample dictionaries.
        :return: A dictionary with stacked 'input_ids', 'attention_mask', and 'labels'.
        """
        return {
            key: torch.stack(
                [torch.as_tensor(item[key], dtype=torch.long) for item in batch]
            )
            for key in ("input_ids", "attention_mask", "labels")
        }

    def build_dataloader(self, batch_size: int, shuffle: bool, num_workers: int, device: torch.device,) -> DataLoader:
        """
        Builds and returns a DataLoader for the current dataset, using the concrete subclass's collate_fn.

        :param batch_size: Number of samples per batch.
        :param shuffle: Whether to shuffle the data at every epoch.
        :param num_workers: Number of subprocesses for data loading.
        :param device: Target device; used to enable pin_memory for CUDA.
        :return: A configured torch.utils.data.DataLoader instance.
        """
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            # pin_memory accelerates CPU->GPU transfers on CUDA devices
            pin_memory=(device.type == "cuda"),
            # Resolve collate_fn via the concrete subclass to pick up any override
            collate_fn=self.__class__.collate_fn,
            # Keep workers alive between batches when num_workers > 0
            persistent_workers=(num_workers > 0),
        )


######################
# TermExtractionData #
######################

class TermExtractionData(BaseTermDataset):
    """
    Wraps a list of pre-tokenized BIO samples as a PyTorch Dataset.

    Each sample in 'data' must be a dictionary with the keys:
        - 'input_ids'      (torch.LongTensor)
        - 'attention_mask' (torch.LongTensor)
        - 'labels'         (torch.LongTensor, token-level BIO tag IDs)

    These are produced by BIOTokenizer.process_files().
    Uses the default BaseTermDataset.collate_fn since all three values are pre-shaped tensors of identical length.
    """


###############
# BaseTrainer #
###############

class BaseTrainer(ABC):
    """
    Abstract base class providing shared training loop infrastructure for all term-task trainers.

    Handles:
        - AdamW optimiser setup
        - Automatic mixed precision (AMP) on CUDA
        - The epoch and batch loop (forward pass, backward pass, gradient step)
        - MPS memory logging
        - Best-checkpoint saving based on a scalar score returned by subclasses

    Subclasses must implement:
        - _forward_pass(batch) -> torch.Tensor (scalar loss)
        - _evaluate_on_dev(dev_data, evaluator) -> dict with 'score' and 'log_str'
        - _save_checkpoint(output_dir) -> None

    Subclasses:
        - TermExtractionTrainer (src.termExtractor.HFTermExtractor)
        - TermClassificationTrainer (src.termClassifier.HFTermClassifier)
    """

    def __init__(self, model, tokenizer, device: torch.device, max_length: int = 512,):
        """
        Initializes the trainer with a model, tokenizer, target device, and sequence length cap.

        :param model: A HuggingFace model or custom nn.Module instance.
        :param tokenizer: A HuggingFace tokenizer instance.
        :param device: The torch.device to run computations on.
        :param max_length: Maximum token sequence length (default 512).
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_length = max_length

    # ------------------------------------------------------------------ #
    # Training                                                             #
    # ------------------------------------------------------------------ #

    def train(self, train_loader: DataLoader, num_epochs: int, learning_rate: float, output_dir: str, dev_data=None, use_amp: bool = False, evaluator=None,) -> None:
        """
        Runs the full training loop, optionally evaluating on development data after each epoch and saving the best-score checkpoint.

        The type of 'dev_data' is intentionally left open: concrete subclasses control what is passed (raw data dict, DataLoader, etc.) and consume it in their _evaluate_on_dev implementation.

        :param train_loader: DataLoader for the training set.
        :param num_epochs: Total number of training epochs.
        :param learning_rate: Learning rate for the AdamW optimiser.
        :param output_dir: Directory where model checkpoints are saved.
        :param dev_data: Optional development data passed to _evaluate_on_dev.
        :param use_amp: Enable CUDA automatic mixed precision (ignored on other devices).
        :param evaluator: Optional evaluator instance passed to _evaluate_on_dev.
        :return: None
        """
        self.model.to(self.device)

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)

        # AMP is only supported on CUDA
        amp_enabled = use_amp and self.device.type == "cuda"
        scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
        amp_context = torch.cuda.amp.autocast if amp_enabled else nullcontext

        best_score = -1.0

        for epoch in tqdm(range(num_epochs), desc="Training", unit="epoch"):
            # -- Training step --
            self.model.train()
            total_loss = 0.0

            for batch in tqdm(
                train_loader,
                desc=f"Epoch {epoch + 1}/{num_epochs}",
                leave=False,
            ):
                # Move the whole batch to the target device
                batch = move_to_device(batch, self.device)
                optimizer.zero_grad(set_to_none=True)

                with amp_context():
                    loss = self._forward_pass(batch)

                # Backward pass -- scale when AMP is active
                if amp_enabled:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

                total_loss += loss.detach().float().item()

            avg_loss = total_loss / max(len(train_loader), 1)
            print(f"Epoch {epoch + 1}/{num_epochs} | Avg. train loss: {avg_loss:.4f}")

            self._log_mps_memory()

            # -- Development evaluation --
            if dev_data is not None and evaluator is not None:
                self.model.eval()
                metrics = self._evaluate_on_dev(dev_data, evaluator)
                print(f"  Dev  {metrics['log_str']}")

                # Save checkpoint if this epoch's score is the best so far
                if metrics["score"] > best_score:
                    best_score = metrics["score"]
                    self._save_checkpoint(output_dir)
                    print(f"\tNew best model saved (score = {best_score:.4f})")

        # If no dev set was used, save the final model
        if dev_data is None:
            self._save_checkpoint(output_dir)
            print("Model saved (no dev set -- saved final epoch).")

    @abstractmethod
    def _forward_pass(self, batch: dict) -> torch.Tensor:
        """
        Runs a single forward pass and returns the scalar loss tensor.
        The batch is already on the correct device when this is called.

        :param batch: A dict with 'input_ids', 'attention_mask', and 'labels' tensors.
        :return: A scalar loss tensor (requires_grad=True).
        """

    @abstractmethod
    def _evaluate_on_dev(self, dev_data, evaluator) -> dict:
        """
        Evaluates the model on the development set and returns a result dict.
        The model is already in eval mode when this is called.

        The concrete type of 'dev_data' is determined by the subclass (e.g. a raw dict for extraction, a DataLoader for classification).

        :param dev_data:  Development data in the subclass-specific format.
        :param evaluator: An evaluator instance appropriate for the task.
        :return: A dictionary with at minimum:
                - 'score' (float): scalar used to select the best checkpoint.
                - 'log_str' (str): human-readable metrics summary for printing.
        """

    @abstractmethod
    def _save_checkpoint(self, output_dir: str) -> None:
        """
        Persists the model (and tokenizer) to 'output_dir'.

        :param output_dir: Directory path for the checkpoint.
        :return: None
        """

    def _log_mps_memory(self) -> None:
        """
        Prints MPS memory statistics when running on Apple Silicon.
        No-op on other device types.
        """
        if self.device.type == "mps":
            try:
                print(f"  MPS allocated: {torch.mps.current_allocated_memory()}")
                print(f"  MPS driver:    {torch.mps.driver_allocated_memory()}")
            except Exception:
                pass


#########################
# TermExtractionTrainer #
#########################

class TermExtractionTrainer(BaseTrainer):
    """
    Handles both the training loop and the inference pipeline for the transformer-based term extraction model.

    Training:
        - Fine-tunes an AutoModelForTokenClassification with AdamW.
        - Optionally evaluates on a development set after every epoch and saves the best checkpoint based on F1 score.
        - Supports automatic mixed precision (AMP) on CUDA.

    Inference:
        - Runs a BIO NER pipeline over title/abstract text.
        - Merges subword predictions into full entity spans.
        - Corrects casing by copying spans directly from the source text.
    """

    def __init__(self, model, tokenizer, device: torch.device, max_length: int = 512,):
        """
        Initializes the trainer with a model, tokenizer, target device, and sequence length cap.

        :param model: An AutoModelForTokenClassification instance.
        :param tokenizer: A fast HuggingFace tokenizer instance.
        :param device: The torch.device to run computations on.
        :param max_length: Maximum token sequence length (default 512).
        """
        super().__init__(model, tokenizer, device, max_length)

        # Build reverse label mapping from the model config
        self.id2label = model.config.id2label

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
        Runs full inference over the raw development data and computes precision, recall, and F1 via the provided evaluator.

        :param dev_data: Raw development data dict (paper ID -> content).
        :param evaluator: A TermExtractionEvaluator instance.
        :return: A dict with 'score' (F1) and 'log_str' (P/R/F1 summary).
        """
        dev_predictions = self.perform_inference(dev_data)
        metrics = evaluator.evaluate(dev_predictions, dev_data)
        return {
            "score": metrics["f1"],
            "log_str": (
                f"P: {metrics['precision']:.4f} | "
                f"R: {metrics['recall']:.4f} | "
                f"F1: {metrics['f1']:.4f}"
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
        Runs the term extraction inference pipeline over an entire dataset, returning a copy of the input data with 'entities' fields populated.

        :param data: A dict mapping paper IDs to content dicts (GBIE format).
        :return: A dict with the same structure as 'data' but with predicted entities.
        """
        result = {}
        self.model.eval()

        for paper_id, content in tqdm(data.items(), total=len(data), desc="Inference"):
            title, abstract = get_title_and_abstract(content)
            entity_predictions = []

            # Process title and abstract independently to preserve location labels
            for section, text in [("title", title), ("abstract", abstract)]:
                token_preds = self._ner_pipeline(text)
                merged = self._merge_entities(token_preds, section)
                adjusted = self._adjust_casing(merged, text)
                entity_predictions.extend(adjusted)

            # Shallow copy of the content and overwrite entities field
            output_content = dict(content)
            output_content["entities"] = entity_predictions
            result[paper_id] = output_content

        return result

    def _ner_pipeline(self, text: str) -> list:
        """
        Tokenizes a single text string, runs a forward pass through the model, and returns per-token predictions with confidence scores.

        :param text: Raw input text (title or abstract).
        :return: A list of dicts with keys 'entity', 'word', 'start', 'end', and 'confscore' -- only non-O tokens are included.
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

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            # Softmax over label dimension; remove batch dimension and special tokens
            probabilities = torch.softmax(outputs.logits, dim=-1)[0][1:-1]
            predictions = torch.argmax(probabilities, dim=-1).tolist()

        return [
            {
                "entity": self.id2label[prediction],
                "word": token,
                "start": int(start),
                "end": int(end),
                "confscore": float(probabilities[i][prediction].item()),
            }
            for i, (prediction, token, (start, end)) in enumerate(
                zip(predictions, tokens, offsets)
            )
            if self.id2label[prediction] != "O"  # skip non-entity tokens
        ]

    def _merge_entities(self, token_predictions: list, location: str) -> list:
        """
        Merges consecutive BIO-tagged subword predictions into whole entity spans.
        Confidence scores are averaged across constituent tokens.

        :param token_predictions: List of per-token prediction dicts from _ner_pipeline.
        :param location: Section label -- 'title' or 'abstract'.
        :return: A list of merged entity dicts with keys 'start_idx', 'end_idx', 'location', 'text_span', 'label', and 'confscore'.
        """
        merged = []
        current_entity = None  # Tracks the entity being assembled

        for token_pred in token_predictions:
            # Split "B-term" -> prefix="B", type="term"
            prefix, _ = token_pred["entity"].split("-", 1)
            # Remove subword marker so pieces join cleanly
            word = token_pred["word"].replace("##", "")

            if prefix == "B" or current_entity is None:
                # Finalize the previous entity before starting a new one
                if current_entity:
                    current_entity["confscore"] = round(
                        sum(current_entity["_scores"]) / len(current_entity["_scores"]), 6
                    )
                    del current_entity["_scores"]
                    merged.append(current_entity)

                # Start a fresh entity
                current_entity = {
                    "start_idx": token_pred["start"],
                    "end_idx": token_pred["end"] - 1,
                    "location": location,
                    "text_span": word,
                    "label": "term",
                    "_scores": [token_pred["confscore"]],  # temp list, removed later
                }
            else:
                # I-tag: extend the current entity
                if token_pred["start"] == current_entity["end_idx"] + 1:
                    # Adjacent token -- no space needed
                    current_entity["text_span"] += word
                else:
                    # Gap between tokens -- insert a space
                    current_entity["text_span"] += " " + word
                current_entity["end_idx"] = token_pred["end"] - 1
                current_entity["_scores"].append(token_pred["confscore"])

        # Don't forget the last open entity
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

    # Save BIO label list so inference can reconstruct the id2label mapping
    save_json_data({"bio_labels": BIO_LABELS}, os.path.join(output_dir, "label_mapping.json"))

    # Save all training hyper-parameters for reproducibility
    save_json_data(vars(args), os.path.join(output_dir, "training_args.json"))


#############################
# Top-level entry functions #
#############################

def run_training(args: argparse.Namespace) -> None:
    """
    Orchestrates data loading, model initialization, training, and (optional) development evaluation for a full training run.

    :param args: Parsed CLI arguments.
    :return: None
    """
    seed_everything(args.seed)

    device = get_device()
    print_device_info(device)

    # Build label↔id mappings from the global BIO label list
    label2id = {label: idx for idx, label in enumerate(BIO_LABELS)}
    id2label = {idx: label for label, idx in label2id.items()}

    # -- Load raw data --
    train_data = load_merge_json_data(args.train_data_paths)
    dev_data = load_json_data(args.dev_data_path) if args.dev_data_path else None

    # -- Initialise tokenizer and model --
    hf_tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    model = AutoModelForTokenClassification.from_pretrained(
        args.model_name,
        num_labels=len(BIO_LABELS),
        id2label=id2label,
        label2id=label2id,
    )

    # -- Preprocess training data --
    bio_tokenizer = BIOTokenizer(
        data=train_data,
        tokenizer=hf_tokenizer,
        label2id=label2id,
        max_length=args.max_length,
    )
    train_samples = bio_tokenizer.process_files()
    train_dataset = TermExtractionData(train_samples)
    train_loader = train_dataset.build_dataloader(
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        device=device,
    )

    print(f"Training samples: {len(train_samples)}")

    # -- Build trainer and evaluator --
    trainer = TermExtractionTrainer(
        model=model,
        tokenizer=hf_tokenizer,
        device=device,
        max_length=args.max_length,
    )
    evaluator = TermExtractionEvaluator() if dev_data is not None else None

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
    Loads a saved checkpoint and runs inference over a JSON dataset, writing results to the specified output path.

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
    trainer = TermExtractionTrainer(
        model=model,
        tokenizer=hf_tokenizer,
        device=device,
        max_length=max_length,
    )
    results = trainer.perform_inference(data)

    save_json_data(results, args.inference_output_path)
    print(f"Inference results saved to {args.inference_output_path}")

    maybe_empty_cache(device)


#######
# CLI #
#######

def parse_args() -> argparse.Namespace:
    """
    Defines and parses command-line arguments for both training and inference modes.

    :return: An argparse.Namespace object with all parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Train or run inference for a transformer-based biomedical term extractor"
    )

    # -- Mode --
    parser.add_argument(
        "--inference_only",
        action="store_true",
        help="Run inference only instead of training",
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
        default="michiyasunaga/BioLinkBERT-base",
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

    # -- Validation --
    if args.inference_only:
        if not args.model_path:
            parser.error("--model_path is required when --inference_only is used")
        if not args.inference_data_path:
            parser.error("--inference_data_path is required when --inference_only is used")
        if not args.inference_output_path:
            parser.error("--inference_output_path is required when --inference_only is used")
    else:
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