#!/usr/bin/env python3
"""
LLMTermExtractor.py

LLM-based term extraction module.
Supports both online providers (OpenAI, Azure OpenAI, Groq, and any other OpenAI-compatible endpoint) and local providers (LM Studio, Ollama, and HuggingFace Transformers running on the same machine).

Organized into six sections:
    - Provider classes : thin wrappers that expose a uniform chat_completion() interface over every supported backend
    - ResponseParser : converts raw LLM text output into structured entity lists
    - LLMTermExtractor : zero-shot term extraction inference orchestrator
    - TermExtractionEvaluator : precision / recall / F1 evaluation (API-compatible with HFTermExtractor.TermExtractionEvaluator)
    - Helper functions : shared utilities (load_prompts, build_provider, etc.)
    - CLI : argument parsing and top-level entry point

Reusable exports (imported by other modules like LLMTermClassifier):
    - BaseLLMProvider, OpenAICompatibleProvider, AzureOpenAIProvider, HuggingFaceLocalProvider
    - PROVIDER_REGISTRY, build_provider()
    - load_prompts()
    - DEFAULT_TEMPERATURE, DEFAULT_MAX_NEW_TOKENS

Adding a new provider requires only:
    1. Subclassing BaseLLMProvider and implementing chat_completion().
    2. Registering the subclass in PROVIDER_REGISTRY at the bottom of the "Provider classes" section.

Entry point: run with --help to see CLI options.
"""

import argparse
import json
import os
import re

import torch
from tqdm import tqdm

# Allow unsupported MPS ops to fall back to CPU instead of crashing
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
# Disable HuggingFace tokenizer parallelism to avoid fork-related warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Import utilities implemented in utils.py
from src.utils.utils import (
    clean_llm_decoding_artifacts,
    get_device,
    load_json_data,
    maybe_empty_cache,
    print_device_info,
    save_json_data,
)

# Import the evaluator 
from src.termExtractor.TermExtractionEvaluator import TermExtractionEvaluator  # noqa: F401  (re-exported for callers)

#############
# Constants #
#############

# Default generation parameters used across all providers
DEFAULT_TEMPERATURE = 1.0
DEFAULT_MAX_NEW_TOKENS = 512

# Confidence score placeholder: LLMs do not expose per-span probabilities,
# so we assign a uniform sentinel value that downstream code can detect.
LLM_CONFSCORE_PLACEHOLDER = -1.0
LLM_LOCATION_PLACEHOLDER = "location_placeholder"
HF_LOCATION_PLACEHOLDER = "DEA_ANMNS_value"

#######################
# Reusable exports    #
#######################

__all__ = [
    # Provider classes and utilities
    "BaseLLMProvider",
    "OpenAICompatibleProvider",
    "AzureOpenAIProvider",
    "HuggingFaceLocalProvider",
    "PROVIDER_REGISTRY",
    "build_provider",
    # Constants
    "DEFAULT_TEMPERATURE",
    "DEFAULT_MAX_NEW_TOKENS",
    "LLM_CONFSCORE_PLACEHOLDER",
    # Helper functions
    "load_prompts",
    # Term extraction classes
    "ResponseParser",
    "LLMTermExtractor",
    # Evaluator
    "TermExtractionEvaluator",
]


####################
# Provider classes #
####################

class BaseLLMProvider:
    """
    Abstract base class for all LLM provider wrappers.

    Every concrete provider must implement chat_completion(), which accepts a system prompt, a user prompt, and a sampling temperature, and returns the model's response as a plain string.

    Subclasses are free to accept any provider-specific kwargs in __init__; only chat_completion() is part of the public contract.
    """

    def chat_completion(self, system_prompt: str, user_prompt: str, temperature: float = DEFAULT_TEMPERATURE,) -> str:
        """
        Runs a single chat-style completion and returns the model's response.

        :param system_prompt: The system-role message sent to the model.
        :param user_prompt: The user-role message sent to the model.
        :param temperature: Sampling temperature (0.0 = greedy, higher = more random).
        :return: The model's text response as a plain string.
        :raises NotImplementedError: If not overridden by a concrete subclass.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement chat_completion()"
        )


def _build_chat_messages(system_prompt: str | None, user_prompt: str) -> list[dict[str, str]]:
    """
    Builds chat messages, omitting the system role when no system prompt is provided.
    """
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})
    return messages


##################################################
# OpenAI-compatible providers (online and local) #
##################################################

class OpenAICompatibleProvider(BaseLLMProvider):
    """
    Provider wrapper for any server that exposes an OpenAI-compatible /v1/chat/completions endpoint.

    This single class covers a wide range of backends:

        Provider     | base_url                            | Notes
        -------------|-------------------------------------|-------------------------------
        OpenAI       | None (library default)              | Requires OPENAI_API_KEY
        Groq         | https://api.groq.com/openai/v1      | Requires GROQ_API_KEY
        Together AI  | https://api.together.xyz/v1         | Requires TOGETHER_API_KEY
        LM Studio    | http://localhost:1234/v1            | API key unused (any string)
        Ollama       | http://localhost:11434/v1           | API key unused (any string)

    Any endpoint that follows the OpenAI wire format can be plugged in by supplying the appropriate base_url and api_key.
    """

    def __init__(self, model: str, api_key: str | None = None, base_url: str | None = None, max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,):
        """
        Initializes the provider and lazily imports the 'openai' package.

        :param model: Model identifier as expected by the target server (e.g. "gpt-4o", "gemma-3-27b-it", "llama3").
        :param api_key: API key for authentication.  If None, the library falls back to the OPENAI_API_KEY environment variable.
        :param base_url: Override the default OpenAI endpoint.  Set to the local server URL when using LM Studio or Ollama.
        :param max_new_tokens: Maximum number of tokens the model may generate.
        """
        # Defer the import so that users who only use HuggingFaceLocalProvider are not forced to install the openai package.
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError(
                "The 'openai' package is required for OpenAICompatibleProvider.  "
                "Install it with: pip install openai"
            ) from exc

        self.model = model
        self.max_new_tokens = max_new_tokens

        # base_url=None keeps the default OpenAI endpoint; any local server URL overrides it.  
        # api_key="lm-studio" (or similar) satisfies servers that require a non-empty string but do not actually validate its value.
        self.client = OpenAI(base_url=base_url, api_key=api_key)

    def chat_completion(self, system_prompt: str, user_prompt: str, temperature: float = DEFAULT_TEMPERATURE,) -> str:
        """
        Calls the OpenAI-compatible chat completions endpoint.

        :param system_prompt: System-role message.
        :param user_prompt: User-role message.
        :param temperature: Sampling temperature.
        :return: The model's text response.
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=_build_chat_messages(system_prompt, user_prompt),
            temperature=temperature,
            max_tokens=self.max_new_tokens,
        )
        return response.choices[0].message.content


class AzureOpenAIProvider(BaseLLMProvider):
    """
    Provider wrapper for Azure OpenAI deployments.

    Azure uses a separate client class and requires a deployment name, an API version string, and an endpoint URL in addition to the API key.
    These are most conveniently supplied via environment variables:

        AZURE_OPENAI_ENDPOINT - e.g. https://my-resource.openai.azure.com/
        AZURE_OPENAI_API_KEY - the resource's secret key
        OPENAI_API_VERSION - e.g. "2024-02-01"

    or passed directly to __init__ for programmatic use.
    """

    def __init__(self, deployment: str, azure_endpoint: str | None = None, api_key: str | None = None, api_version: str = "2024-02-01", max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,):
        """
        Initializes the Azure OpenAI provider.

        :param deployment: Name of the Azure model deployment (e.g. "gpt-4o-deployment").
        :param azure_endpoint: Full Azure resource endpoint URL. Falls back to the AZURE_OPENAI_ENDPOINT environment variable.
        :param api_key: Azure resource API key. Falls back to the AZURE_OPENAI_API_KEY environment variable.
        :param api_version: Azure API version string. Falls back to the OPENAI_API_VERSION environment variable.
        :param max_new_tokens: Maximum tokens the model may generate.
        """
        try:
            from openai import AzureOpenAI
        except ImportError as exc:
            raise ImportError(
                "The 'openai' package is required for AzureOpenAIProvider.  "
                "Install it with: pip install openai"
            ) from exc

        self.deployment = deployment
        self.max_new_tokens = max_new_tokens

        # Resolve credentials: explicit argument > environment variable
        self.client = AzureOpenAI(
            azure_endpoint=azure_endpoint or os.environ["AZURE_OPENAI_ENDPOINT"],
            api_key=api_key or os.environ.get("AZURE_OPENAI_API_KEY"),
            api_version=api_version,
        )

    def chat_completion(self, system_prompt: str, user_prompt: str, temperature: float = DEFAULT_TEMPERATURE,) -> str:
        """
        Calls the Azure OpenAI chat completions endpoint.

        :param system_prompt: System-role message.
        :param user_prompt: User-role message.
        :param temperature: Sampling temperature.
        :return: The model's text response.
        """
        response = self.client.chat.completions.create(
            model=self.deployment,   # Azure uses deployment name here, not model name
            messages=_build_chat_messages(system_prompt, user_prompt),
            temperature=temperature,
            max_tokens=self.max_new_tokens,
        )
        return response.choices[0].message.content


##############################
# Local HuggingFace provider #
##############################

class HuggingFaceLocalProvider(BaseLLMProvider):
    """
    Provider wrapper for HuggingFace causal language models loaded locally via the Transformers library.

    This is the local-inference counterpart of the script in classify_terms.py.
    It supports CUDA, Apple Silicon MPS, and CPU execution.  
    For very large models (e.g. 27 B parameters) consider quantisation or switching to an OpenAI-compatible local server such as LM Studio or Ollama instead.
    """

    def __init__(self, model_id: str, device: str | None = None, max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS, top_p: float = 1.0,):
        """
        Loads the tokenizer and model from the HuggingFace Hub (or a local path).

        :param model_id: HuggingFace model identifier or path to a local directory (e.g. "google/gemma-3-27b-it").
        :param device: Target device string ("cuda", "mps", "cpu"). If None, the best available device is selected automatically via get_device().
        :param max_new_tokens: Maximum tokens the model may generate per call.
        :param top_p: Nucleus sampling parameter.
        """
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "The 'transformers' package is required for HuggingFaceLocalProvider.  "
                "Install it with: pip install transformers"
            ) from exc

        self.max_new_tokens = max_new_tokens
        self.top_p = top_p

        # Prefer explicitly requested device; fall back to get_device() utility
        self.device = device or get_device().type

        print(f"Loading '{model_id}' onto '{self.device}'...")

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            # float16 is the best trade-off on MPS; float32 elsewhere avoids
            # precision issues on CPU where float16 support is limited.
            torch_dtype=torch.float16 if self.device == "mps" else torch.float32,
            # Reduces RAM spike during weight loading for large models
            low_cpu_mem_usage=True,
        )
        self.model.to(self.device)
        self.model.eval()

    def chat_completion(self, system_prompt: str, user_prompt: str, temperature: float = DEFAULT_TEMPERATURE,) -> str:
        """
        Runs a chat-template generation pass and returns only the newly generated text.

        :param system_prompt: System-role message.
        :param user_prompt: User-role message.
        :param temperature: Sampling temperature. Temperature 0.0 disables sampling (greedy decoding) to avoid NaN issues on some backends.
        :return: The model's decoded text response.
        """
        messages = _build_chat_messages(system_prompt, user_prompt)

        # apply_chat_template formats the messages using the model's own template
        # (e.g. Gemma's <start_of_turn> markers) and returns ready-to-use tensors.
        inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        input_len = inputs["input_ids"].shape[-1]

        # Only use sampling when temperature > 0; greedy decoding is deterministic and avoids invalid temperature arguments on some model configurations.
        do_sample = temperature > 0.0
        bad_words_ids, suppress_tokens = _build_suppressed_token_args(self.tokenizer)

        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=do_sample,
                temperature=temperature if do_sample else None,
                top_p=self.top_p if do_sample else None,
                # Silence a common warning about missing pad_token_id
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                bad_words_ids=bad_words_ids,
                suppress_tokens=suppress_tokens,
            )

        # Slice off the prompt tokens so we return only the generated portion
        generated_ids = output_ids[0][input_len:]
        decoded = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        # Flush the MPS cache to reduce memory fragmentation on Apple Silicon
        if self.device == "mps":
            try:
                torch.mps.empty_cache()
            except Exception:
                pass

        return decoded


def _build_suppressed_token_args(tokenizer) -> tuple[list[list[int]] | None, list[int] | None]:
    """
    Prevent generation from starting with pad-only output for gemma-3-family models.
    """
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if pad_token_id is None or pad_token_id == eos_token_id:
        return None, None
    return [[pad_token_id]], [pad_token_id]


#####################
# Provider registry #
#####################

# Maps provider name strings (as used in the CLI) to their classes.
# To add a new provider, insert an entry here; no other code needs changing.
PROVIDER_REGISTRY: dict[str, type[BaseLLMProvider]] = {
    # Online / OpenAI-compatible endpoints
    "openai":      OpenAICompatibleProvider,   # api.openai.com
    "groq":        OpenAICompatibleProvider,   # api.groq.com/openai/v1
    "together":    OpenAICompatibleProvider,   # api.together.xyz/v1
    "azure":       AzureOpenAIProvider,
    # Local servers with OpenAI-compatible API
    "lmstudio":    OpenAICompatibleProvider,   # localhost:1234/v1
    "ollama":      OpenAICompatibleProvider,   # localhost:11434/v1
    # Local HuggingFace model (no server required)
    "huggingface": HuggingFaceLocalProvider,
}

# Default base URLs for well-known local servers
_LOCAL_DEFAULT_BASE_URLS: dict[str, str] = {
    "lmstudio": "http://localhost:1234/v1",
    "ollama":   "http://localhost:11434/v1",
}


def build_provider(provider_name: str, model: str, api_key: str | None = None, base_url: str | None = None, azure_endpoint: str | None = None, api_version: str = "2024-02-01", max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS, device: str | None = None,) -> BaseLLMProvider:
    """
    Factory function that instantiates the correct provider class from the PROVIDER_REGISTRY given a provider name string.

    :param provider_name: One of the keys in PROVIDER_REGISTRY (e.g. "openai", "lmstudio", "huggingface").
    :param model: Model or deployment identifier.
    :param api_key: API key (not required for local providers).
    :param base_url: Custom base URL for OpenAI-compatible endpoints. Defaults to well-known local URLs for "lmstudio" and "ollama" if not explicitly provided.
    :param azure_endpoint: Azure resource endpoint URL (AzureOpenAIProvider only).
    :param api_version: Azure API version string (AzureOpenAIProvider only).
    :param max_new_tokens: Maximum tokens to generate.
    :param device: Target device string (HuggingFaceLocalProvider only).
    :return: A configured BaseLLMProvider instance.
    :raises ValueError: If provider_name is not in PROVIDER_REGISTRY.
    """
    provider_name = provider_name.lower()

    if provider_name not in PROVIDER_REGISTRY:
        raise ValueError(
            f"Unknown provider '{provider_name}'.  "
            f"Available providers: {sorted(PROVIDER_REGISTRY.keys())}"
        )

    cls = PROVIDER_REGISTRY[provider_name]

    # -- Azure gets its own constructor signature --
    if provider_name == "azure":
        return cls(
            deployment=model,
            azure_endpoint=azure_endpoint,
            api_key=api_key,
            api_version=api_version,
            max_new_tokens=max_new_tokens,
        )

    # -- Local HuggingFace model: no networking parameters needed --
    if provider_name == "huggingface":
        return cls(
            model_id=model,
            device=device,
            max_new_tokens=max_new_tokens,
        )

    # -- OpenAI-compatible providers --
    # Fall back to a known default base URL for local servers if the caller did not supply one
    # None means "use the standard OpenAI endpoint".
    resolved_base_url = base_url or _LOCAL_DEFAULT_BASE_URLS.get(provider_name)
    # Dummy key so local servers that require a non-empty string don't reject
    resolved_key = api_key or (provider_name if resolved_base_url else None)

    return cls(
        model=model,
        api_key=resolved_key,
        base_url=resolved_base_url,
        max_new_tokens=max_new_tokens,
    )


##################
# ResponseParser #
##################

class ResponseParser:
    """
    Converts raw LLM text output into a list of entity dicts compatible with the format produced by HFTermExtractor.TermExtractionTrainer.perform_inference().

    The parser attempts to interpret the model's response as one of several common list formats (JSON array, bullet/dash list, newline-separated, comma-separated). 
    For each extracted term string it then performs a case-insensitive substring search over the source text to recover character offsets, mimicking the (start_idx, end_idx) fields produced by the BIO-based pipeline.

    Because a single term may appear more than once in the source text, all occurrences are returned as separate entity entries.
    """

    # Bullet / dash / numbered list item prefix patterns
    _BULLET_RE = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s*", re.MULTILINE)

    @staticmethod
    def parse(response: str, source_text: str, location: str) -> list[dict]:
        """
        Main entry point.  Parses 'response' into a list of entity dicts, anchoring each extracted term to its position(s) in 'source_text'.

        :param response: Raw string returned by the LLM.
        :param source_text: The original text that was presented to the model.
        :param location: Location label embedded in every entity. Current workflows use a placeholder value.
        :return: A list of entity dicts, each with keys: 'start_idx', 'end_idx', 'location', 'text_span', 'label', 'confscore'.
        """
        response = clean_llm_decoding_artifacts(response)
        term_strings = ResponseParser._extract_term_strings(response)
        entities = []

        for term in term_strings:
            if not term.strip():
                continue
            # Find all (possibly overlapping) occurrences of the term in the source
            spans = ResponseParser._find_spans(term, source_text)
            for start, end in spans:
                entities.append(
                    {
                        "start_idx": start,
                        # HFTermExtractor uses inclusive end; adjust accordingly
                        "end_idx": end - 1,
                        "location": location,
                        # Recover exact casing from source text
                        "text_span": source_text[start:end],
                        "label": "term",
                        # LLMs do not expose per-span probabilities; use sentinel
                        "confscore": LLM_CONFSCORE_PLACEHOLDER,
                    }
                )

        return entities

    @staticmethod
    def _extract_term_strings(response: str) -> list[str]:
        """
        Attempts to extract a flat list of term strings from the LLM response, trying each format in order of likelihood.

        Formats tried, in order:
            1. JSON array of strings  – ["term1", "term2", ...]
            2. Markdown bullet list   – - term1\n  * term2\n  3. term3
            3. Newline-separated list – term1\nterm2
            4. Comma-separated list   – term1, term2

        :param response: Raw LLM output string.
        :return: A (possibly empty) list of candidate term strings.
        """
        # 1. JSON array: look for the outermost [...] and try to parse it
        json_match = re.search(r"\[.*?\]", response, re.DOTALL)
        if json_match:
            try:
                parsed = json.loads(json_match.group())
                if isinstance(parsed, list):
                    return [str(t).strip() for t in parsed if t]
            except json.JSONDecodeError:
                pass

        # 2. Markdown bullet / numbered list
        lines = response.splitlines()
        bullet_lines = [
            ResponseParser._BULLET_RE.sub("", line).strip()
            for line in lines
            if ResponseParser._BULLET_RE.match(line)
        ]
        if bullet_lines:
            return bullet_lines

        # 3. Newline-separated (multi-line plain list)
        stripped_lines = [ln.strip() for ln in lines if ln.strip()]
        if len(stripped_lines) > 1:
            return stripped_lines

        # 4. Comma-separated (single-line)
        if "," in response:
            return [t.strip() for t in response.split(",") if t.strip()]

        # 5. Last resort: treat the whole response as a single term
        return [response.strip()]

    @staticmethod
    def _find_spans(term: str, text: str) -> list[tuple[int, int]]:
        """
        Returns all (start, end) byte-offset pairs where 'term' appears in 'text', using a case-insensitive word-boundary-aware search.

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


####################
# LLMTermExtractor #
####################

class LLMTermExtractor:
    """
    Zero-shot LLM-based term extraction pipeline.

    Given a dataset in the GBIE format (dict mapping paper IDs to content dicts with a text field), this class:
        1. Formats each text with the user-supplied prompts.
        2. Sends each text through the configured LLM provider.
        3. Parses the raw response into structured entity dicts.
        4. Returns results in the same format as HFTermExtractor.perform_inference().

    Responses are optionally cached to a JSONL checkpoint file so that long inference runs can be interrupted and resumed without re-processing already completed examples.
    """

    def __init__(self, provider: BaseLLMProvider, system_prompt: str, user_prompt_template: str, temperature: float = DEFAULT_TEMPERATURE, checkpoint_path: str | None = None,):
        """
        Initializes the extractor with an LLM provider and prompt configuration.

        :param provider: A configured BaseLLMProvider instance.
        :param system_prompt: The system-role message sent to the model for every inference call.
        :param user_prompt_template: A Python format string whose {text} placeholder will be filled with the source text on each call.
        :param temperature: Sampling temperature forwarded to the provider.
        :param checkpoint_path: Optional path to a JSONL file used for resumable inference checkpointing. Already-processed example IDs are skipped on restart.
        """
        self.provider = provider
        self.system_prompt = system_prompt
        self.user_prompt_template = user_prompt_template
        self.temperature = temperature
        self.checkpoint_path = checkpoint_path

    # -- Public interface --
    def perform_inference(self, data: dict) -> dict:
        """
        Runs zero-shot term extraction over an entire dataset.

        :param data: A dict mapping paper IDs to content dicts (GBIE format). Each content dict must have a "text" field.
        :return: A copy of 'data' with an "entities" list populated for every paper. Each entity follows the HFTermExtractor schema: {start_idx, end_idx, location, text_span, label, confscore}.
        """
        # Load already-processed IDs from an existing checkpoint so that a re-run after interruption skips completed examples.
        processed = self._load_checkpoint()

        result = {}

        with self._open_checkpoint() as ckpt_file:
            for id, content in tqdm(data.items(), total=len(data), desc="LLM Inference"):
                # Resume: reconstruct result dict from checkpoint
                if id in processed:
                    result[id] = processed[id]
                    continue

                text = content.get('text')
                if not text:
                    continue
                entity_predictions = []

                raw_response = self._call_llm(text)
                entities = ResponseParser.parse(raw_response, text, LLM_LOCATION_PLACEHOLDER)
                entity_predictions.extend(entities)

                output_content = dict(content)
                output_content["entities"] = entity_predictions
                result[id] = output_content

                # Persist immediately so progress is not lost on crash
                if ckpt_file is not None:
                    ckpt_file.write(
                        json.dumps({"id": id, "content": output_content}) + "\n"
                    )
                    ckpt_file.flush()

        return result

    def run_raw_inference(self, data: dict) -> dict:
        """
        Runs inference and returns the raw LLM responses instead of parsed entities.
        Useful for debugging prompt quality or post-processing with a custom parser.

        :param data: A dict mapping paper IDs to content dicts (GBIE format).
        :return: A dict mapping paper IDs to {"raw_response": str}.
        """
        processed_raw = self._load_raw_checkpoint()
        result = {}

        with self._open_checkpoint(suffix=".raw") as ckpt_file:
            for id, content in tqdm(data.items(), total=len(data), desc="LLM Raw Inference"):
                if id in processed_raw:
                    result[id] = processed_raw[id]
                    continue

                text = content.get('text')
                if not text:
                    continue
                entry = {"raw_response": self._call_llm(text) if text else ""}
                result[id] = entry

                if ckpt_file is not None:
                    ckpt_file.write(
                        json.dumps({"id": id, **entry}) + "\n"
                    )
                    ckpt_file.flush()

        return result

    ####################
    # Internal helpers #
    ####################

    def _call_llm(self, text: str) -> str:
        """
        Formats the user prompt template with the given text and calls the provider.

        :param text: Source text to embed into the user_prompt_template.
        :return: Raw string response from the LLM.
        """
        user_prompt = self.user_prompt_template.format(text=text)
        return self.provider.chat_completion(
            system_prompt=self.system_prompt,
            user_prompt=user_prompt,
            temperature=self.temperature,
        )

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
                    paper_id = entry.pop("id")
                    processed[paper_id] = entry
                except (json.JSONDecodeError, KeyError):
                    pass
        return processed

    def _open_checkpoint(self, suffix: str = ""):
        """
        Opens the checkpoint file for appending, or returns a null context if no checkpoint path was configured.

        :param suffix: Optional suffix appended to self.checkpoint_path (e.g. ".raw").
        :return: A file object opened in append mode, or None.
        """
        if self.checkpoint_path is None:
            # Return None; callers guard with 'if ckpt_file is not None'
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

####################
# Helper functions #
####################

def load_prompts(prompts_path: str, system_key: str = "base", user_key: str = "base") -> tuple[str, str]:
    """
    Loads system and user prompt strings from a JSON file.

    The expected file structure is:
        {
            "system_prompts": { "base": "...", ... },
            "user_prompts":   { "base": "...", ... }
        }

    :param prompts_path: Path to the prompts JSON file.
    :param system_key: Key selecting the system prompt variant.
    :param user_key: Key selecting the user prompt variant.
    :return: (system_prompt, user_prompt_template) tuple of strings.
    :raises KeyError: If the requested key is absent from the prompts file.
    """
    with open(prompts_path, "r", encoding="utf-8") as f:
        prompts = json.load(f)

    system_prompt        = prompts["system_prompts"][system_key]
    user_prompt_template = prompts["user_prompts"][user_key]
    return system_prompt, user_prompt_template


#######
# CLI #
#######

def parse_args() -> argparse.Namespace:
    """
    Defines and parses command-line arguments for zero-shot LLM inference.

    :return: An argparse.Namespace object with all parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Zero-shot LLM-based biomedical term extraction.  "
            "Supports OpenAI, Azure, Groq, LM Studio, Ollama, and HuggingFace backends."
        )
    )

    # -- Provider --
    parser.add_argument(
        "--provider",
        type=str,
        required=True,
        choices=sorted(PROVIDER_REGISTRY.keys()),
        help="LLM backend to use (e.g. 'openai', 'lmstudio', 'huggingface')",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help=(
            "Model identifier as expected by the provider "
            "(e.g. 'gpt-4o', 'gemma-3-27b-it', 'llama3')"
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
        required=True,
        help="Path to the JSON file to run inference on (GBIE format).",
    )
    parser.add_argument(
        "--inference_output_path",
        type=str,
        required=True,
        help="Path where inference results JSON will be written.",
    )
    parser.add_argument(
        "--prompts_path",
        type=str,
        default="../data/prompts.json",
        help="Path to the prompts JSON file (default: ../data/prompts.json).",
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
            "Optional path to a ground-truth JSON file.  When provided, "
            "precision, recall, and F1 are printed after inference."
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

    # -- Raw output mode --
    parser.add_argument(
        "--raw",
        action="store_true",
        help=(
            "Save raw LLM responses instead of parsed entities.  "
            "Useful for debugging prompt quality."
        ),
    )

    return parser.parse_args()


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
    )

    # -- Load prompts --
    system_prompt, user_prompt_template = load_prompts(
        prompts_path=args.prompts_path,
        system_key=args.system_prompt_key,
        user_key=args.user_prompt_key,
    )

    # -- Build extractor --
    extractor = LLMTermExtractor(
        provider=provider,
        system_prompt=system_prompt,
        user_prompt_template=user_prompt_template,
        temperature=args.temperature,
        checkpoint_path=args.checkpoint_path,
    )

    # -- Load inference data --
    data = load_json_data(args.inference_data_path)

    # -- Run inference --
    if args.raw:
        results = extractor.run_raw_inference(data)
    else:
        results = extractor.perform_inference(data)

    save_json_data(results, args.inference_output_path)
    print(f"Inference results saved to {args.inference_output_path}")

    # -- Optional evaluation --
    if args.eval_data_path and not args.raw:
        ground_truth = load_json_data(args.eval_data_path)
        evaluator = TermExtractionEvaluator()
        metrics = evaluator.evaluate(results, ground_truth)
        print(
            f"Evaluation  P: {metrics['precision']:.4f} | "
            f"R: {metrics['recall']:.4f} | "
            f"F1: {metrics['f1']:.4f}"
        )

    maybe_empty_cache(device)


if __name__ == "__main__":
    args = parse_args()
    run_inference(args)
