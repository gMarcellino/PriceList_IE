#!/usr/bin/env python3
"""
LLMProvider.py

LLM provider module.
Supports both online providers (OpenAI, Azure OpenAI, Groq, and any other OpenAI-compatible endpoint) and local providers (LM Studio, Ollama, and HuggingFace Transformers running on the same machine).

Reusable exports:
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
    get_device,
    load_json_data,
    maybe_empty_cache,
    print_device_info,
    save_json_data,
)

#############
# Constants #
#############

# Default generation parameters used across all providers
DEFAULT_TEMPERATURE = 1.0
DEFAULT_MAX_NEW_TOKENS = 512

# Confidence score placeholder: LLMs do not expose per-span probabilities,
# so we assign a uniform sentinel value that downstream code can detect.
LLM_CONFSCORE_PLACEHOLDER = -1.0

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
    # Helper functions
    "load_prompts",
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


def _validate_chat_content(name: str, content: str | None, allow_none: bool = False) -> str | None:
    """
    Ensures OpenAI-compatible chat message content is a plain string.
    """
    if content is None and allow_none:
        return None
    if not isinstance(content, str):
        raise TypeError(
            f"{name} must be a string before it can be sent as chat message content; "
            f"got {type(content).__name__}."
        )
    return content


def _build_chat_messages(system_prompt: str | None, user_prompt: str) -> list[dict[str, str]]:
    """
    Builds chat messages, omitting the system role when no system prompt is provided.
    """
    system_prompt = _validate_chat_content("system_prompt", system_prompt, allow_none=True)
    user_prompt = _validate_chat_content("user_prompt", user_prompt)

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

        :param model: Model identifier as expected by the target server (e.g. "gpt-4o", "medgemma-27b-text-it", "llama3").
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

SUPPORTED_INFERENCE_DTYPES = ("auto", "float16", "bfloat16", "float32")


def _is_gemma3_multimodal_config(config) -> bool:
    """
    Returns True for Gemma 3 checkpoints whose top-level config includes the
    vision tower, even when inference only needs their language model.
    """
    return getattr(config, "model_type", None) == "gemma3"


def _mps_supports_bfloat16() -> bool:
    """
    Probes whether the active PyTorch/MPS stack can execute basic BF16 ops.
    """
    if not (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    ):
        return False
    try:
        value = torch.ones(1, device="mps", dtype=torch.bfloat16)
        (value + value).cpu()
        return True
    except Exception:
        return False


def _resolve_inference_dtype(
    dtype_name: str,
    device: str,
    prefer_bfloat16: bool = False,
):
    """
    Resolves a configured dtype while keeping Gemma 3 out of unstable FP16 on
    Apple Silicon. Gemma 3 checkpoints are natively BF16; FP16 can produce NaN
    logits on MPS, which appear as endless generated pad tokens.
    """
    if dtype_name not in SUPPORTED_INFERENCE_DTYPES:
        raise ValueError(
            f"Unsupported torch_dtype '{dtype_name}'. "
            f"Available: {SUPPORTED_INFERENCE_DTYPES}"
        )

    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "float32":
        return torch.float32
    if dtype_name == "bfloat16":
        if device == "mps" and not _mps_supports_bfloat16():
            raise ValueError(
                "torch_dtype='bfloat16' was requested, but this PyTorch/MPS "
                "installation cannot execute BF16 operations."
            )
        return torch.bfloat16

    if prefer_bfloat16:
        if device == "cuda" and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        if device == "mps":
            if _mps_supports_bfloat16():
                return torch.bfloat16
            print(
                "Warning: BF16 is unavailable on this MPS installation; "
                "loading Gemma 3 in float32 to avoid unstable float16 logits."
            )
            return torch.float32

    # Preserve the provider's existing defaults for other model families.
    return torch.float16 if device == "mps" else torch.float32


def _model_load_dtype_kwarg(transformers_version: str, dtype) -> dict:
    """
    Uses the current `dtype` keyword while retaining compatibility with
    Transformers releases that only support the legacy `torch_dtype` name.
    """
    version_match = re.match(r"^(\d+)\.(\d+)", transformers_version)
    version = (
        (int(version_match.group(1)), int(version_match.group(2)))
        if version_match
        else (0, 0)
    )
    key = "dtype" if version >= (4, 56) else "torch_dtype"
    return {key: dtype}


def _as_structured_text_messages(messages: list[dict]) -> list[dict]:
    """
    Converts plain chat content to the typed content format documented for
    Gemma 3. It remains text-only and does not require an AutoProcessor.
    """
    return [
        {
            "role": message["role"],
            "content": [{"type": "text", "text": message["content"]}],
        }
        for message in messages
    ]


class _FirstStepNaNLogitsProcessor:
    """
    Fails immediately when a backend/dtype combination produces NaN logits,
    instead of spending thousands of decoding steps generating pad tokens.
    """

    def __init__(self, device: str, dtype):
        self.device = device
        self.dtype = dtype
        self.checked = False

    def __call__(self, input_ids, scores):
        if not self.checked:
            self.checked = True
            if torch.isnan(scores).any().item():
                raise RuntimeError(
                    "The model produced NaN logits on the first generation "
                    f"step (device={self.device}, dtype={self.dtype}). "
                    "For Gemma 3 on Apple Silicon, use torch_dtype='bfloat16' "
                    "or 'float32' and a recent PyTorch/Transformers release."
                )
        return scores


class HuggingFaceLocalProvider(BaseLLMProvider):
    """
    Provider wrapper for HuggingFace causal language models loaded locally via the Transformers library.

    This is the local-inference counterpart of the script in classify_terms.py.
    It supports CUDA, Apple Silicon MPS, and CPU execution.  
    For very large models (e.g. 27 B parameters) consider quantisation or switching to an OpenAI-compatible local server such as LM Studio or Ollama instead.
    """

    def __init__(
        self,
        model_id: str,
        device: str | None = None,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        top_p: float = 1.0,
        torch_dtype: str = "auto",
    ):
        """
        Loads the tokenizer and model from the HuggingFace Hub (or a local path).

        :param model_id: HuggingFace model identifier or path to a local directory (e.g. "google/medgemma-27b-text-it").
        :param device: Target device string ("cuda", "mps", "cpu"). If None, the best available device is selected automatically via get_device().
        :param max_new_tokens: Maximum tokens the model may generate per call.
        :param top_p: Nucleus sampling parameter.
        :param torch_dtype: Weight dtype ("auto", "float16", "bfloat16", or "float32").
        """
        try:
            import transformers
            from transformers import (
                AutoConfig,
                AutoModelForCausalLM,
                AutoTokenizer,
                LogitsProcessorList,
            )
        except ImportError as exc:
            raise ImportError(
                "The 'transformers' package is required for HuggingFaceLocalProvider.  "
                "Install it with: pip install transformers"
            ) from exc

        self.max_new_tokens = max_new_tokens
        self.top_p = top_p

        # Prefer explicitly requested device; fall back to get_device() utility
        self.device = device or get_device().type

        config = AutoConfig.from_pretrained(model_id)
        self._uses_gemma3_text_format = _is_gemma3_multimodal_config(config)
        self.dtype = _resolve_inference_dtype(
            dtype_name=torch_dtype,
            device=self.device,
            prefer_bfloat16=self._uses_gemma3_text_format,
        )
        self._logits_processor_list_cls = LogitsProcessorList

        print(
            f"Loading '{model_id}' onto '{self.device}' "
            f"with dtype '{self.dtype}'..."
        )

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

        model_cls = AutoModelForCausalLM
        if self._uses_gemma3_text_format:
            # Gemma 3 4B/12B/27B checkpoints store their decoder under the
            # `language_model.*` prefix. Loading these checkpoints directly
            # with Gemma3ForCausalLM expects `model.*` keys instead and can
            # silently initialize a new random language model. Keep the exact
            # checkpoint architecture even though this provider supplies only
            # text inputs.
            try:
                model_cls = transformers.Gemma3ForConditionalGeneration
            except AttributeError as exc:
                raise ImportError(
                    "Gemma 3 inference requires transformers>=4.50.0."
                ) from exc

        model_kwargs = {
            "low_cpu_mem_usage": True,
            **_model_load_dtype_kwarg(transformers.__version__, self.dtype),
        }
        print(f"Using Hugging Face model class: {model_cls.__name__}")
        self.model = model_cls.from_pretrained(model_id, **model_kwargs)
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
        if self._uses_gemma3_text_format:
            messages = _as_structured_text_messages(messages)

        try:
            self.tokenizer.apply_chat_template(messages, tokenize=False)
        except Exception:
            # Template rejected the messages (likely system role unsupported)
            if system_prompt:
                combined = f"{system_prompt}\n\n{user_prompt}"
                messages = [{"role": "user", "content": combined}]
                if self._uses_gemma3_text_format:
                    messages = _as_structured_text_messages(messages)

        # apply_chat_template formats the messages using the model's own template
        # (e.g. Gemma's <start_of_turn> markers) and returns ready-to-use tensors.
        inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        if (
            self._uses_gemma3_text_format
            and "token_type_ids" not in inputs
        ):
            # Gemma3ForConditionalGeneration uses 0 for text tokens and 1 for
            # image tokens. AutoProcessor normally supplies this field, but a
            # merged LoRA directory often contains only tokenizer artifacts.
            inputs["token_type_ids"] = torch.zeros_like(inputs["input_ids"])
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        input_len = inputs["input_ids"].shape[-1]

        # Only use sampling when temperature > 0; greedy decoding is deterministic and avoids invalid temperature arguments on some model configurations.
        do_sample = temperature > 0.0
        bad_words_ids, suppress_tokens = _build_suppressed_token_args(self.tokenizer)
        generation_kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": do_sample,
            "logits_processor": self._logits_processor_list_cls(
                [_FirstStepNaNLogitsProcessor(self.device, self.dtype)]
            ),
        }
        if do_sample:
            generation_kwargs["temperature"] = temperature
            generation_kwargs["top_p"] = self.top_p
        if bad_words_ids is not None:
            generation_kwargs["bad_words_ids"] = bad_words_ids
        if suppress_tokens is not None:
            generation_kwargs["suppress_tokens"] = suppress_tokens

        # Keep the checkpoint's generation_config, including Gemma's
        # end-of-turn token. Only provide a padding fallback when the model
        # itself has none.
        generation_config = getattr(self.model, "generation_config", None)
        if getattr(generation_config, "pad_token_id", None) is None:
            pad_token_id = self.tokenizer.pad_token_id
            if pad_token_id is None:
                pad_token_id = self.tokenizer.eos_token_id
            generation_kwargs["pad_token_id"] = pad_token_id

        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                **generation_kwargs,
            )

        # Slice off the prompt tokens so we return only the generated portion
        generated_ids = output_ids[0][input_len:]
        decoded = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        if not decoded:
            first_generated_id = generated_ids[0].item() if generated_ids.numel() else None
            generation_eos_token_id = getattr(
                getattr(self.model, "generation_config", None),
                "eos_token_id",
                None,
            )
            print(
                "Warning: HuggingFaceLocalProvider generated an empty response "
                f"(first_generated_token_id={first_generated_id}, "
                f"generated_token_count={generated_ids.numel()}, "
                f"pad_token_id={self.tokenizer.pad_token_id}, "
                f"tokenizer_eos_token_id={self.tokenizer.eos_token_id}, "
                f"generation_eos_token_id={generation_eos_token_id})."
            )

        # Flush the MPS cache to reduce memory fragmentation on Apple Silicon
        if self.device == "mps":
            try:
                torch.mps.empty_cache()
            except Exception:
                pass

        return decoded



def _build_suppressed_token_args(tokenizer) -> tuple[list[list[int]] | None, list[int] | None]:
    """
    Prevent generation from starting with pad-only output for Gemma-family models.
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


def build_provider(provider_name: str, model: str, api_key: str | None = None, base_url: str | None = None, azure_endpoint: str | None = None, api_version: str = "2024-02-01", max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS, device: str | None = None, torch_dtype: str = "auto",) -> BaseLLMProvider:
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
    :param torch_dtype: Weight dtype for HuggingFaceLocalProvider.
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
            torch_dtype=torch_dtype,
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

