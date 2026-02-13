"""
API Adapter Layer for LLM Evaluation Framework

CRITICAL: This module handles TECHNICAL API formatting only, NOT prompt content modification.

PURPOSE:
Different LLM providers and API backends expect different input formats even though the
prompt CONTENT remains identical across all providers (required for fair comparison).

ARCHITECTURE:
This module separates two orthogonal concerns:
1. PROVIDER (model): Which model-specific template to apply (e.g., Gemma3 chat markers)
2. BACKEND (API): Which API format to use (e.g., Replicate, Direct, Hugging Face)

FAIRNESS GUARANTEE:
- The semantic prompt content is IDENTICAL across all providers and backends
- Only the API-level wrapper format changes to match technical requirements
- Example: "Diagnose fault X" remains "Diagnose fault X" in all cases
  - (GEMINI, DIRECT) → Gemini's native API format
  - (GEMINI, REPLICATE) → Replicate's simple prompt format
  - (GEMMA3, REPLICATE) → Gemma3 chat template via Replicate
  - (GEMMA3, HUGGINGFACE) → Gemma3 chat template via Hugging Face

WHY THIS IS NEEDED:
Without proper API formatting, models like Gemma 3 IT underperform because they expect
specific chat template markers. This is a TECHNICAL requirement, not a prompt engineering trick.

EXTENSIBILITY:
This design makes it easy for future developers to:
- Add new backends (e.g., AWS Bedrock, Azure OpenAI)
- Use the same model through different APIs
- Mix and match providers and backends

REFERENCES:
- Gemma 3 formatting: https://ai.google.dev/gemma/docs/formatting
- Gemini API: https://ai.google.dev/api/rest
"""

from typing import Dict, Any, Optional
from .core import LLMProvider, APIBackend


class APIAdapter:
    """
    Adapter for converting identical prompt content into provider+backend-specific API formats.

    This class ensures FAIRNESS by keeping prompt content constant while adapting
    to each provider's technical API requirements and backend's API format.

    Architecture:
    - prepare_input() → routes to backend-specific formatter
    - Backend formatter → applies provider-specific template if needed
    - Template helpers → reusable chat templates (Gemma3, DeepSeek, etc.)
    """

    @staticmethod
    def prepare_input(
        prompt_content: str,
        provider: LLMProvider,
        backend: APIBackend = APIBackend.REPLICATE,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Convert prompt content to provider+backend-specific API format.

        IMPORTANT: This method does NOT modify the prompt content itself, only wraps
        it in the format expected by each backend's API. The semantic meaning of the
        prompt remains identical across all providers and backends.

        Args:
            prompt_content: The actual prompt text (IDENTICAL across all providers)
            provider: Target LLM provider (determines which model/template)
            backend: API backend to use (default: REPLICATE for backward compatibility)
            system_prompt: Optional system instruction (if supported)
            **kwargs: Additional parameters (temperature, max_tokens, etc.)

        Returns:
            Dictionary with backend-specific input format

        Examples:
            >>> # Same content, different backend+provider combinations
            >>> content = "What causes error code 502?"
            >>>
            >>> # Gemini via Replicate (simple prompt)
            >>> APIAdapter.prepare_input(content, LLMProvider.GEMINI, APIBackend.REPLICATE)
            >>> # Returns: {"prompt": "What causes error code 502?"}
            >>>
            >>> # Gemma3 via Replicate (with chat template)
            >>> APIAdapter.prepare_input(content, LLMProvider.GEMMA3, APIBackend.REPLICATE)
            >>> # Returns: {"prompt": "<start_of_turn>user\\nWhat causes..."}
            >>>
            >>> # Gemini via Direct API (structured format)
            >>> APIAdapter.prepare_input(content, LLMProvider.GEMINI, APIBackend.DIRECT)
            >>> # Returns: {"contents": [{"role": "user", "parts": [{"text": "What causes..."}]}]}
            >>>
            >>> # Future: Gemma3 via Hugging Face
            >>> APIAdapter.prepare_input(content, LLMProvider.GEMMA3, APIBackend.HUGGINGFACE)
            >>> # Returns: Hugging Face format with Gemma3 template
        """
        # Route to backend-specific formatter
        if backend == APIBackend.REPLICATE:
            return APIAdapter._format_replicate(prompt_content, provider, system_prompt, **kwargs)
        elif backend == APIBackend.DIRECT:
            return APIAdapter._format_direct_api(prompt_content, provider, system_prompt, **kwargs)
        elif backend == APIBackend.HUGGINGFACE:
            return APIAdapter._format_huggingface(prompt_content, provider, system_prompt, **kwargs)
        else:
            raise ValueError(f"Unsupported backend: {backend}")

    # ================= Backend-Specific Formatters =================

    @staticmethod
    def _format_replicate(
        prompt_content: str,
        provider: LLMProvider,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Format for Replicate API.

        Replicate typically uses a simple {"prompt": "..."} format, but some models
        require specific chat templates applied to the prompt text.

        Args:
            prompt_content: Raw prompt text
            provider: LLM provider (determines if template is needed)
            system_prompt: Optional system instruction
            **kwargs: Additional parameters (temperature, max_tokens, etc.)

        Returns:
            {"prompt": formatted_prompt, **kwargs}
        """
        # Apply provider-specific template if needed
        if provider in (LLMProvider.GEMMA3, LLMProvider.GEMMA3_4B,
                       LLMProvider.GEMMA3_12B, LLMProvider.GEMMA3_27B):
            # Gemma3 needs chat template markers
            formatted_prompt = APIAdapter._apply_gemma3_template(prompt_content, system_prompt)
        elif provider == LLMProvider.DEEPSEEK:
            # DeepSeek needs User/Assistant format
            formatted_prompt = APIAdapter._apply_deepseek_template(prompt_content, system_prompt)
        else:
            # Gemini, GPT5, and others use plain prompt via Replicate
            formatted_prompt = prompt_content

        inputs = {"prompt": formatted_prompt}
        inputs.update(kwargs)
        return inputs

    @staticmethod
    def _format_direct_api(
        prompt_content: str,
        provider: LLMProvider,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Format for Direct API calls (provider's official API).

        Each provider has its own native API format:
        - Gemini: Structured contents array with role/parts
        - GPT-5: Messages array with role/content
        - Others: May vary

        Args:
            prompt_content: Raw prompt text
            provider: LLM provider (determines API format)
            system_prompt: Optional system instruction
            **kwargs: Additional parameters

        Returns:
            Provider-specific API format
        """
        if provider == LLMProvider.GEMINI:
            # Gemini's native API uses structured format
            contents = [{
                "role": "user",
                "parts": [{"text": prompt_content}]
            }]

            result: Dict[str, Any] = {"contents": contents}

            # System prompt handled separately in Gemini API
            if system_prompt:
                result["systemInstruction"] = {
                    "parts": [{"text": system_prompt}]
                }

            # Add generation config if provided
            if kwargs:
                # Map common kwargs to Gemini's parameter names
                generation_config = {}
                if "temperature" in kwargs:
                    generation_config["temperature"] = kwargs["temperature"]
                if "max_tokens" in kwargs:
                    generation_config["maxOutputTokens"] = kwargs["max_tokens"]
                if "top_p" in kwargs:
                    generation_config["topP"] = kwargs["top_p"]
                if "seed" in kwargs:
                    generation_config["seed"] = kwargs["seed"]

                if generation_config:
                    result["generationConfig"] = generation_config

                # Pass through any other kwargs not in generation_config
                for key, value in kwargs.items():
                    if key not in ("temperature", "max_tokens", "top_p", "seed"):
                        result[key] = value

            return result

        elif provider == LLMProvider.GPT5:
            # GPT-5's native API uses messages array
            messages = []

            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})

            messages.append({"role": "user", "content": prompt_content})

            result = {"messages": messages}
            result.update(kwargs)
            return result

        else:
            # For other providers, assume simple prompt format
            # (can be extended as more providers are added)
            result = {"prompt": prompt_content}
            if system_prompt:
                result["system_prompt"] = system_prompt
            result.update(kwargs)
            return result

    @staticmethod
    def _format_huggingface(
        prompt_content: str,
        provider: LLMProvider,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Format for Hugging Face Inference API.

        Hugging Face typically uses {"inputs": "..."} format. For chat models,
        we apply the appropriate chat template before passing to the API.

        Args:
            prompt_content: Raw prompt text
            provider: LLM provider (determines if template is needed)
            system_prompt: Optional system instruction
            **kwargs: Additional parameters

        Returns:
            {"inputs": formatted_prompt, "parameters": {...}}
        """
        # Apply provider-specific template if needed
        if provider in (LLMProvider.GEMMA3, LLMProvider.GEMMA3_4B,
                       LLMProvider.GEMMA3_12B, LLMProvider.GEMMA3_27B):
            # Gemma3 needs chat template markers
            formatted_prompt = APIAdapter._apply_gemma3_template(prompt_content, system_prompt)
        elif provider == LLMProvider.DEEPSEEK:
            # DeepSeek needs User/Assistant format
            formatted_prompt = APIAdapter._apply_deepseek_template(prompt_content, system_prompt)
        else:
            # Plain prompt for other models
            formatted_prompt = prompt_content
            if system_prompt:
                formatted_prompt = f"{system_prompt}\n\n{formatted_prompt}"

        # Hugging Face format
        result: Dict[str, Any] = {"inputs": formatted_prompt}

        # Parameters go in separate dict
        if kwargs:
            result["parameters"] = kwargs

        return result

    # ================= Chat Template Helpers =================

    @staticmethod
    def _apply_gemma3_template(
        prompt_content: str,
        system_prompt: Optional[str] = None
    ) -> str:
        """
        Apply Gemma 3 IT chat template markers.

        Gemma 3 IT expects chat turn markers following the official format:
        <start_of_turn>user
        {user message}
        <end_of_turn>
        <start_of_turn>model
        {model response}
        <end_of_turn>

        TECHNICAL REQUIREMENT:
        Without these markers, Gemma 3 IT underperforms because it's specifically
        trained to expect this format. This is documented in the official Gemma docs.

        FAIRNESS NOTE:
        The prompt_content itself is unchanged. We only add the technical markers
        that signal "this is a user message" to the model. This is equivalent to
        using the "role: user" field in other APIs - it's metadata, not content.

        Reference: https://ai.google.dev/gemma/docs/formatting

        Args:
            prompt_content: Raw prompt text (unchanged)
            system_prompt: Optional system instruction (prepended as system turn if provided)

        Returns:
            Formatted prompt with chat turn markers
        """
        formatted_parts = []

        # Add system prompt as system turn if provided
        if system_prompt:
            formatted_parts.append(f"<start_of_turn>system\n{system_prompt}<end_of_turn>")

        # Add user message
        formatted_parts.append(f"<start_of_turn>user\n{prompt_content}<end_of_turn>")

        # Add model turn marker (prompts model to respond)
        formatted_parts.append("<start_of_turn>model\n")

        return "\n".join(formatted_parts)

    @staticmethod
    def _apply_deepseek_template(
        prompt_content: str,
        system_prompt: Optional[str] = None
    ) -> str:
        """
        Apply DeepSeek conversational template.

        DeepSeek expects a conversational format with "User:" and "Assistant:" markers.
        This is similar to many instruction-tuned models that use role-based formatting.

        FAIRNESS NOTE:
        Like Gemma 3, this is a technical requirement for the model to recognize
        the input as a user query. The actual prompt content remains unchanged.

        Args:
            prompt_content: Raw prompt text (unchanged)
            system_prompt: Optional system instruction (prepended before user message)

        Returns:
            Formatted prompt with User/Assistant markers
        """
        formatted_parts = []

        # Add system prompt if provided
        if system_prompt:
            formatted_parts.append(system_prompt)

        # Add user message
        formatted_parts.append(f"User: {prompt_content}")

        # Add assistant marker (prompts model to respond)
        formatted_parts.append("Assistant:")

        return "\n\n".join(formatted_parts)
