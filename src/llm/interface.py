"""
LLM provider abstraction.

Unified interface for LLM completions supporting OpenAI, Ollama, and mock providers.
"""
from abc import ABC, abstractmethod
from typing import Any, Optional
import json
import re


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""
    
    @property
    def is_mock(self) -> bool:
        """Return True if this is a mock provider.

        Returns:
            True if the provider is a mock; otherwise False.
        """
        return False
    
    @abstractmethod
    def complete(self, prompt: str, temperature: float = 0.3) -> str:
        """Generate a completion for the given prompt.

        Args:
            prompt: Prompt text to complete.
            temperature: Sampling temperature (0-1).

        Returns:
            The model-generated completion text.
        """
        pass
    
    def complete_json(self, prompt: str, temperature: float = 0.1) -> dict[str, Any]:
        """Generate a completion and parse it as JSON.

        Args:
            prompt: Prompt text (should instruct the model to output JSON).
            temperature: Sampling temperature (0-1).

        Returns:
            A parsed JSON object.

        Raises:
            ValueError: If JSON cannot be extracted from the completion.
        """
        response = self.complete(prompt, temperature)
        # Try to extract JSON from response
        return self._extract_json(response)
    
    def _extract_json(self, text: str) -> dict[str, Any]:
        """Extract a JSON object from text that may include extra formatting.

        Args:
            text: Raw model output that should contain a JSON object.

        Returns:
            The parsed JSON object.

        Raises:
            ValueError: If JSON cannot be extracted/parsed from the text.
        """
        # Try direct parse first
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        
        # Try to find JSON block in markdown
        # Many models wrap JSON in ```json ... ``` fences; extract the first object block.
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
        if json_match:
            return json.loads(json_match.group(1))
        
        # Try to find any JSON object
        json_match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except json.JSONDecodeError:
                pass
        
        raise ValueError(f"Could not extract JSON from response: {text[:200]}...")


class OpenAILLMProvider(LLMProvider):
    """OpenAI LLM provider."""
    
    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-mini",
        max_tokens: int = 1000
    ) -> None:
        """Initialize the OpenAI provider.

        Args:
            api_key: OpenAI API key.
            model: Model name.
            max_tokens: Maximum tokens in the response.

        Raises:
            ImportError: If the `openai` package is not installed.
        """
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai package required")
        
        self._client = OpenAI(api_key=api_key)
        self._model = model
        self._max_tokens = max_tokens
    
    def complete(self, prompt: str, temperature: float = 0.3) -> str:
        """Generate a completion using the OpenAI chat completions API.

        Args:
            prompt: Prompt text to complete.
            temperature: Sampling temperature (0-1).

        Returns:
            The model-generated completion text.
        """
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that outputs structured data. Always respond with valid JSON when requested."},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_tokens=self._max_tokens,
        )
        return response.choices[0].message.content


class OllamaLLMProvider(LLMProvider):
    """
    Ollama LLM provider for local models.
    
    Recommended model: qwen2.5:7b-instruct
    - Excellent instruction following
    - Strong JSON output compliance
    - Good balance of speed and quality for local inference
    - Multilingual support (useful for personal memory systems)
    """
    
    # Default to Qwen for best local performance
    DEFAULT_MODEL = "qwen2.5:7b-instruct"
    
    # System prompt optimized for structured extraction
    SYSTEM_PROMPT = """You are a memory extraction assistant. Your task is to analyze text and output structured JSON data.

CRITICAL RULES:
1. ALWAYS respond with valid JSON only - no explanations, no markdown
2. Follow the exact schema requested in the prompt
3. Never return null; use empty strings/arrays or defaults instead
4. Be precise and factual in extractions"""

    def __init__(
        self,
        model: Optional[str] = None,
        base_url: str = "http://localhost:11434",
        max_tokens: int = 1000,
        temperature: float = 0.2,  # Lower default for determinism
    ) -> None:
        """Initialize the Ollama provider.

        Args:
            model: Model name (default: qwen2.5:7b-instruct).
            base_url: Ollama API base URL.
            max_tokens: Maximum tokens in response.
            temperature: Default sampling temperature (lower is more deterministic).
        """
        self._model = model or self.DEFAULT_MODEL
        self._base_url = base_url.rstrip("/")
        self._max_tokens = max_tokens
        self._default_temperature = temperature
    
    @property
    def model_name(self) -> str:
        """Return the model name being used.

        Returns:
            The configured Ollama model name.
        """
        return self._model
    
    def complete(self, prompt: str, temperature: Optional[float] = None) -> str:
        """Generate a completion using the Ollama `/api/generate` endpoint.

        Args:
            prompt: Prompt text to complete.
            temperature: Sampling temperature (uses instance default if None).

        Returns:
            The model-generated completion text.

        Raises:
            ConnectionError: If the Ollama server cannot be reached.
            ValueError: If the requested model is not found.
            httpx.HTTPError: For other HTTP-layer failures.
        """
        import httpx
        
        temp = temperature if temperature is not None else self._default_temperature
        
        try:
            response = httpx.post(
                f"{self._base_url}/api/generate",
                json={
                    "model": self._model,
                    "prompt": prompt,
                    "system": self.SYSTEM_PROMPT,
                    "stream": False,
                    "options": {
                        "temperature": temp,
                        "num_predict": self._max_tokens,
                        # Qwen-specific optimizations
                        "top_p": 0.9,
                        "repeat_penalty": 1.1,
                    }
                },
                timeout=120.0  # Longer timeout for local inference
            )
            response.raise_for_status()
            return response.json()["response"]
        except httpx.ConnectError:
            raise ConnectionError(
                f"Could not connect to Ollama at {self._base_url}. "
                f"Make sure Ollama is running: `ollama serve`"
            )
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                raise ValueError(
                    f"Model '{self._model}' not found. "
                    f"Pull it first: `ollama pull {self._model}`"
                )
            raise
    
    def is_available(self) -> bool:
        """Return True if the Ollama server is reachable.

        Returns:
            True if the server responds successfully; otherwise False.
        """
        import httpx
        try:
            response = httpx.get(f"{self._base_url}/api/tags", timeout=5.0)
            return response.status_code == 200
        except Exception:
            return False
    
    def list_models(self) -> list[str]:
        """List available models from the Ollama server.

        Returns:
            A list of model names, or an empty list on failure.
        """
        import httpx
        try:
            response = httpx.get(f"{self._base_url}/api/tags", timeout=10.0)
            response.raise_for_status()
            models = response.json().get("models", [])
            return [m["name"] for m in models]
        except Exception:
            return []


class MockLLMProvider(LLMProvider):
    """
    Mock LLM provider for testing without external dependencies.
    
    Produces deterministic, reasonable responses based on prompt analysis.
    """
    
    def __init__(self) -> None:
        """Initialize the mock provider."""
    
    @property
    def is_mock(self) -> bool:
        """Return True for mock provider.

        Returns:
            True.
        """
        return True
    
    def complete(self, prompt: str, temperature: float = 0.3) -> str:
        """Generate a deterministic mock completion based on prompt patterns.

        Args:
            prompt: Prompt text to "complete".
            temperature: Unused; present to match interface.

        Returns:
            A JSON string matching the requested schema when recognizable.
        """
        prompt_lower = prompt.lower()
        
        # Memory worthiness classification
        if "memory curator" in prompt_lower and "worth remembering" in prompt_lower:
            # Check for likely memory-worthy content
            has_personal = any(p in prompt_lower for p in [
                "i am", "i'm", "my name", "i want", "i like", "i prefer",
                "i learned", "i work", "i live", "my goal"
            ])
            
            return json.dumps({
                "is_memory_worthy": has_personal,
                "confidence": 0.8 if has_personal else 0.3,
                "reason": "Contains personal information" if has_personal else "No personal information detected",
                "memory_type": "episodic" if has_personal else "none"
            })
        
        # Episode extraction
        if "extracting structured episodic memory" in prompt_lower:
            # Extract some basic info from the prompt
            return json.dumps({
                "content": "User shared information (mock extraction)",
                "memory_type": "episodic",
                "topics": ["general"],
                "entities": [],
                "importance": 0.5,
                "occurred_at_offset": "none"
            })
        
        # Summarization
        if "creating a narrative summary" in prompt_lower:
            return json.dumps({
                "summary": "Mock summary of recent events and memories.",
                "key_events": ["Event 1", "Event 2"],
                "themes": ["general theme"],
                "notable_changes": []
            })
        
        # Fact extraction
        if "extracting stable facts" in prompt_lower:
            return json.dumps({
                "new_facts": [],
                "updated_facts": [],
                "contradicted_facts": []
            })
        
        # Query analysis
        if "analyze this query" in prompt_lower:
            return json.dumps({
                "query_type": "semantic",
                "time_relevance": "all_time",
                "time_filter": {"since": "", "until": ""},
                "search_concepts": ["general"],
                "topic_filters": [],
                "reformulated_query": "general query"
            })
        
        # Answer synthesis
        if "synthesizing an answer" in prompt_lower:
            return json.dumps({
                "answer": "Based on the available memories, here is a synthesized response.",
                "confidence": 0.7,
                "key_sources": ["memory source"],
                "gaps": ["some information might be missing"]
            })
        
        # Narrative synthesis
        if "reconstructing a narrative" in prompt_lower:
            return json.dumps({
                "narrative": "This is a mock narrative of the user's journey.",
                "timeline": [{"date": "2024-01-01", "event": "Mock event"}],
                "key_moments": ["Key moment"],
                "current_status": "Status unknown"
            })
        
        # Default: return a generic JSON response
        return json.dumps({
            "response": "Mock response",
            "status": "success"
        })


def get_llm_provider(
    provider: str = "openai",
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    temperature: Optional[float] = None,
) -> LLMProvider:
    """Factory to create an `LLMProvider` implementation.

    Args:
        provider: Provider name: "openai", "ollama", or "mock".
        api_key: API key (required for OpenAI).
        model: Model name (optional; uses provider defaults).
        base_url: Base URL (for Ollama).
        temperature: Default temperature (for Ollama).

    Returns:
        A configured `LLMProvider` instance.

    Raises:
        ValueError: If the provider is unknown or required configuration is missing.
    """
    if provider == "openai":
        if not api_key:
            raise ValueError("API key required for OpenAI provider")
        return OpenAILLMProvider(
            api_key=api_key,
            model=model or "gpt-4o-mini"
        )
    elif provider == "ollama":
        return OllamaLLMProvider(
            model=model,  # Will use DEFAULT_MODEL if None
            base_url=base_url or "http://localhost:11434",
            temperature=temperature or 0.2
        )
    elif provider == "mock":
        return MockLLMProvider()
    else:
        raise ValueError(f"Unknown provider: {provider}")

