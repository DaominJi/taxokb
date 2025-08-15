"""
LLM Factory - A unified interface for managing multiple LLM providers
"""

import os
import yaml
import time
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass
from enum import Enum
import asyncio
from concurrent.futures import ThreadPoolExecutor
import json

# Third-party imports (install via pip)
import openai
import google.generativeai as genai
import anthropic
import httpx


class LLMProvider(Enum):
    """Enumeration of supported LLM providers"""
    OPENAI = "openai"
    GOOGLE = "google"
    ANTHROPIC = "anthropic"
    GROK = "grok"


@dataclass
class LLMResponse:
    """Standard response format for all LLM providers"""
    content: str
    model: str
    provider: str
    usage: Dict[str, int]
    raw_response: Any
    metadata: Dict[str, Any] = None


@dataclass
class ModelConfig:
    """Configuration for a specific model"""
    name: str
    max_tokens: int
    temperature: float
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    max_output_tokens: Optional[int] = None  # For Google


class BaseLLM(ABC):
    """Abstract base class for LLM providers"""
    
    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.api_key = self._get_api_key()
        self.base_url = config.get('base_url')
        self.timeout = config.get('timeout', 30)
        self.max_retries = config.get('max_retries', 3)
        self.retry_delay = config.get('retry_delay', 1)
        
    def _get_api_key(self) -> str:
        """Get API key from config or environment variable"""
        api_key = self.config.get('api_key', '')
        if api_key.startswith('${') and api_key.endswith('}'):
            env_var = api_key[2:-1]
            api_key = os.getenv(env_var)
            if not api_key:
                raise ValueError(f"Environment variable {env_var} not found")
        return api_key
    
    @abstractmethod
    async def generate(self, prompt: str, model: Optional[str] = None, **kwargs) -> LLMResponse:
        """Generate a response from the LLM"""
        pass
    
    @abstractmethod
    def generate_sync(self, prompt: str, model: Optional[str] = None, **kwargs) -> LLMResponse:
        """Synchronous version of generate"""
        pass
    
    def _retry_with_backoff(self, func, *args, **kwargs):
        """Retry function with exponential backoff"""
        for attempt in range(self.max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if attempt == self.max_retries - 1:
                    raise
                wait_time = self.retry_delay * (2 ** attempt)
                self.logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {wait_time}s...")
                time.sleep(wait_time)


class OpenAILLM(BaseLLM):
    """OpenAI LLM implementation"""
    
    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        super().__init__(config, logger)
        self.client = openai.OpenAI(api_key=self.api_key)
        if self.base_url:
            self.client.base_url = self.base_url
    
    def generate_sync(self, prompt: str, model: Optional[str] = None, **kwargs) -> LLMResponse:
        """Synchronous generation for OpenAI"""
        model = model or self.config.get('default_model', 'gpt-4o-2024-08-06')
        
        # Get model-specific config
        model_config = self._get_model_config(model)
        
        params = {
            'model': model,
            'messages': [{"role": "user", "content": prompt}],
            'max_tokens': kwargs.get('max_tokens', model_config.get('max_tokens', 2048)),
            'temperature': kwargs.get('temperature', model_config.get('temperature', 0.7)),
            'top_p': kwargs.get('top_p', model_config.get('top_p', 1.0)),
        }
        
        # Add optional parameters
        if 'frequency_penalty' in model_config:
            params['frequency_penalty'] = model_config['frequency_penalty']
        if 'presence_penalty' in model_config:
            params['presence_penalty'] = model_config['presence_penalty']
        
        response = self._retry_with_backoff(
            self.client.chat.completions.create,
            **params
        )
        
        return LLMResponse(
            content=response.choices[0].message.content,
            model=model,
            provider=LLMProvider.OPENAI.value,
            usage={
                'prompt_tokens': response.usage.prompt_tokens,
                'completion_tokens': response.usage.completion_tokens,
                'total_tokens': response.usage.total_tokens
            },
            raw_response=response
        )
    
    async def generate(self, prompt: str, model: Optional[str] = None, **kwargs) -> LLMResponse:
        """Async generation for OpenAI"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.generate_sync, prompt, model, **kwargs)
    
    def _get_model_config(self, model: str) -> Dict[str, Any]:
        """Get configuration for a specific model"""
        models = self.config.get('models', [])
        for m in models:
            if m['name'] == model:
                return m
        return {}


class GoogleLLM(BaseLLM):
    """Google Gemini LLM implementation"""
    
    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        super().__init__(config, logger)
        genai.configure(api_key=self.api_key)
    
    def generate_sync(self, prompt: str, model: Optional[str] = None, **kwargs) -> LLMResponse:
        """Synchronous generation for Google Gemini"""
        model_name = model or self.config.get('default_model', 'gemini-pro')
        
        # Get model-specific config
        model_config = self._get_model_config(model_name)
        
        generation_config = genai.GenerationConfig(
            max_output_tokens=kwargs.get('max_output_tokens', 
                                        model_config.get('max_output_tokens', 2048)),
            temperature=kwargs.get('temperature', model_config.get('temperature', 0.7)),
            top_p=kwargs.get('top_p', model_config.get('top_p', 0.8)),
            top_k=kwargs.get('top_k', model_config.get('top_k', 40)),
        )
        
        model = genai.GenerativeModel(model_name)
        
        response = self._retry_with_backoff(
            model.generate_content,
            prompt,
            generation_config=generation_config
        )
        
        return LLMResponse(
            content=response.text,
            model=model_name,
            provider=LLMProvider.GOOGLE.value,
            usage={
                'prompt_tokens': response.usage_metadata.prompt_token_count,
                'completion_tokens': response.usage_metadata.candidates_token_count,
                'total_tokens': response.usage_metadata.total_token_count
            },
            raw_response=response
        )
    
    async def generate(self, prompt: str, model: Optional[str] = None, **kwargs) -> LLMResponse:
        """Async generation for Google Gemini"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.generate_sync, prompt, model, **kwargs)
    
    def _get_model_config(self, model: str) -> Dict[str, Any]:
        """Get configuration for a specific model"""
        models = self.config.get('models', [])
        for m in models:
            if m['name'] == model:
                return m
        return {}


class AnthropicLLM(BaseLLM):
    """Anthropic Claude LLM implementation"""
    
    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        super().__init__(config, logger)
        self.client = anthropic.Anthropic(api_key=self.api_key)
        if self.base_url:
            self.client.base_url = self.base_url
    
    def generate_sync(self, prompt: str, model: Optional[str] = None, **kwargs) -> LLMResponse:
        """Synchronous generation for Anthropic Claude"""
        model = model or self.config.get('default_model', 'claude-3-opus-20240229')
        
        # Get model-specific config
        model_config = self._get_model_config(model)
        
        params = {
            'model': model,
            'max_tokens': kwargs.get('max_tokens', model_config.get('max_tokens', 4096)),
            'temperature': kwargs.get('temperature', model_config.get('temperature', 0.7)),
            'messages': [{"role": "user", "content": prompt}]
        }
        
        # Add optional parameters
        if 'top_p' in model_config:
            params['top_p'] = model_config['top_p']
        if 'top_k' in model_config:
            params['top_k'] = model_config['top_k']
        
        response = self._retry_with_backoff(
            self.client.messages.create,
            **params
        )
        
        return LLMResponse(
            content=response.content[0].text,
            model=model,
            provider=LLMProvider.ANTHROPIC.value,
            usage={
                'prompt_tokens': response.usage.input_tokens,
                'completion_tokens': response.usage.output_tokens,
                'total_tokens': response.usage.input_tokens + response.usage.output_tokens
            },
            raw_response=response
        )
    
    async def generate(self, prompt: str, model: Optional[str] = None, **kwargs) -> LLMResponse:
        """Async generation for Anthropic Claude"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.generate_sync, prompt, model, **kwargs)
    
    def _get_model_config(self, model: str) -> Dict[str, Any]:
        """Get configuration for a specific model"""
        models = self.config.get('models', [])
        for m in models:
            if m['name'] == model:
                return m
        return {}


class GrokLLM(BaseLLM):
    """Grok LLM implementation (X.AI)"""
    
    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        super().__init__(config, logger)
        # Grok uses OpenAI-compatible API
        self.client = openai.OpenAI(
            api_key=self.api_key,
            base_url=self.base_url or "https://api.x.ai/v1"
        )
    
    def generate_sync(self, prompt: str, model: Optional[str] = None, **kwargs) -> LLMResponse:
        """Synchronous generation for Grok"""
        model = model or self.config.get('default_model', 'grok-beta')
        
        # Get model-specific config
        model_config = self._get_model_config(model)
        
        params = {
            'model': model,
            'messages': [{"role": "user", "content": prompt}],
            'max_tokens': kwargs.get('max_tokens', model_config.get('max_tokens', 4096)),
            'temperature': kwargs.get('temperature', model_config.get('temperature', 0.7)),
            'top_p': kwargs.get('top_p', model_config.get('top_p', 1.0)),
        }
        
        response = self._retry_with_backoff(
            self.client.chat.completions.create,
            **params
        )
        
        return LLMResponse(
            content=response.choices[0].message.content,
            model=model,
            provider=LLMProvider.GROK.value,
            usage={
                'prompt_tokens': response.usage.prompt_tokens,
                'completion_tokens': response.usage.completion_tokens,
                'total_tokens': response.usage.total_tokens
            },
            raw_response=response
        )
    
    async def generate(self, prompt: str, model: Optional[str] = None, **kwargs) -> LLMResponse:
        """Async generation for Grok"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.generate_sync, prompt, model, **kwargs)
    
    def _get_model_config(self, model: str) -> Dict[str, Any]:
        """Get configuration for a specific model"""
        models = self.config.get('models', [])
        for m in models:
            if m['name'] == model:
                return m
        return {}


class LLMFactory:
    """Factory class for managing multiple LLM providers"""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = config_path
        self.config = self._load_config()
        self.logger = self._setup_logging()
        self.providers = {}
        self._initialize_providers()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration"""
        log_config = self.config.get('logging', {})
        log_level = getattr(logging, log_config.get('level', 'INFO'))
        
        logger = logging.getLogger('LLMFactory')
        logger.setLevel(log_level)
        
        # Clear existing handlers
        logger.handlers = []
        
        # Console handler
        if log_config.get('log_to_console', True):
            console_handler = logging.StreamHandler()
            console_handler.setLevel(log_level)
            console_format = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            console_handler.setFormatter(console_format)
            logger.addHandler(console_handler)
        
        # File handler
        if log_config.get('log_file'):
            file_handler = logging.FileHandler(log_config['log_file'])
            file_handler.setLevel(log_level)
            file_format = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            file_handler.setFormatter(file_format)
            logger.addHandler(file_handler)
        
        return logger
    
    def _initialize_providers(self):
        """Initialize enabled LLM providers"""
        provider_classes = {
            LLMProvider.OPENAI: OpenAILLM,
            LLMProvider.GOOGLE: GoogleLLM,
            LLMProvider.ANTHROPIC: AnthropicLLM,
            LLMProvider.GROK: GrokLLM
        }
        
        llm_providers = self.config.get('llm_providers', {})
        
        for provider_name, provider_config in llm_providers.items():
            if provider_config.get('enabled', False):
                try:
                    provider_enum = LLMProvider(provider_name)
                    provider_class = provider_classes[provider_enum]
                    self.providers[provider_enum] = provider_class(
                        provider_config, 
                        self.logger
                    )
                    self.logger.info(f"Initialized {provider_name} provider")
                except Exception as e:
                    self.logger.error(f"Failed to initialize {provider_name}: {e}")
    
    def get_provider(self, provider: Union[str, LLMProvider]) -> BaseLLM:
        """Get a specific LLM provider"""
        if isinstance(provider, str):
            provider = LLMProvider(provider)
        
        if provider not in self.providers:
            raise ValueError(f"Provider {provider.value} not available or not enabled")
        
        return self.providers[provider]
    
    def generate(self, 
                prompt: str, 
                provider: Union[str, LLMProvider] = None,
                model: Optional[str] = None,
                **kwargs) -> LLMResponse:
        """Generate a response using specified provider"""
        if provider is None:
            # Use first available provider
            if not self.providers:
                raise ValueError("No LLM providers are enabled")
            provider = list(self.providers.keys())[0]
        
        llm = self.get_provider(provider)
        
        # Log API call if configured
        if self.config.get('logging', {}).get('log_api_calls', True):
            self.logger.info(f"Calling {provider} with model {model or 'default'}")
        
        response = llm.generate_sync(prompt, model, **kwargs)
        
        # Log response if configured
        if self.config.get('logging', {}).get('log_responses', False):
            self.logger.debug(f"Response: {response.content[:100]}...")
        
        return response
    
    async def generate_async(self,
                            prompt: str,
                            provider: Union[str, LLMProvider] = None,
                            model: Optional[str] = None,
                            **kwargs) -> LLMResponse:
        """Asynchronously generate a response using specified provider"""
        if provider is None:
            if not self.providers:
                raise ValueError("No LLM providers are enabled")
            provider = list(self.providers.keys())[0]
        
        llm = self.get_provider(provider)
        
        if self.config.get('logging', {}).get('log_api_calls', True):
            self.logger.info(f"Async calling {provider} with model {model or 'default'}")
        
        response = await llm.generate(prompt, model, **kwargs)
        
        if self.config.get('logging', {}).get('log_responses', False):
            self.logger.debug(f"Response: {response.content[:100]}...")
        
        return response
    
    def compare_providers(self, 
                         prompt: str,
                         providers: List[Union[str, LLMProvider]] = None,
                         **kwargs) -> Dict[str, LLMResponse]:
        """Compare responses from multiple providers"""
        if providers is None:
            providers = list(self.providers.keys())
        
        results = {}
        for provider in providers:
            try:
                response = self.generate(prompt, provider, **kwargs)
                results[str(provider)] = response
            except Exception as e:
                self.logger.error(f"Error with {provider}: {e}")
                results[str(provider)] = None
        
        return results
    
    def list_available_providers(self) -> List[str]:
        """List all available providers"""
        return [p.value for p in self.providers.keys()]
    
    def list_models(self, provider: Union[str, LLMProvider]) -> List[str]:
        """List available models for a provider"""
        if isinstance(provider, str):
            provider = LLMProvider(provider)
        
        provider_config = self.config.get('llm_providers', {}).get(provider.value, {})
        models = provider_config.get('models', [])
        return [m['name'] for m in models]


# Example usage
if __name__ == "__main__":
    # Initialize the factory
    factory = LLMFactory("config.yaml")
    
    # Example 1: Simple generation with default provider
    response = factory.generate("What is the capital of France?")
    print(f"Response from {response.provider}: {response.content}")
    
    # Example 2: Specific provider and model
    response = factory.generate(
        "Explain quantum computing in simple terms",
        provider="openai",
        model="gpt-4",
        temperature=0.5
    )
    print(f"Response: {response.content[:200]}...")
    
    # Example 3: Compare providers
    results = factory.compare_providers(
        "Write a haiku about programming",
        providers=["openai", "anthropic", "google"]
    )
    for provider, response in results.items():
        if response:
            print(f"\n{provider}:\n{response.content}")
    
    # Example 4: Async usage
    async def async_example():
        response = await factory.generate_async(
            "What are the benefits of async programming?",
            provider="anthropic"
        )
        print(f"Async response: {response.content[:200]}...")
    
    # Run async example
    # asyncio.run(async_example())
    
    # Example 5: List available resources
    print("\nAvailable providers:", factory.list_available_providers())
    print("OpenAI models:", factory.list_models("openai"))