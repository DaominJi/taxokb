"""
Example usage of the LLM Factory
"""

import asyncio
import os
from llm_factory import LLMFactory, LLMProvider

def basic_usage():
    """Basic usage examples"""
    
    # Set environment variables (or use .env file)
    # os.environ['OPENAI_API_KEY'] = 'your-openai-key'
    # os.environ['GOOGLE_API_KEY'] = 'your-google-key'
    # os.environ['ANTHROPIC_API_KEY'] = 'your-anthropic-key'
    # os.environ['GROK_API_KEY'] = 'your-grok-key'
    
    # Initialize factory
    factory = LLMFactory("config.yaml")
    
    print("=" * 50)
    print("LLM Factory Examples")
    print("=" * 50)
    
    # 1. List available providers
    print("\n1. Available Providers:")
    providers = factory.list_available_providers()
    for provider in providers:
        print(f"   - {provider}")
        models = factory.list_models(provider)
        for model in models:
            print(f"     • {model}")
    
    # 2. Simple generation with default settings
    print("\n2. Simple Generation (Default Provider):")
    try:
        response = factory.generate("What is the meaning of life in one sentence?")
        print(f"   Provider: {response.provider}")
        print(f"   Model: {response.model}")
        print(f"   Response: {response.content}")
        print(f"   Tokens used: {response.usage}")
    except Exception as e:
        print(f"   Error: {e}")
    
    # 3. Specific provider with custom parameters
    print("\n3. Custom Provider & Parameters:")
    try:
        response = factory.generate(
            prompt="Write a creative tagline for a coffee shop",
            provider="openai",
            model="gpt-3.5-turbo",
            temperature=0.9,
            max_tokens=50
        )
        print(f"   Response: {response.content}")
    except Exception as e:
        print(f"   Error: {e}")
    
    # 4. Compare multiple providers
    print("\n4. Provider Comparison:")
    prompt = "Explain machine learning in exactly 2 sentences."
    results = factory.compare_providers(
        prompt=prompt,
        providers=["openai", "google", "anthropic"],
        temperature=0.3
    )
    
    for provider, response in results.items():
        if response:
            print(f"\n   {provider}:")
            print(f"   {response.content}")
            print(f"   (Tokens: {response.usage['total_tokens']})")
        else:
            print(f"\n   {provider}: Failed to generate response")


async def async_usage():
    """Asynchronous usage examples"""
    
    factory = LLMFactory("config.yaml")
    
    print("\n" + "=" * 50)
    print("Async Examples")
    print("=" * 50)
    
    # 1. Single async call
    print("\n1. Single Async Call:")
    try:
        response = await factory.generate_async(
            "What are the advantages of async programming?",
            provider="openai"
        )
        print(f"   Response: {response.content[:150]}...")
    except Exception as e:
        print(f"   Error: {e}")
    
    # 2. Parallel async calls
    print("\n2. Parallel Async Calls:")
    prompts = [
        "Define artificial intelligence",
        "Define machine learning",
        "Define deep learning"
    ]
    
    tasks = [
        factory.generate_async(prompt, provider="openai")
        for prompt in prompts
    ]
    
    try:
        responses = await asyncio.gather(*tasks)
        for prompt, response in zip(prompts, responses):
            print(f"\n   Prompt: {prompt}")
            print(f"   Response: {response.content[:100]}...")
    except Exception as e:
        print(f"   Error: {e}")


def advanced_usage():
    """Advanced usage patterns"""
    
    factory = LLMFactory("config.yaml")
    
    print("\n" + "=" * 50)
    print("Advanced Usage Patterns")
    print("=" * 50)
    
    # 1. Conversation with context (using message history)
    print("\n1. Multi-turn Conversation:")
    
    # Note: For multi-turn conversations, you might want to extend
    # the factory to handle message history. Here's a simple approach:
    
    conversation = [
        "Hello! I'm learning about space exploration.",
        "What was the first satellite launched into space?",
        "When was it launched?",
        "What country launched it?"
    ]
    
    context = ""
    for turn, message in enumerate(conversation, 1):
        # Build context from previous messages
        full_prompt = f"{context}\nUser: {message}\nAssistant:"
        
        try:
            response = factory.generate(
                full_prompt,
                provider="openai",
                temperature=0.7,
                max_tokens=150
            )
            print(f"\n   Turn {turn}:")
            print(f"   User: {message}")
            print(f"   Assistant: {response.content}")
            
            # Update context for next turn
            context += f"\nUser: {message}\nAssistant: {response.content}"
        except Exception as e:
            print(f"   Error in turn {turn}: {e}")
            break
    
    # 2. Structured output generation
    print("\n2. Structured Output (JSON):")
    
    json_prompt = """Generate a JSON object for a book with the following fields:
    - title (string)
    - author (string)
    - year (number)
    - genres (array of strings)
    - rating (number, 1-5)
    
    Make it about a science fiction book. Return only valid JSON."""
    
    try:
        response = factory.generate(
            json_prompt,
            provider="openai",
            model="gpt-3.5-turbo",
            temperature=0.3
        )
        print(f"   Generated JSON:\n   {response.content}")
        
        # Validate JSON
        import json
        parsed = json.loads(response.content)
        print(f"   Valid JSON: ✓")
        print(f"   Book Title: {parsed.get('title', 'N/A')}")
    except json.JSONDecodeError:
        print(f"   Invalid JSON generated")
    except Exception as e:
        print(f"   Error: {e}")
    
    # 3. Custom retry logic for critical operations
    print("\n3. Custom Retry Logic:")
    
    def generate_with_validation(factory, prompt, validator_func, max_attempts=3):
        """Generate with custom validation and retry"""
        for attempt in range(max_attempts):
            try:
                response = factory.generate(prompt, temperature=0.3)
                if validator_func(response.content):
                    return response
                print(f"   Attempt {attempt + 1}: Validation failed, retrying...")
            except Exception as e:
                print(f"   Attempt {attempt + 1}: Error - {e}")
        return None
    
    # Example: Generate a number between 1 and 10
    def validate_number(content):
        try:
            num = int(content.strip())
            return 1 <= num <= 10
        except:
            return False
    
    number_prompt = "Generate only a single random number between 1 and 10. Return only the number, nothing else."
    
    result = generate_with_validation(factory, number_prompt, validate_number)
    if result:
        print(f"   Successfully generated: {result.content}")
    else:
        print(f"   Failed to generate valid number after retries")


def cost_tracking_example():
    """Example of tracking API costs"""
    
    factory = LLMFactory("config.yaml")
    
    print("\n" + "=" * 50)
    print("Cost Tracking Example")
    print("=" * 50)
    
    # Approximate costs per 1K tokens (as of 2024)
    COST_PER_1K_TOKENS = {
        "gpt-4": {"input": 0.03, "output": 0.06},
        "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
        "claude-3-opus-20240229": {"input": 0.015, "output": 0.075},
        "gemini-pro": {"input": 0.00025, "output": 0.0005}
    }
    
    total_cost = 0
    
    prompts = [
        "Write a short poem about coding",
        "Explain quantum computing",
        "What is the weather like today?"
    ]
    
    for prompt in prompts:
        try:
            response = factory.generate(prompt, provider="openai", model="gpt-3.5-turbo")
            
            # Calculate cost
            model_costs = COST_PER_1K_TOKENS.get(response.model, {"input": 0, "output": 0})
            input_cost = (response.usage['prompt_tokens'] / 1000) * model_costs['input']
            output_cost = (response.usage['completion_tokens'] / 1000) * model_costs['output']
            request_cost = input_cost + output_cost
            total_cost += request_cost
            
            print(f"\n   Prompt: {prompt[:50]}...")
            print(f"   Model: {response.model}")
            print(f"   Tokens: {response.usage['total_tokens']}")
            print(f"   Cost: ${request_cost:.6f}")
            
        except Exception as e:
            print(f"   Error: {e}")
    
    print(f"\n   Total Cost: ${total_cost:.6f}")


if __name__ == "__main__":
    # Run basic examples
    basic_usage()
    
    # Run advanced examples
    advanced_usage()
    
    # Run cost tracking
    cost_tracking_example()
    
    # Run async examples
    print("\nRunning async examples...")
    asyncio.run(async_usage())
    
    print("\n" + "=" * 50)
    print("Examples completed!")
    print("=" * 50)