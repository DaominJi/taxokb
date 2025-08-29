"""
Reference Taxonomy Builder - Generate reference taxonomies for research topics
"""

from llm_factory import LLMFactory
import yaml
import json
import os
from typing import Dict, Any, Optional
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ReferenceTaxonomyBuilder:
    """
    Build reference taxonomies for research topics to guide the taxonomy generation process.
    """
    
    def __init__(self, llm_factory: Optional[LLMFactory] = None):
        """
        Initialize the reference taxonomy builder.
        
        Args:
            llm_factory: LLMFactory instance for LLM operations
        """
        if llm_factory is None:
            self.llm_factory = LLMFactory("config.yaml")
        else:
            self.llm_factory = llm_factory
        
        # Load prompts
        with open('prompts.yaml', 'r', encoding='utf-8') as f:
            self.prompts = yaml.safe_load(f)
    
    def generate_method_taxonomy(self, topic: str, output_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate a reference method taxonomy for a given topic.
        
        Args:
            topic: The research topic to generate taxonomy for
            output_path: Optional path to save the generated taxonomy
            
        Returns:
            Dictionary containing the hierarchical method taxonomy
        """
        try:
            # Get the method taxonomy prompt template
            prompt_template = self.prompts['reference_taxonomy_builder']['method_taxonomy']
            
            # Replace the topic placeholder
            prompt = prompt_template.replace('{{topic}}', topic)
            
            logger.info(f"Generating method taxonomy for topic: {topic}")
            
            # Generate the taxonomy using LLM
            response = self.llm_factory.generate(
                prompt,
                model='gpt-4o-2024-08-06',
                max_tokens=10000,
                temperature=0.7
            )
            
            # Parse the JSON response
            try:
                # Clean the response to extract JSON
                json_str = response.content
                # Remove markdown code blocks if present
                json_str = json_str.replace('```json', '').replace('```', '')
                json_str = json_str.strip()
                
                taxonomy = json.loads(json_str)
                
                # Wrap in a structured format
                result = {
                    "topic": topic,
                    "type": "method_taxonomy",
                    "taxonomy": taxonomy
                }
                
                # Save if output path provided
                if output_path:
                    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
                    with open(output_path, 'w', encoding='utf-8') as f:
                        json.dump(result, f, ensure_ascii=False, indent=2)
                    logger.info(f"Method taxonomy saved to {output_path}")
                
                return result
                
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON response: {e}")
                logger.debug(f"Response content: {response.content}")
                return None
                
        except Exception as e:
            logger.error(f"Error generating method taxonomy: {e}")
            return None
    
    def generate_task_taxonomy(self, topic: str, output_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate a reference task taxonomy for a given topic.
        
        Args:
            topic: The research topic to generate taxonomy for
            output_path: Optional path to save the generated taxonomy
            
        Returns:
            Dictionary containing the hierarchical task taxonomy
        """
        try:
            # Get the task taxonomy prompt template
            prompt_template = self.prompts['reference_taxonomy_builder']['task_taxonomy']
            
            # Replace the topic placeholder
            prompt = prompt_template.replace('{{topic}}', topic)
            
            logger.info(f"Generating task taxonomy for topic: {topic}")
            
            # Generate the taxonomy using LLM
            response = self.llm_factory.generate(
                prompt,
                model='gpt-4o-2024-08-06',
                max_tokens=10000,
                temperature=0.7
            )
            
            # Parse the JSON response
            try:
                # Clean the response to extract JSON
                json_str = response.content
                # Remove markdown code blocks if present
                json_str = json_str.replace('```json', '').replace('```', '')
                json_str = json_str.strip()
                
                taxonomy = json.loads(json_str)
                
                # Wrap in a structured format
                result = {
                    "topic": topic,
                    "type": "task_taxonomy",
                    "taxonomy": taxonomy
                }
                
                # Save if output path provided
                if output_path:
                    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
                    with open(output_path, 'w', encoding='utf-8') as f:
                        json.dump(result, f, ensure_ascii=False, indent=2)
                    logger.info(f"Task taxonomy saved to {output_path}")
                
                return result
                
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON response: {e}")
                logger.debug(f"Response content: {response.content}")
                return None
                
        except Exception as e:
            logger.error(f"Error generating task taxonomy: {e}")
            return None

def main():
    """
    Main function for command-line usage.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate reference taxonomies for research topics")
    parser.add_argument("topic", type=str, help="Research topic (e.g., 'entity resolution')")
    parser.add_argument("--type", choices=['method', 'task', 'both'], default='both',
                       help="Type of taxonomy to generate")
    parser.add_argument("--output", type=str, default="output/reference_taxonomies",
                       help="Output directory for generated taxonomies")
    
    args = parser.parse_args()
    
    # Initialize builder
    builder = ReferenceTaxonomyBuilder()
    
    # Generate taxonomies based on type
    topic_slug = args.topic.lower().replace(' ', '_')
    
    
    result = builder.generate_method_taxonomy(
        args.topic, 
        os.path.join(args.output, f"{topic_slug}_method_taxonomy.json")
    )
        
    result = builder.generate_task_taxonomy(
        args.topic,
        os.path.join(args.output, f"{topic_slug}_task_taxonomy.json")
    )
    
    if result:
        print(f"\n✅ Successfully generated {args.type} taxonomy for '{args.topic}'")
        print(f"📁 Output saved to: {args.output}")
    else:
        print(f"\n❌ Failed to generate taxonomy for '{args.topic}'")
        return 1
    
    return 0


if __name__ == "__main__":
    import sys
    
    # Quick test mode - set to True to run example
    TEST_MODE = True
    
    if TEST_MODE:
        # Example usage
        builder = ReferenceTaxonomyBuilder()
        
        # Generate reference taxonomies for entity resolution
        topic = "entity resolution"
        
        print(f"Generating reference taxonomies for: {topic}")
        print("-" * 50)
        
        result = builder.generate_method_taxonomy(
            "entity resolution", 
            os.path.join("output", f"entity_resolution_method_taxonomy.json")
        )
            
        result = builder.generate_task_taxonomy(
            "entity resolution",
            os.path.join("output", f"entity_resolution_task_taxonomy.json")
        )
        
        if result:
            print("\n✅ Reference taxonomies generated successfully!")
            print(f"Method taxonomy nodes: {json.dumps(result['method_taxonomy']['taxonomy']['name'], indent=2)}")
            print(f"Task taxonomy nodes: {json.dumps(result['task_taxonomy']['taxonomy']['name'], indent=2)}")
    else:
        # Run main CLI
        sys.exit(main())