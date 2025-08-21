from llm_factory import LLMFactory
import yaml
import json

from llm_factory import LLMFactory
import yaml
import json
import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import logging
from chunker import PaperChunker

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TaskTaxonomyGenerator:    
    def __init__(self, llm_factory: LLMFactory):
        self.llm_factory = llm_factory
        self.prompts = yaml.safe_load(open('prompts.yaml', 'r', encoding='utf-8'))
        self.chunker = PaperChunker(llm_factory)
        
    def extract_problem_definition(self, paper_path: str) -> Dict[str, Any]:
        """Extract problem definition from a single paper."""
        try:
            # First, extract and categorize sections
            section_dict, categorization = self.chunker.process_paper(paper_path)
            
            # Get problem definition sections
            problem_sections = []
            with open('config.yaml', 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            task_sections = config.get('taxonomy_generator', {}).get('task_sections', [])
            
            for t_section in task_sections:
                for section in categorization[t_section]:
                    problem_sections.append(section_dict[section])

            print (len(problem_sections))
            
            # Combine relevant sections
            paper_content = '\n\n'.join(problem_sections)  # Limit to avoid token limits
            
            # Get prompt and generate problem definition
            prompt = self.prompts['taxonomy_generator']['task_taxonomy']['extract_problem_definition']
            prompt = prompt.replace('[Paper Content]', paper_content)
            
            logger.info(f"Extracting problem definition from {Path(paper_path).stem}")
            response = self.llm_factory.generate(
                prompt,
                model='gpt-4.1',
                max_tokens=10000,
                temperature=0
            )
            
            # Parse YAML response
            import re
            yaml_match = re.search(r'```yaml\s*(.*?)\s*```', response.content, re.DOTALL)
            if yaml_match:
                yaml_content = yaml_match.group(1)
            else:
                # Try to find YAML content without backticks
                yaml_content = response.content
            
            problem_def = yaml.safe_load(yaml_content)
            return problem_def
            
        except Exception as e:
            logger.error(f"Error extracting problem definition from {paper_path}: {e}")
            return None
    
    def classify_aspects(self, problem_definitions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Classify problem definitions by aspects (input/output)."""
        try:
            # Format problem definitions for the prompt
            problems_text = yaml.dump(problem_definitions, default_flow_style=False)
            
            prompt = self.prompts['taxonomy_generator']['task_taxonomy']['aspect_classification']
            prompt = prompt.replace('[Problem Definitions]', problems_text)
            
            logger.info("Classifying problem aspects...")
            response = self.llm_factory.generate(
                prompt,
                model='gpt-4.1',
                max_tokens=10000,
                temperature=0
            )
            
            # Parse the markdown tables from response
            classification = response.content.strip('```json').strip('```')
            return classification
            
        except Exception as e:
            logger.error(f"Error classifying aspects: {e}")
            return None
    
    def generate_hierarchy(self, classification: Dict[str, Any]) -> Dict[str, Any]:
        """Generate hierarchical taxonomy from classifications."""
        try:
            # Format classification for prompt
            
            prompt = self.prompts['taxonomy_generator']['task_taxonomy']['taxonomy_generation']
            prompt = prompt.replace('[Classification Result]', classification)
            
            logger.info("Generating hierarchical taxonomy...")
            response = self.llm_factory.generate(
                prompt,
                model='gpt-4.1',
                max_tokens=10000,
                temperature=0
            )
            
            # Parse the hierarchical structure
            taxonomy = response.content.strip('```json').strip('```')
            return taxonomy
            
        except Exception as e:
            logger.error(f"Error generating hierarchy: {e}")
            return None
    
    def generate_taxonomy(self, papers_dir: str, output_path: Optional[str] = None) -> Dict[str, Any]:
        """Generate complete taxonomy from a directory of papers."""
        papers_path = Path(papers_dir)
        
        # Find all markdown files
        md_files = list(papers_path.glob("*.md"))
        logger.info(f"Found {len(md_files)} papers to process")
        
        # Extract problem definitions from all papers
        problem_definitions = []
        for md_file in md_files:
            logger.info(f"Processing {md_file.name}")
            problem_def = self.extract_problem_definition(str(md_file))
            if problem_def:
                problem_definitions.append(problem_def)
        
        if not problem_definitions:
            logger.error("No problem definitions extracted")
            return None
        
        logger.info(f"Extracted {len(problem_definitions)} problem definitions")
        
        # Classify aspects
        classification = self.classify_aspects(problem_definitions)
        if not classification:
            logger.error("Failed to classify aspects")
            return None
        
        self.classification = classification
        # Generate hierarchy
        taxonomy = self.generate_hierarchy(classification)
        if not taxonomy:
            logger.error("Failed to generate hierarchy")
            return None
        
        # Add metadata
        result = {
            "metadata": {
                "num_papers": len(problem_definitions),
                "papers_processed": [p.get('paper_id', 'Unknown') for p in problem_definitions],
                "generator": "TaskTaxonomyGenerator"
            },
            "problem_definitions": problem_definitions,
            "aspect_classification": classification,
            "taxonomy": taxonomy
        }
        
        # Save if output path provided
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            logger.info(f"Taxonomy saved to {output_path}")
         
        return result
    
    def _parse_classification_result(self, content: str) -> Dict[str, List[Dict[str, Any]]]:
        """Parse results from classification response."""
        return content.strip('```json').strip('```')
        
    
    def _parse_hierarchy(self, content: str) -> Dict[str, Any]:
        """Parse hierarchical taxonomy from response."""
        import re
        
        taxonomy = {
            "root": "ROOT",
            "tasks": []
        }
        
        # Find task definitions in the content
        task_pattern = r'TASK:L\d+:(\w+).*?\n.*?Input:\s*(.*?)\n.*?Output:\s*(.*?)\n.*?Explanation:\s*(.*?)\n.*?Papers:\s*(.*?)(?:\n|$)'
        
        matches = re.finditer(task_pattern, content, re.MULTILINE | re.DOTALL)
        
        for match in matches:
            task = {
                "name": f"TASK:{match.group(1)}",
                "input": match.group(2).strip(),
                "output": match.group(3).strip(),
                "explanation": match.group(4).strip(),
                "papers": [p.strip() for p in match.group(5).split(',')]
            }
            taxonomy["tasks"].append(task)
        
        # Try to extract hierarchy structure
        hierarchy_lines = []
        for line in content.split('\n'):
            if line.strip().startswith('-'):
                hierarchy_lines.append(line)
        
        if hierarchy_lines:
            taxonomy["hierarchy_text"] = '\n'.join(hierarchy_lines)
        
        return taxonomy


class MethodTaxonomyGenerator:
    def __init__(self, llm_factory: LLMFactory):
        self.llm_factory = llm_factory
        self.prompts = yaml.safe_load(open('prompts.yaml', 'r', encoding='utf-8'))
        self.chunker = PaperChunker(llm_factory)
        
    def generate_taxonomy(self, papers_dir: str, output_path: Optional[str] = None) -> Dict[str, Any]:
        """Generate method taxonomy from papers."""
        # This is a placeholder for method taxonomy generation
        # The implementation would be similar to TaskTaxonomyGenerator
        # but focused on extracting and classifying methods/algorithms
        
        logger.info("Method taxonomy generation not yet implemented")
        return {
            "status": "not_implemented",
            "message": "Method taxonomy generation will be implemented based on the prompts structure"
        }




class MethodTaxonomyGenerator:
    def __init__(self, llm_factory: LLMFactory):
        self.llm_factory = llm_factory
        self.prompts = yaml.safe_load(open('prompts.yaml', 'r', encoding='utf-8'))
        
    def generate_taxonomy(self, paper_path: str):
        pass

