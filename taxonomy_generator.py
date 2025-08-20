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
            if 'Problem Definition' in categorization:
                for section_title in categorization['Problem Definition']:
                    if section_title in section_dict:
                        problem_sections.append(section_dict[section_title])
            
            # If no explicit problem definition, look in introduction and abstract
            if not problem_sections:
                for category in ['Introduction', 'Abstract']:
                    if category in categorization:
                        for section_title in categorization[category]:
                            if section_title in section_dict:
                                problem_sections.append(section_dict[section_title])
                                break
                        if problem_sections:
                            break
            
            # Combine relevant sections
            paper_content = '\n\n'.join(problem_sections)  # Limit to avoid token limits
            
            # Get prompt and generate problem definition
            prompt = self.prompts['taxonomy_generator']['task_taxonomy']['extract_problem_definition']
            prompt = prompt.replace('[Paper Content]', paper_content)
            
            logger.info(f"Extracting problem definition from {Path(paper_path).stem}")
            response = self.llm_factory.generate(
                prompt,
                model='gpt-4.1',
                max_tokens=1000,
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
                model='gpt-4o-mini',
                max_tokens=3000,
                temperature=0
            )
            
            # Parse the markdown tables from response
            classification = self._parse_classification_tables(response.content)
            return classification
            
        except Exception as e:
            logger.error(f"Error classifying aspects: {e}")
            return None
    
    def generate_hierarchy(self, classification: Dict[str, Any]) -> Dict[str, Any]:
        """Generate hierarchical taxonomy from classifications."""
        try:
            # Format classification for prompt
            classification_text = self._format_classification_for_prompt(classification)
            
            prompt = self.prompts['taxonomy_generator']['task_taxonomy']['taxonomy_generation']
            prompt = prompt.replace('[Classification Tables]', classification_text)
            
            logger.info("Generating hierarchical taxonomy...")
            response = self.llm_factory.generate(
                prompt,
                model='gpt-4o-mini',
                max_tokens=3000,
                temperature=0
            )
            
            # Parse the hierarchical structure
            taxonomy = self._parse_hierarchy(response.content)
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
    
    def _parse_classification_tables(self, content: str) -> Dict[str, List[Dict[str, Any]]]:
        """Parse markdown tables from classification response."""
        classification = {
            "input_classification": [],
            "output_classification": []
        }
        
        import re
        
        # Find Input Classification table
        input_match = re.search(r'Input Classification.*?\n\n(.*?)\n\n', content, re.DOTALL)
        if input_match:
            input_table = input_match.group(1)
            classification["input_classification"] = self._parse_markdown_table(input_table)
        
        # Find Output Classification table
        output_match = re.search(r'Output Classification.*?\n\n(.*?)(?:\n\n|$)', content, re.DOTALL)
        if output_match:
            output_table = output_match.group(1)
            classification["output_classification"] = self._parse_markdown_table(output_table)
        
        return classification
    
    def _parse_markdown_table(self, table_text: str) -> List[Dict[str, Any]]:
        """Parse a markdown table into list of dictionaries."""
        lines = table_text.strip().split('\n')
        if len(lines) < 3:  # Need header, separator, and at least one data row
            return []
        
        # Parse header
        headers = [h.strip() for h in lines[0].split('|')[1:-1]]
        
        # Parse data rows (skip separator line)
        rows = []
        for line in lines[2:]:
            if line.strip():
                values = [v.strip() for v in line.split('|')[1:-1]]
                if len(values) == len(headers):
                    row = dict(zip(headers, values))
                    # Parse papers list
                    if 'Papers' in row:
                        row['Papers'] = [p.strip() for p in row['Papers'].split(',')]
                    rows.append(row)
        
        return rows
    
    def _format_classification_for_prompt(self, classification: Dict[str, Any]) -> str:
        """Format classification back to markdown for prompt."""
        result = []
        
        # Format Input Classification
        if classification.get("input_classification"):
            result.append("### Input Classification\n")
            result.append("| Input Class | Class Description | Papers |")
            result.append("|-------------|-------------------|--------|")
            for item in classification["input_classification"]:
                papers = ", ".join(item.get('Papers', []))
                result.append(f"| {item.get('Input Class', '')} | {item.get('Class Description', '')} | {papers} |")
        
        result.append("")
        
        # Format Output Classification
        if classification.get("output_classification"):
            result.append("### Output Classification\n")
            result.append("| Output Class | Class Description | Papers |")
            result.append("|--------------|-------------------|--------|")
            for item in classification["output_classification"]:
                papers = ", ".join(item.get('Papers', []))
                result.append(f"| {item.get('Output Class', '')} | {item.get('Class Description', '')} | {papers} |")
        
        return "\n".join(result)
    
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

