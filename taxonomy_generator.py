from llm_factory import LLMFactory
import yaml
import json
import pandas as pd
from llm_factory import LLMFactory
import yaml
import json
import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import logging
from chunker import PaperChunker
from pdf_to_md_converter import extract_title_from_md

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TaskTaxonomyGenerator:    
    def __init__(self, llm_factory: LLMFactory):
        self.llm_factory = llm_factory
        self.prompts = yaml.safe_load(open('prompts.yaml', 'r', encoding='utf-8'))
        self.chunker = PaperChunker(llm_factory)
        self.paper_register = pd.read_csv('data/paper_register.csv')

    def extract_problem_definition(self, paper_path: str) -> Dict[str, Any]:
        """Extract problem definition from a single paper."""
        try:
            # First, extract and categorize sections
            section_dict, categorization = self.chunker.process_paper(paper_path)
            with open(paper_path, 'r', encoding='utf-8') as f:
                paper_content = f.read()
            title = extract_title_from_md(paper_content)
            id = self.paper_register['id'][self.paper_register['title'] == title].values[0]
            
            # Get problem definition sections
            problem_sections = []
            with open('config.yaml', 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            task_sections = config.get('taxonomy_generator', {}).get('task_sections', [])
            
            for t_section in task_sections:
                for section in categorization[t_section]:
                    problem_sections.append(section_dict[section])

            #print (len(problem_sections))
            
            # Combine relevant sections
            paper_content = '\n\n'.join(problem_sections)  # Limit to avoid token limits
            
            # Get prompt and generate problem definition
            prompt = self.prompts['taxonomy_generator']['task_taxonomy']['extract_problem_definition']
            prompt = prompt.replace('[Paper Content]', paper_content)
            prompt = prompt.replace('[Paper ID]', id)
            prompt = prompt.replace('[Paper Title]', title)

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
        self.paper_register = pd.read_csv('data/paper_register.csv')
        
    def extract_method_summary(self, paper_path: str) -> Dict[str, Any]:
        """Extract method summary from a single paper."""
        try:
            # First, extract and categorize sections
            section_dict, categorization = self.chunker.process_paper(paper_path)
            with open(paper_path, 'r', encoding='utf-8') as f:
                paper_content = f.read()
            title = extract_title_from_md(paper_content)
            id = self.paper_register['id'][self.paper_register['title'] == title].values[0]
            
            # Get methodology sections
            method_sections = []
            with open('config.yaml', 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            # Get sections that contain methodology information
            methodology_sections = config.get('taxonomy_generator', {}).get('methodology_sections', [])
            
            for m_section in methodology_sections:
                if m_section in categorization:
                    for section in categorization[m_section]:
                        if section in section_dict:
                            method_sections.append(section_dict[section])
            
            if not method_sections:
                logger.warning(f"No methodology sections found in {paper_path}")
                return None
            
            # Combine relevant sections
            section_text = '\n\n'.join(method_sections)  
            
            # Get prompt and generate method summary
            prompt = self.prompts['taxonomy_generator']['method_taxonomy']['extract_method_summary']
            prompt = prompt.replace('{{method_section_text}}', section_text)
            
            logger.info(f"Extracting method summary from {Path(paper_path).stem}")
            response = self.llm_factory.generate(
                prompt,
                model='gpt-4.1',
                max_tokens=10000,
                temperature=0.7
            )
            
            # Parse and structure the response
            paper_name = Path(paper_path).stem
            method_summary = {
                "paper_id": id,
                "methodology_summary": response.content
            }
            
            return method_summary
            
        except Exception as e:
            logger.error(f"Error extracting method summary from {paper_path}: {e}")
            return None
    
    def extract_pros_and_cons(self, method_summaries: Dict[str, Any]) -> str:
        """Extract pros and cons from method summaries."""
        try:
            # Format method summaries for the prompt
            methods_text = json.dumps(method_summaries, indent=2)
            
            prompt = self.prompts['taxonomy_generator']['method_taxonomy']['extract_pros_and_cons']
            prompt = prompt.replace('{{method_summaries}}', methods_text)
            
            logger.info("Extracting pros and cons from methods...")
            response = self.llm_factory.generate(
                prompt,
                model='gpt-4.1',
                max_tokens=30000,
                temperature=0.7
            )
            
            return response.content
            
        except Exception as e:
            logger.error(f"Error extracting pros and cons: {e}")
            return None
    
    def generate_method_taxonomy(self, method_summaries: Dict[str, Any], 
                                pros_cons_summary: str):
        """Generate hierarchical taxonomy of methods."""
        try:
            # Format inputs for prompt
            prompt = self.prompts['taxonomy_generator']['method_taxonomy']['taxonomy_generation']
            
            # Replace placeholders in prompt
            prompt = prompt.replace('{method_summaries}', json.dumps(method_summaries, indent=2))
            prompt = prompt.replace('{pros_cons_summary}', pros_cons_summary)
            
            logger.info("Generating method taxonomy...")
            response = self.llm_factory.generate(
                prompt,
                model='gpt-4.1',
                max_tokens=10000,
                temperature=0.7
            )
            
            # Parse JSON response
            taxonomy_json = response.content.strip('```json').strip('```').strip()
            taxonomy = json.loads(taxonomy_json)
            
            return taxonomy
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse taxonomy JSON: {e}")
            logger.debug(f"Response: {response.content if 'response' in locals() else 'N/A'}")
            return None
        except Exception as e:
            logger.error(f"Error generating method taxonomy: {e}")
            return None
    
    def generate_taxonomy(self, papers_dir: str, output_path: Optional[str] = None) -> Dict[str, Any]:
        """Generate complete method taxonomy from a directory of papers."""
        papers_path = Path(papers_dir)
        
        # Find all markdown files
        md_files = list(papers_path.glob("*.md"))
        logger.info(f"Found {len(md_files)} papers to process")
        
        if not md_files:
            logger.error("No markdown files found in directory")
            return None
        
        # Step 1: Extract method summaries from all papers
        method_summaries = {}
        for md_file in md_files:
            logger.info(f"Processing {md_file.name}")
            method_summary = self.extract_method_summary(str(md_file))
            if method_summary:
                paper_name = md_file.stem
                method_summaries[f"{paper_name}.md"] = method_summary
        
        if not method_summaries:
            logger.error("No method summaries extracted")
            return None
        
        logger.info(f"Extracted {len(method_summaries)} method summaries")
        
        pros_cons_summary = self.extract_pros_and_cons(method_summaries)
        if not pros_cons_summary:
            logger.warning("Failed to extract pros and cons")
            pros_cons_summary = ""
        
        
        # Step 5: Generate taxonomy
        taxonomy = self.generate_method_taxonomy(
            method_summaries,
            pros_cons_summary
        )
        
        if not taxonomy:
            logger.error("Failed to generate taxonomy")
            return None
        
        # Create final result
        result = {
            "metadata": {
                "num_papers": len(method_summaries),
                "papers_processed": list(method_summaries.keys()),
                "generator": "MethodTaxonomyGenerator"
            },
            "method_summaries": method_summaries,
            "pros_cons_analysis": pros_cons_summary,
            "introductions_extracted": len(introductions),
            "related_works_extracted": len(related_works),
            "taxonomy": taxonomy
        }
        
        # Save if output path provided
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            logger.info(f"Method taxonomy saved to {output_path}")
        
        return result

