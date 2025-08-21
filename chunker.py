import os
import re
import json
import argparse
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from llm_factory import LLMFactory
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PaperChunker:
    """A class for chunking and categorizing research papers."""
    
    def __init__(self, llm_factory: Optional[LLMFactory] = None):
        """
        Initialize the chunker with an optional LLM factory.
        
        Args:
            llm_factory: LLMFactory instance for LLM operations
        """
        self.llm_factory = llm_factory
        self.section_categories = [
            "Abstract", "Introduction", "Problem Definition",
            "Methodology", "Related Work", "Experiment", "Conclusion",
            "References", "Acknowledgement", "Appendix"
        ]
    
    def extract_sections(self, md_file_path: str) -> Dict[str, str]:
        """
        Extract sections from a markdown file based on heading structure.
        
        Args:
            md_file_path: Path to the markdown file
            
        Returns:
            Dictionary mapping section titles to their content
        """
        section_dict = {}
        
        try:
            # Read the markdown content
            with open(md_file_path, 'r', encoding='utf-8') as f:
                md_content = f.read()
            
            with open('config.yaml', 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            pattern = config.get('chunker', {}).get('pattern', '')
            if not pattern:
                logger.error("No pattern found in config.yaml")
                raise ValueError("Missing pattern in config.yaml")
            matches = list(re.finditer(pattern, md_content, flags=re.MULTILINE))
            
            if not matches:
                logger.warning(f"No sections found in {md_file_path}")
                return {"Full Document": md_content}
            
            # Add dummy end to simplify slicing
            spans = [m.span() for m in matches] + [(len(md_content), len(md_content))]
            
            # Extract content under each heading
            for i in range(len(matches)):
                # Get the full heading line (including #'s)
                heading_line = matches[i].group(0).strip()
                # Get just the title text
                title = matches[i].group(2).strip()
                
                # Clean the title for use as a key
                clean_title = self._clean_section_title(heading_line)
                
                # Extract content between this heading and the next
                start = spans[i][1]  # Start after heading line
                end = spans[i+1][0]   # End before next heading
                content = md_content[start:end].strip()
                
                # Store with cleaned title as key
                section_dict[clean_title] = content
            
            logger.info(f"Extracted {len(section_dict)} sections from {md_file_path}")
            return section_dict
            
        except FileNotFoundError:
            logger.error(f"File not found: {md_file_path}")
            raise
        except Exception as e:
            logger.error(f"Error extracting sections: {e}")
            raise
    
    def _clean_section_title(self, title: str) -> str:
        """
        Clean section title for use as dictionary key.
        
        Args:
            title: Raw section title
            
        Returns:
            Cleaned title
        """
        # Remove problematic characters while preserving structure
        title = title.replace(':', '_')
        title = title.replace('/', '_')
        title = title.replace('\\', '_')
        title = re.sub(r'\s+', ' ', title)  # Normalize whitespace
        return title.strip()
    
    def categorize_sections(self, section_dict: Dict[str, str], 
                          output_path: Optional[str] = None) -> Dict[str, List[str]]:
        """
        Categorize sections using LLM.
        
        Args:
            section_dict: Dictionary of section titles and content
            output_path: Optional path to save categorization results
            
        Returns:
            Dictionary mapping categories to lists of section titles
        """
        if not self.llm_factory:
            logger.error("LLM Factory not initialized")
            raise ValueError("LLM Factory required for categorization")
        
        # Load prompts
        prompts = self._load_prompts()
        prompt_template = prompts.get('chunker', {}).get('section_labeling', '')
        
        if not prompt_template:
            logger.error("Section labeling prompt not found")
            raise ValueError("Missing section labeling prompt")
        
        # Prepare section titles for LLM
        section_titles = list(section_dict.keys())
        titles_str = '\n'.join(f"- {title}" for title in section_titles)
        
        # Generate prompt
        prompt = prompt_template.replace('[Title List]', titles_str)
        
        try:
            # Call LLM
            logger.info("Categorizing sections using LLM...")
            response = self.llm_factory.generate(
                prompt,
                model='gpt-4o-2024-08-06',
                max_tokens=10000,
                temperature=0.7
            )
            
            # Parse response
            response_json = json.loads(response.content)
            
            # Validate response structure
            categorized = self._validate_categorization(response_json, section_titles)
            
            # Save if output path provided
            if output_path:
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(categorized, f, ensure_ascii=False, indent=2)
                logger.info(f"Categorization saved to {output_path}")
            
            return categorized
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response as JSON: {e}")
            logger.debug(f"Response content: {response.content if 'response' in locals() else 'N/A'}")
            raise
        except Exception as e:
            logger.error(f"Error during categorization: {e}")
            raise
    
    def _load_prompts(self) -> Dict[str, Any]:
        """Load prompts from YAML file."""
        prompt_file = Path(__file__).parent / 'prompts.yaml'
        
        try:
            with open(prompt_file, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.warning(f"Error loading prompts: {e}")
            raise
    
    def _validate_categorization(self, categorized: Dict[str, List[str]], 
                                section_titles: List[str]) -> Dict[str, List[str]]:
        """
        Validate and clean the categorization results.
        
        Args:
            categorized: Raw categorization from LLM
            section_titles: Original section titles
            
        Returns:
            Validated categorization
        """
        # Ensure all categories exist
        validated = {}
        for category in self.section_categories:
            if category in categorized:
                # Ensure it's a list
                if isinstance(categorized[category], list):
                    validated[category] = categorized[category]
                else:
                    validated[category] = []
            else:
                validated[category] = []
        
        # Check for uncategorized sections
        categorized_titles = set()
        for titles in validated.values():
            categorized_titles.update(titles)
        
        uncategorized = set(section_titles) - categorized_titles
        if uncategorized:
            logger.warning(f"Uncategorized sections: {uncategorized}")
            # Add to most likely category (Methodology as default)
            validated["Methodology"].extend(list(uncategorized))
        
        return validated
    
    def process_paper(self, md_file_path: str, output_dir: Optional[str] = None) -> Tuple[Dict[str, str], Dict[str, List[str]]]:
        """
        Complete processing pipeline for a research paper.
        
        Args:
            md_file_path: Path to markdown file
            output_dir: Optional output directory for results
            
        Returns:
            Tuple of (section_dict, categorization)
        """
        # Extract sections
        section_dict = self.extract_sections(md_file_path)
        
        # Save sections if output directory provided
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            base_name = Path(md_file_path).stem
            sections_file = output_path / f"{base_name}_sections.json"
            
            with open(sections_file, 'w', encoding='utf-8') as f:
                json.dump(section_dict, f, ensure_ascii=False, indent=2)
            logger.info(f"Sections saved to {sections_file}")
        
        # Categorize sections if LLM is available
        categorization = None
        if self.llm_factory:
            categorization_file = None
            if output_dir:
                categorization_file = output_path / f"{base_name}_categorized.json"
            
            categorization = self.categorize_sections(section_dict, categorization_file)
        
        return section_dict, categorization


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description="Process research papers for chunking and categorization")
    parser.add_argument("--input", type=str, required=True,
                       help="Input markdown file or directory")
    parser.add_argument("--output", type=str, default="output",
                       help="Output directory for processed files")
    parser.add_argument("--use-llm", action="store_true",
                       help="Use LLM for section categorization")
    parser.add_argument("--extract-related", action="store_true",
                       help="Extract and combine Related Work sections")
    
    args = parser.parse_args()
    
    # Initialize LLM factory if needed
    llm_factory = None
    if args.use_llm:
        try:
            llm_factory = LLMFactory("config.yaml")
            logger.info("LLM Factory initialized")
        except Exception as e:
            logger.warning(f"Could not initialize LLM Factory: {e}")
    
    # Initialize chunker
    chunker = PaperChunker(llm_factory)
    
    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    
    if input_path.is_file():
        # Process single file
        logger.info(f"Processing single file: {input_path}")
        section_dict, categorization = chunker.process_paper(str(input_path), str(output_path))
        logger.info("Processing complete")
        
    elif input_path.is_dir():
        # Process all markdown files in directory
        md_files = list(input_path.glob("*.md"))
        logger.info(f"Found {len(md_files)} markdown files to process")
        
        for md_file in md_files:
            logger.info(f"Processing: {md_file}")
            try:
                chunker.process_paper(str(md_file), str(output_path))
            except Exception as e:
                logger.error(f"Error processing {md_file}: {e}")
        
        # Extract related work if requested
    
    else:
        logger.error(f"Input path does not exist: {input_path}")
        return 1
    
    return 0


if __name__ == "__main__":
    main()