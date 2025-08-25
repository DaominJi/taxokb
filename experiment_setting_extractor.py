from llm_factory import LLMFactory
import yaml
import json
import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import logging
from chunker import PaperChunker
import re
import pandas as pd
from pdf_to_md_converter import extract_title_from_md

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ExperimentSettingExtractor:
    """Extract experiment settings from research papers."""
    
    def __init__(self, llm_factory: LLMFactory):
        """
        Initialize the experiment setting extractor.
        
        Args:
            llm_factory: LLMFactory instance for LLM operations
        """
        self.llm_factory = llm_factory
        self.prompts = yaml.safe_load(open('prompts.yaml', 'r', encoding='utf-8'))
        self.chunker = PaperChunker(llm_factory)
        self.paper_register = pd.read_csv("data/paper_register.csv")
        
    def extract_experiment_summary(self, paper_path: str) -> Dict[str, Any]:
        """
        Extract experiment summary from a single paper.
        
        Args:
            paper_path: Path to the markdown file
            
        Returns:
            Dictionary containing experiment settings
        """
        try:
            # First, extract and categorize sections
            section_dict, categorization = self.chunker.process_paper(paper_path)

            with open(paper_path, 'r', encoding='utf-8') as f:
                paper_content = f.read()
            title = extract_title_from_md(paper_content)
            paper_id = self.paper_register['id'][self.paper_register['title'] == title].values[0]
            
            # Get experiment sections
            experiment_sections = []
            if 'Experiment' in categorization:
                for section_title in categorization['Experiment']:
                    if section_title in section_dict:
                        experiment_sections.append(section_dict[section_title])
            
            if not experiment_sections:
                logger.warning(f"No experiment sections found in {paper_path}")
                return None
            
            # Combine experiment sections
            experiment_text = '\n\n'.join(experiment_sections)
            
            # Get prompt and generate experiment summary
            prompt = self.prompts['experiment_setting_extractor']['extract_experiment_summary']
            prompt = prompt.replace('{paper_id}', paper_id)
            prompt = prompt.replace('{paper_title}', title)
            prompt = prompt.replace('{experiment_text}', experiment_text)
            
            logger.info(f"Extracting experiment settings from {Path(paper_path).stem}")
            response = self.llm_factory.generate(
                prompt,
                model='gpt-4.1',
                max_tokens=10000,
                temperature=0.7
            )
            
            # Parse JSON response
            try:
                # Clean the response to extract JSON
                json_str = response.content
                # Remove markdown code blocks if present
                json_str = re.sub(r'```json\s*', '', json_str)
                json_str = re.sub(r'```\s*$', '', json_str)
                json_str = json_str.strip()
                
                experiment_data = json.loads(json_str)
                return experiment_data
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON response: {e}")
                logger.debug(f"Response content: {response.content}")
                return None
                
        except Exception as e:
            logger.error(f"Error extracting experiment settings from {paper_path}: {e}")
            return None
    
    def merge_baselines(self, all_baselines: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Merge baseline entries from multiple papers into a unified list.
        
        Args:
            all_baselines: List of baseline entries from all papers
            
        Returns:
            Unified list of canonical baselines
        """
        try:
            if not all_baselines:
                return []
            
            # Format baselines for prompt
            baselines_input = {"baselines": all_baselines}
            
            prompt = self.prompts['experiment_setting_extractor']['merge_baselines']
            prompt = prompt.replace('[Baseline_input]', json.dumps(baselines_input, indent=2))
            
            logger.info("Merging baseline methods...")
            response = self.llm_factory.generate(
                prompt,
                model='gpt-4.1',
                max_tokens=10000,
                temperature=0.7
            )
            
            # Parse JSON response
            json_str = response.content
            json_str = re.sub(r'```json\s*', '', json_str)
            json_str = re.sub(r'```\s*$', '', json_str)
            json_str = json_str.strip()
            
            result = json.loads(json_str)
            return result.get('baselines', [])
            
        except Exception as e:
            logger.error(f"Error merging baselines: {e}")
            return []
    
    def merge_datasets(self, all_datasets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Merge dataset entries from multiple papers into a unified list.
        
        Args:
            all_datasets: List of dataset entries from all papers
            
        Returns:
            Unified list of canonical datasets
        """
        try:
            if not all_datasets:
                return []
            
            # Format datasets for prompt
            datasets_input = {"datasets": all_datasets}
            
            prompt = self.prompts['experiment_setting_extractor']['merge_datasets']
            prompt = prompt.replace('[Dataset_input]', json.dumps(datasets_input, indent=2))
            
            logger.info("Merging datasets...")
            response = self.llm_factory.generate(
                prompt,
                model='gpt-4.1',
                max_tokens=10000,
                temperature=0.7
            )
            
            # Parse JSON response
            json_str = response.content
            json_str = re.sub(r'```json\s*', '', json_str)
            json_str = re.sub(r'```\s*$', '', json_str)
            json_str = json_str.strip()
            
            result = json.loads(json_str)
            return result.get('datasets', [])
            
        except Exception as e:
            logger.error(f"Error merging datasets: {e}")
            return []
    
    def merge_metrics(self, all_metrics: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Merge metric entries from multiple papers into a unified list.
        
        Args:
            all_metrics: List of metric entries from all papers
            
        Returns:
            Unified list of canonical metrics
        """
        try:
            if not all_metrics:
                return []
            
            # Format metrics for prompt
            metrics_input = {"metrics": all_metrics}
            
            prompt = self.prompts['experiment_setting_extractor']['merge_metrics']
            prompt = prompt.replace('[Metric_input]', json.dumps(metrics_input, indent=2))
            
            logger.info("Merging metrics...")
            response = self.llm_factory.generate(
                prompt,
                model='gpt-4.1',
                max_tokens=10000,
                temperature=0.7
            )
            
            # Parse JSON response
            json_str = response.content
            json_str = re.sub(r'```json\s*', '', json_str)
            json_str = re.sub(r'```\s*$', '', json_str)
            json_str = json_str.strip()
            
            result = json.loads(json_str)
            return result.get('metrics', [])
            
        except Exception as e:
            logger.error(f"Error merging metrics: {e}")
            return []
    
    def extract_all_experiments(self, papers_dir: str, output_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Extract experiment settings from all papers in a directory.
        
        Args:
            papers_dir: Directory containing markdown files
            output_path: Optional path to save the results
            
        Returns:
            Dictionary containing all experiment data
        """
        papers_path = Path(papers_dir)
        
        # Find all markdown files
        md_files = list(papers_path.glob("*.md"))
        logger.info(f"Found {len(md_files)} papers to process")
        
        if not md_files:
            logger.error("No markdown files found in directory")
            return None
        
        # Extract experiment settings from each paper
        all_experiments = []
        all_datasets = []
        all_metrics = []
        all_baselines = []
        
        for md_file in md_files:
            logger.info(f"Processing {md_file.name}")
            experiment_data = self.extract_experiment_summary(str(md_file))
            
            if experiment_data:
                all_experiments.append(experiment_data)
                
                # Collect all datasets, metrics, and baselines
                if 'datasets' in experiment_data:
                    dataset_info = {}
                    dataset_info['paper_id'] = experiment_data['paper_id']
                    dataset_info['datasets'] = experiment_data['datasets']
                    all_datasets.append(dataset_info)
                if 'metrics' in experiment_data:
                    metric_info = {}
                    metric_info['paper_id'] = experiment_data['paper_id']
                    metric_info['metrics'] = experiment_data['metrics']
                    all_metrics.append(metric_info)
                if 'baselines' in experiment_data:
                    baseline_info = {}
                    baseline_info['paper_id'] = experiment_data['paper_id']
                    baseline_info['baselines'] = experiment_data['baselines']
                    all_baselines.append(baseline_info)
        
        with open('output/dataset_input.json', 'w', encoding='utf-8') as f:
            json.dump(all_datasets, f, ensure_ascii=False, indent=2)
        with open('output/metric_input.json', 'w', encoding='utf-8') as f:
            json.dump(all_metrics, f, ensure_ascii=False, indent=2)
        with open('output/baseline_input.json', 'w', encoding='utf-8') as f:
            json.dump(all_baselines, f, ensure_ascii=False, indent=2)
        
        if not all_experiments:
            logger.error("No experiment data extracted from any paper")
            return None
        
        logger.info(f"Extracted experiment data from {len(all_experiments)} papers")
        
        # Merge and canonicalize entities
        logger.info("Merging extracted entities...")
        canonical_datasets = self.merge_datasets(all_datasets)
        canonical_metrics = self.merge_metrics(all_metrics)
        canonical_baselines = self.merge_baselines(all_baselines)
        
        # Create final result
        result = {
            "metadata": {
                "num_papers": len(all_experiments),
                "papers_processed": [exp.get('paper_id', 'Unknown') for exp in all_experiments],
                "generator": "ExperimentSettingExtractor",
                "total_datasets": len(canonical_datasets),
                "total_metrics": len(canonical_metrics),
                "total_baselines": len(canonical_baselines)
            },
            "paper_experiments": all_experiments,
            "canonical_datasets": canonical_datasets,
            "canonical_metrics": canonical_metrics,
            "canonical_baselines": canonical_baselines,
            "summary_statistics": {
                "avg_datasets_per_paper": len(all_datasets) / len(all_experiments) if all_experiments else 0,
                "avg_metrics_per_paper": len(all_metrics) / len(all_experiments) if all_experiments else 0,
                "avg_baselines_per_paper": len(all_baselines) / len(all_experiments) if all_experiments else 0,
                "unique_datasets": len(canonical_datasets),
                "unique_metrics": len(canonical_metrics),
                "unique_baselines": len(canonical_baselines)
            }
        }
        
        # Save if output path provided
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            logger.info(f"Experiment settings saved to {output_path}")
        
        return result
    
    def generate_experiment_report(self, experiment_data: Dict[str, Any]) -> str:
        """
        Generate a human-readable report from experiment data.
        
        Args:
            experiment_data: Extracted experiment data
            
        Returns:
            Markdown-formatted report
        """
        report = []
        report.append("# Experiment Settings Analysis Report\n")
        
        # Metadata
        meta = experiment_data.get('metadata', {})
        report.append(f"## Summary")
        report.append(f"- **Papers Analyzed**: {meta.get('num_papers', 0)}")
        report.append(f"- **Total Unique Datasets**: {meta.get('total_datasets', 0)}")
        report.append(f"- **Total Unique Metrics**: {meta.get('total_metrics', 0)}")
        report.append(f"- **Total Unique Baselines**: {meta.get('total_baselines', 0)}\n")
        
        # Canonical Datasets
        report.append("## Canonical Datasets")
        datasets = experiment_data.get('canonical_datasets', [])
        for dataset in datasets:
            report.append(f"\n### {dataset.get('canonical_name', 'Unknown')}")
            if 'aliases' in dataset:
                report.append(f"**Aliases**: {', '.join(dataset['aliases'])}")
            if 'description' in dataset:
                report.append(f"**Description**: {dataset['description']}")
            if 'task_type' in dataset:
                report.append(f"**Task Type**: {', '.join(dataset['task_type']) if isinstance(dataset['task_type'], list) else dataset['task_type']}")
            if 'usage_frequency' in dataset:
                report.append(f"**Usage Frequency**: {dataset['usage_frequency']} papers")
        
        # Canonical Metrics
        report.append("\n## Canonical Metrics")
        metrics = experiment_data.get('canonical_metrics', [])
        for metric in metrics:
            report.append(f"\n### {metric.get('canonical_name', 'Unknown')}")
            if 'aliases' in metric:
                report.append(f"**Aliases**: {', '.join(metric['aliases'])}")
            if 'description' in metric:
                report.append(f"**Description**: {metric['description']}")
            if 'formulation' in metric:
                report.append(f"**Formulation**: `{metric['formulation']}`")
            if 'usage_frequency' in metric:
                report.append(f"**Usage Frequency**: {metric['usage_frequency']} papers")
        
        # Canonical Baselines
        report.append("\n## Canonical Baselines")
        baselines = experiment_data.get('canonical_baselines', [])
        for baseline in baselines:
            report.append(f"\n### {baseline.get('canonical_name', 'Unknown')}")
            if 'aliases' in baseline:
                report.append(f"**Aliases**: {', '.join(baseline['aliases'])}")
            if 'description' in baseline:
                report.append(f"**Description**: {baseline['description']}")
            if 'category' in baseline:
                report.append(f"**Category**: {baseline['category']}")
            if 'usage_frequency' in baseline:
                report.append(f"**Usage Frequency**: {baseline['usage_frequency']} papers")
        
        # Statistics
        stats = experiment_data.get('summary_statistics', {})
        report.append("\n## Statistics")
        report.append(f"- **Average Datasets per Paper**: {stats.get('avg_datasets_per_paper', 0):.2f}")
        report.append(f"- **Average Metrics per Paper**: {stats.get('avg_metrics_per_paper', 0):.2f}")
        report.append(f"- **Average Baselines per Paper**: {stats.get('avg_baselines_per_paper', 0):.2f}")
        
        return '\n'.join(report)


# Example usage and testing
if __name__ == "__main__":
    # Initialize the extractor
    llm_factory = LLMFactory("config.yaml")
    extractor = ExperimentSettingExtractor(llm_factory)
    
    # Test with a single paper
    # test_paper = "test/paper1.md"  # Adjust path as needed
    # if Path(test_paper).exists():
    #     logger.info(f"Testing with single paper: {test_paper}")
    #     single_result = extractor.extract_experiment_summary(test_paper)
    #     if single_result:
    #         print("\nSingle Paper Result:")
    #         print(json.dumps(single_result, indent=2))
    
    # Process all papers in a directory
    papers_dir = "test"  # Adjust path as needed
    if Path(papers_dir).exists():
        logger.info(f"\nProcessing all papers in {papers_dir}")
        all_results = extractor.extract_all_experiments(
            papers_dir,
            output_path="experiment_settings.json"
        )
        
        if all_results:
            # Generate and save report
            report = extractor.generate_experiment_report(all_results)
            with open("experiment_report.md", "w", encoding="utf-8") as f:
                f.write(report)
            logger.info("Report saved to experiment_report.md")
            
            # Print summary
            print("\n" + "="*50)
            print("Experiment Extraction Summary")
            print("="*50)
            meta = all_results['metadata']
            print(f"Papers processed: {meta['num_papers']}")
            print(f"Unique datasets found: {meta['total_datasets']}")
            print(f"Unique metrics found: {meta['total_metrics']}")
            print(f"Unique baselines found: {meta['total_baselines']}")