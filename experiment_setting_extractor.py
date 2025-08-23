"""
Experiment Setting Extractor
Extracts datasets, baselines, and metrics from research papers
"""

import os
import re
import json
import argparse
import hashlib
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import yaml
from collections import defaultdict

# Import your existing modules
from llm_factory import LLMFactory
from chunker import PaperChunker

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class Entity:
    """Base class for extracted entities"""
    name: str
    canonical_name: Optional[str] = None
    aliases: List[str] = field(default_factory=list)
    description: Optional[str] = None
    usage_frequency: int = 0
    supporting_papers: List[Dict[str, Any]] = field(default_factory=list)
    id: Optional[str] = None
    
    def __post_init__(self):
        if not self.canonical_name:
            self.canonical_name = self.canonicalize_name(self.name)
        if not self.id:
            self.id = self.generate_id()
    
    def canonicalize_name(self, name: str) -> str:
        """Convert name to canonical form"""
        # Lowercase, strip, normalize separators
        canonical = name.lower().strip()
        canonical = re.sub(r'[^\w\s-]', '', canonical)
        canonical = re.sub(r'[-_\s]+', '', canonical)
        
        # Expand common acronyms
        acronym_map = {
            'cifar10': 'cifar10',
            'cifar-10': 'cifar10',
            'mnist': 'mnist',
            'imagenet': 'imagenet',
            'coco': 'coco',
            'glue': 'glue',
            'squad': 'squad',
            'bert': 'bert',
            'gpt': 'gpt',
            'lstm': 'lstm',
            'cnn': 'cnn',
            'rnn': 'rnn',
            'xgb': 'xgboost',
            'rf': 'randomforest',
            'svm': 'svm',
            'lr': 'logisticregression',
        }
        
        for pattern, replacement in acronym_map.items():
            if canonical == pattern.lower().replace('-', '').replace('_', ''):
                canonical = replacement
                break
        
        return canonical
    
    def generate_id(self) -> str:
        """Generate stable ID from entity type and canonical name"""
        entity_type = self.__class__.__name__.lower()
        id_string = f"{entity_type}:{self.canonical_name}"
        return hashlib.sha1(id_string.encode()).hexdigest()[:16]
    
    def add_support(self, paper_id: str, section: str, quote: str, confidence: float):
        """Add supporting evidence from a paper"""
        self.supporting_papers.append({
            "paper_id": paper_id,
            "section": section,
            "quote": quote[:200],  # Limit quote length
            "confidence": round(confidence, 2)
        })
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, excluding None values"""
        data = asdict(self)
        return {k: v for k, v in data.items() if v is not None and v != [] and v != {}}


@dataclass
class Dataset(Entity):
    """Dataset entity"""
    profile: Optional[Dict[str, Any]] = None
    original_paper: Optional[Dict[str, Any]] = None
    url: Optional[str] = None
    task_type: List[str] = field(default_factory=list)


@dataclass
class Baseline(Entity):
    """Baseline entity"""
    original_paper: Optional[Dict[str, Any]] = None
    category: Optional[str] = None


@dataclass
class Metric(Entity):
    """Metric entity"""
    formulation: Optional[str] = None
    task_scope: List[str] = field(default_factory=list)


class ExperimentSettingExtractor:
    """Main extractor class"""
    
    def __init__(self, llm_factory: LLMFactory, config: Dict[str, Any]):
        self.llm_factory = llm_factory
        self.config = config
        self.chunker = PaperChunker(llm_factory)
        self.cache = {}
        
        # Entity stores
        self.datasets: Dict[str, Dataset] = {}
        self.baselines: Dict[str, Baseline] = {}
        self.metrics: Dict[str, Metric] = {}
        
        # Patterns for detection
        self.patterns = {
            'dataset': [
                r'dataset[s]?\b', r'benchmark[s]?\b', r'corpus\b', r'corpora\b',
                r'we evaluate on\b', r'we use .{0,50} dataset', r'training set',
                r'test set', r'validation set', r'data from\b'
            ],
            'baseline': [
                r'baseline[s]?\b', r'compared? (?:to|with|against)\b',
                r'we compare (?:to|with|against)\b', r'prior work[s]?\b',
                r'state-of-the-art\b', r'SOTA\b', r'existing method[s]?\b',
                r'previous approach[es]?\b'
            ],
            'metric': [
                r'metric[s]?\b', r'evaluate using\b', r'we report\b',
                r'measure[s]?\b', r'score[s]?\b', r'performance metric[s]?\b',
                r'accuracy\b', r'precision\b', r'recall\b', r'F1\b', r'F-score\b',
                r'BLEU\b', r'ROUGE\b', r'perplexity\b', r'AUC\b', r'MAP@\d+',
                r'NDCG@\d+', r'MRR\b', r'RMSE\b', r'MAE\b', r'MSE\b'
            ]
        }
        
        # Common dataset names for better recognition
        self.known_datasets = {
            'cifar10', 'cifar100', 'mnist', 'imagenet', 'coco', 'pascal voc',
            'squad', 'glue', 'superglue', 'wmt', 'penn treebank', 'ptb',
            'wikipedia', 'wikimedia', 'reddit', 'twitter', 'facebook',
            'yelp', 'amazon', 'imdb', 'stanford', 'cornell', 'microsoft',
            'google', 'yahoo', 'netflix', 'movielens', 'lastfm', 'spotify'
        }
        
        # Common baseline methods
        self.known_baselines = {
            'bert', 'gpt', 'gpt2', 'gpt3', 'gpt4', 't5', 'bart', 'roberta',
            'xlnet', 'albert', 'electra', 'deberta', 'transformer', 'lstm',
            'gru', 'rnn', 'cnn', 'resnet', 'vgg', 'alexnet', 'inception',
            'mobilenet', 'efficientnet', 'vit', 'swin', 'xgboost', 'lightgbm',
            'catboost', 'random forest', 'svm', 'logistic regression',
            'naive bayes', 'knn', 'kmeans', 'dbscan', 'pca', 'lda'
        }
        
        # Common metrics
        self.known_metrics = {
            'accuracy', 'precision', 'recall', 'f1', 'f-score', 'f-measure',
            'auc', 'roc-auc', 'pr-auc', 'map', 'mrr', 'ndcg', 'err',
            'bleu', 'rouge', 'rouge-1', 'rouge-2', 'rouge-l', 'meteor',
            'cider', 'spice', 'perplexity', 'ppl', 'wer', 'cer',
            'rmse', 'mae', 'mse', 'mape', 'r2', 'r-squared',
            'iou', 'dice', 'jaccard', 'hamming', 'cosine'
        }
    
    def extract_from_paper(self, paper_path: str) -> Tuple[List[Dataset], List[Baseline], List[Metric]]:
        """Extract entities from a single paper"""
        paper_id = Path(paper_path).stem
        logger.info(f"Processing paper: {paper_id}")
        
        try:
            # Extract and categorize sections
            section_dict, categorization = self.chunker.process_paper(paper_path)
            
            # Prioritize experimental sections
            priority_sections = ['Experiment', 'Methodology', 'Related Work', 'Abstract']
            relevant_sections = []
            
            for section_type in priority_sections:
                if section_type in categorization:
                    for section_title in categorization[section_type]:
                        if section_title in section_dict:
                            relevant_sections.append({
                                'title': section_title,
                                'type': section_type,
                                'content': section_dict[section_title]
                            })
            
            # Extract entities from each section
            paper_datasets = []
            paper_baselines = []
            paper_metrics = []
            
            for section in relevant_sections:
                # Use pattern matching to find candidate snippets
                snippets = self.find_candidate_snippets(section['content'], section['type'])
                
                for snippet_type, snippet_text in snippets:
                    # Extract entities using LLM
                    entities = self.extract_entities_with_llm(
                        snippet_text, snippet_type, paper_id, section['title']
                    )
                    
                    # Categorize extracted entities
                    for entity in entities:
                        if entity['type'] == 'dataset':
                            paper_datasets.append(self.create_dataset(entity, paper_id, section['title']))
                        elif entity['type'] == 'baseline':
                            paper_baselines.append(self.create_baseline(entity, paper_id, section['title']))
                        elif entity['type'] == 'metric':
                            paper_metrics.append(self.create_metric(entity, paper_id, section['title']))
            
            return paper_datasets, paper_baselines, paper_metrics
            
        except Exception as e:
            logger.error(f"Error processing paper {paper_id}: {e}")
            return [], [], []
    
    def find_candidate_snippets(self, text: str, section_type: str) -> List[Tuple[str, str]]:
        """Find text snippets likely to contain entities"""
        snippets = []
        sentences = re.split(r'[.!?]\s+', text)
        
        for i, sentence in enumerate(sentences):
            sentence_lower = sentence.lower()
            
            # Check for dataset patterns
            if any(re.search(pattern, sentence_lower) for pattern in self.patterns['dataset']):
                context = self.get_context(sentences, i, window=2)
                snippets.append(('dataset', context))
            
            # Check for baseline patterns
            if any(re.search(pattern, sentence_lower) for pattern in self.patterns['baseline']):
                context = self.get_context(sentences, i, window=2)
                snippets.append(('baseline', context))
            
            # Check for metric patterns
            if any(re.search(pattern, sentence_lower) for pattern in self.patterns['metric']):
                context = self.get_context(sentences, i, window=1)
                snippets.append(('metric', context))
            
            # Check for known entity names
            for dataset in self.known_datasets:
                if dataset in sentence_lower:
                    context = self.get_context(sentences, i, window=1)
                    snippets.append(('dataset', context))
                    break
            
            for baseline in self.known_baselines:
                if baseline in sentence_lower:
                    context = self.get_context(sentences, i, window=1)
                    snippets.append(('baseline', context))
                    break
            
            for metric in self.known_metrics:
                if metric in sentence_lower:
                    context = self.get_context(sentences, i, window=1)
                    snippets.append(('metric', context))
                    break
        
        return snippets
    
    def get_context(self, sentences: List[str], index: int, window: int = 1) -> str:
        """Get context around a sentence"""
        start = max(0, index - window)
        end = min(len(sentences), index + window + 1)
        return ' '.join(sentences[start:end])
    
    def extract_entities_with_llm(self, snippet: str, entity_type: str, 
                                  paper_id: str, section: str) -> List[Dict[str, Any]]:
        """Extract entities from snippet using LLM"""
        # Check cache
        cache_key = hashlib.md5(f"{snippet}:{entity_type}".encode()).hexdigest()
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # Prepare prompt
        prompt = self.create_extraction_prompt(snippet, entity_type)
        
        try:
            response = self.llm_factory.generate(
                prompt,
                model=self.config.get('model', 'gpt-4'),
                temperature=0,
                max_tokens=1000
            )
            
            # Parse response
            entities = self.parse_llm_response(response.content, entity_type)
            
            # Add confidence and provenance
            for entity in entities:
                entity['confidence'] = self.calculate_confidence(entity, snippet)
                entity['quote'] = snippet[:200]
                entity['paper_id'] = paper_id
                entity['section'] = section
            
            # Filter by confidence
            min_confidence = self.config.get('min_confidence', 0.6)
            entities = [e for e in entities if e['confidence'] >= min_confidence]
            
            # Cache result
            self.cache[cache_key] = entities
            
            return entities
            
        except Exception as e:
            logger.warning(f"LLM extraction failed: {e}")
            return []
    
    def create_extraction_prompt(self, snippet: str, entity_type: str) -> str:
        """Create LLM prompt for entity extraction"""
        type_map = {
            'dataset': 'DATASET (data collection used for training/evaluation)',
            'baseline': 'BASELINE (method/model being compared against)',
            'metric': 'METRIC (evaluation measure or score)'
        }
        
        prompt = f"""Extract {type_map.get(entity_type, entity_type)} entities from the following text snippet.

Text: {snippet}

For each entity found, provide:
- name: The exact name as mentioned
- description: Brief description if available
- aliases: Any alternative names mentioned

Additional fields for DATASET:
- num_instances: Number of samples if mentioned
- modality: Type of data (text/image/audio/video/graph/table)
- task_type: Classification/regression/etc.

Additional fields for BASELINE:
- category: Type of method (heuristic/classical ML/deep learning/LLM/hybrid)

Additional fields for METRIC:
- formulation: Mathematical formula if provided
- task_scope: What tasks this metric applies to

Return as JSON array. Only include fields you can confidently extract from the text.
If no entities found, return empty array [].

Example response:
[{{"name": "CIFAR-10", "description": "Image classification dataset", "modality": "image", "num_instances": 60000}}]
"""
        return prompt
    
    def parse_llm_response(self, response: str, entity_type: str) -> List[Dict[str, Any]]:
        """Parse LLM response to extract entities"""
        try:
            # Try to extract JSON from response
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if json_match:
                entities = json.loads(json_match.group())
            else:
                entities = json.loads(response)
            
            # Add entity type
            for entity in entities:
                entity['type'] = entity_type
            
            return entities
            
        except Exception as e:
            logger.debug(f"Failed to parse LLM response: {e}")
            return []
    
    def calculate_confidence(self, entity: Dict[str, Any], snippet: str) -> float:
        """Calculate confidence score for extracted entity"""
        confidence = 0.5  # Base confidence
        
        # Check if name appears in snippet
        if entity.get('name', '').lower() in snippet.lower():
            confidence += 0.2
        
        # Check for additional fields
        if entity.get('description'):
            confidence += 0.1
        if entity.get('aliases'):
            confidence += 0.05
        
        # Type-specific checks
        if entity['type'] == 'dataset':
            if entity.get('num_instances') or entity.get('modality'):
                confidence += 0.1
        elif entity['type'] == 'baseline':
            if entity.get('category'):
                confidence += 0.1
        elif entity['type'] == 'metric':
            if entity.get('formulation'):
                confidence += 0.15
        
        return min(confidence, 1.0)
    
    def create_dataset(self, entity_data: Dict[str, Any], paper_id: str, section: str) -> Dataset:
        """Create Dataset object from extracted data"""
        dataset = Dataset(name=entity_data['name'])
        
        # Add optional fields
        if 'description' in entity_data:
            dataset.description = entity_data['description']
        
        if 'aliases' in entity_data:
            dataset.aliases = entity_data['aliases']
        
        # Build profile
        profile = {}
        if 'num_instances' in entity_data:
            profile['num_instances'] = entity_data['num_instances']
        if 'modality' in entity_data:
            profile['modality'] = entity_data['modality']
        if profile:
            dataset.profile = profile
        
        if 'task_type' in entity_data:
            dataset.task_type = [entity_data['task_type']] if isinstance(entity_data['task_type'], str) else entity_data['task_type']
        
        # Add support
        dataset.add_support(
            paper_id, section,
            entity_data.get('quote', ''),
            entity_data.get('confidence', 0.5)
        )
        
        return dataset
    
    def create_baseline(self, entity_data: Dict[str, Any], paper_id: str, section: str) -> Baseline:
        """Create Baseline object from extracted data"""
        baseline = Baseline(name=entity_data['name'])
        
        if 'description' in entity_data:
            baseline.description = entity_data['description']
        
        if 'aliases' in entity_data:
            baseline.aliases = entity_data['aliases']
        
        if 'category' in entity_data:
            baseline.category = entity_data['category']
        
        baseline.add_support(
            paper_id, section,
            entity_data.get('quote', ''),
            entity_data.get('confidence', 0.5)
        )
        
        return baseline
    
    def create_metric(self, entity_data: Dict[str, Any], paper_id: str, section: str) -> Metric:
        """Create Metric object from extracted data"""
        metric = Metric(name=entity_data['name'])
        
        if 'description' in entity_data:
            metric.description = entity_data['description']
        
        if 'aliases' in entity_data:
            metric.aliases = entity_data['aliases']
        
        if 'formulation' in entity_data:
            metric.formulation = entity_data['formulation']
        
        if 'task_scope' in entity_data:
            metric.task_scope = [entity_data['task_scope']] if isinstance(entity_data['task_scope'], str) else entity_data['task_scope']
        
        metric.add_support(
            paper_id, section,
            entity_data.get('quote', ''),
            entity_data.get('confidence', 0.5)
        )
        
        return metric
    
    def merge_entities(self, new_entities: List[Entity], entity_store: Dict[str, Entity]):
        """Merge new entities into existing store"""
        for entity in new_entities:
            key = entity.canonical_name
            
            if key in entity_store:
                # Merge with existing entity
                existing = entity_store[key]
                
                # Merge aliases
                for alias in entity.aliases:
                    if alias not in existing.aliases and alias != existing.name:
                        existing.aliases.append(alias)
                
                # Merge supporting papers
                existing.supporting_papers.extend(entity.supporting_papers)
                
                # Update description if better
                if not existing.description and entity.description:
                    existing.description = entity.description
                
                # Type-specific merging
                if isinstance(entity, Dataset) and isinstance(existing, Dataset):
                    if entity.profile and not existing.profile:
                        existing.profile = entity.profile
                    if entity.url and not existing.url:
                        existing.url = entity.url
                    if entity.task_type and not existing.task_type:
                        existing.task_type = entity.task_type
                
                elif isinstance(entity, Baseline) and isinstance(existing, Baseline):
                    if entity.category and not existing.category:
                        existing.category = entity.category
                
                elif isinstance(entity, Metric) and isinstance(existing, Metric):
                    if entity.formulation and not existing.formulation:
                        existing.formulation = entity.formulation
                    if entity.task_scope and not existing.task_scope:
                        existing.task_scope = entity.task_scope
            else:
                # Add new entity
                entity_store[key] = entity
    
    def process_papers(self, papers_dir: str, max_workers: int = 4):
        """Process all papers in directory"""
        papers_path = Path(papers_dir)
        paper_files = list(papers_path.glob("*.md")) + list(papers_path.glob("*.txt"))
        
        logger.info(f"Found {len(paper_files)} papers to process")
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self.extract_from_paper, str(paper)): paper
                for paper in paper_files
            }
            
            for future in as_completed(futures):
                paper = futures[future]
                try:
                    datasets, baselines, metrics = future.result()
                    
                    # Merge into stores
                    self.merge_entities(datasets, self.datasets)
                    self.merge_entities(baselines, self.baselines)
                    self.merge_entities(metrics, self.metrics)
                    
                    logger.info(f"Processed {paper.name}: {len(datasets)} datasets, "
                              f"{len(baselines)} baselines, {len(metrics)} metrics")
                    
                except Exception as e:
                    logger.error(f"Failed to process {paper}: {e}")
    
    def calculate_usage_frequencies(self):
        """Calculate usage frequency for all entities"""
        for entity_store in [self.datasets, self.baselines, self.metrics]:
            for entity in entity_store.values():
                # Count unique papers
                paper_ids = set(sp['paper_id'] for sp in entity.supporting_papers)
                entity.usage_frequency = len(paper_ids)
    
    def save_results(self, output_dir: str):
        """Save extracted entities to JSON files"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Calculate frequencies before saving
        self.calculate_usage_frequencies()
        
        # Save datasets
        datasets_file = output_path / "datasets.json"
        with open(datasets_file, 'w', encoding='utf-8') as f:
            datasets_list = [entity.to_dict() for entity in self.datasets.values()]
            json.dump(datasets_list, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved {len(self.datasets)} datasets to {datasets_file}")
        
        # Save baselines
        baselines_file = output_path / "baselines.json"
        with open(baselines_file, 'w', encoding='utf-8') as f:
            baselines_list = [entity.to_dict() for entity in self.baselines.values()]
            json.dump(baselines_list, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved {len(self.baselines)} baselines to {baselines_file}")
        
        # Save metrics
        metrics_file = output_path / "metrics.json"
        with open(metrics_file, 'w', encoding='utf-8') as f:
            metrics_list = [entity.to_dict() for entity in self.metrics.values()]
            json.dump(metrics_list, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved {len(self.metrics)} metrics to {metrics_file}")
        
        # Print summary statistics
        self.print_summary()
    
    def print_summary(self):
        """Print extraction summary"""
        logger.info("\n" + "="*50)
        logger.info("EXTRACTION SUMMARY")
        logger.info("="*50)
        
        # Dataset statistics
        logger.info(f"\nDatasets: {len(self.datasets)} unique")
        top_datasets = sorted(self.datasets.values(), 
                             key=lambda x: x.usage_frequency, reverse=True)[:5]
        for ds in top_datasets:
            logger.info(f"  - {ds.name}: {ds.usage_frequency} papers")
        
        # Baseline statistics
        logger.info(f"\nBaselines: {len(self.baselines)} unique")
        top_baselines = sorted(self.baselines.values(),
                              key=lambda x: x.usage_frequency, reverse=True)[:5]
        for bl in top_baselines:
            logger.info(f"  - {bl.name}: {bl.usage_frequency} papers")
        
        # Metric statistics
        logger.info(f"\nMetrics: {len(self.metrics)} unique")
        top_metrics = sorted(self.metrics.values(),
                           key=lambda x: x.usage_frequency, reverse=True)[:5]
        for mt in top_metrics:
            logger.info(f"  - {mt.name}: {mt.usage_frequency} papers")


def main():
    """Main CLI function"""
    parser = argparse.ArgumentParser(
        description="Extract datasets, baselines, and metrics from research papers"
    )
    parser.add_argument("--input_dir", type=str, required=True,
                       help="Directory containing papers")
    parser.add_argument("--parsed_json", type=str,
                       help="Optional pre-parsed JSON file")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Output directory for JSON files")
    parser.add_argument("--topic", type=str,
                       help="Research topic/domain")
    parser.add_argument("--llm_provider", type=str, default="openai",
                       help="LLM provider to use")
    parser.add_argument("--model", type=str, default="gpt-4",
                       help="Model to use for extraction")
    parser.add_argument("--cache_dir", type=str, default=".cache",
                       help="Cache directory for LLM calls")
    parser.add_argument("--min_confidence", type=float, default=0.6,
                       help="Minimum confidence threshold (0-1)")
    parser.add_argument("--max_workers", type=int, default=4,
                       help="Maximum parallel workers")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Initialize LLM factory
    try:
        llm_factory = LLMFactory("config.yaml")
        logger.info("LLM Factory initialized")
    except Exception as e:
        logger.error(f"Failed to initialize LLM Factory: {e}")
        return 1
    
    # Create configuration
    config = {
        'model': args.model,
        'min_confidence': args.min_confidence,
        'topic': args.topic,
        'cache_dir': args.cache_dir
    }
    
    # Initialize extractor
    extractor = ExperimentSettingExtractor(llm_factory, config)
    
    # Process papers
    logger.info(f"Processing papers from {args.input_dir}")
    extractor.process_papers(args.input_dir, args.max_workers)
    
    # Save results
    extractor.save_results(args.output_dir)
    
    logger.info("Extraction complete!")
    return 0


if __name__ == "__main__":
    exit(main())