from llm_factory import LLMFactory
import yaml
import json

class TaskTaxonomyGenerator:    
    def __init__(self, llm_factory: LLMFactory):
        self.llm_factory = llm_factory
        self.prompts = yaml.safe_load(open('prompts.yaml', 'r', encoding='utf-8'))
        
    def generate_taxonomy(self, paper_path: str):
        pass

class MethodTaxonomyGenerator:
    def __init__(self, llm_factory: LLMFactory):
        self.llm_factory = llm_factory
        self.prompts = yaml.safe_load(open('prompts.yaml', 'r', encoding='utf-8'))
        
    def generate_taxonomy(self, paper_path: str):
        pass

