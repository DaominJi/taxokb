from llm_factory import LLMFactory
import yaml

prompts = yaml.safe_load(open('prompts.yaml', 'r', encoding='utf-8'))

class KnowledgeExtractor:
    def __init__(self, llm_factory: LLMFactory):
        self.llm_factory = llm_factory
        self.prompts = prompts
        
    def extract_knowledge(self, paper_path: str):
        pass
    
    