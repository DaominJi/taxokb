from llm_factory import LLMFactory
from taxonomy_generator import TaskTaxonomyGenerator

if __name__ == "__main__":
    llm_factory = LLMFactory()
    task_taxonomy_generator = TaskTaxonomyGenerator(llm_factory)
    task_taxonomy = task_taxonomy_generator.generate_taxonomy('test')
    print(task_taxonomy)