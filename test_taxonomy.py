from llm_factory import LLMFactory
from taxonomy_generator import TaskTaxonomyGenerator
from taxonomy_generator import MethodTaxonomyGenerator

if __name__ == "__main__":
    llm_factory = LLMFactory()
    #task_taxonomy_generator = TaskTaxonomyGenerator(llm_factory)
    #task_taxonomy = task_taxonomy_generator.generate_taxonomy('test',output_path='output/task_taxonomy.json')
    #print(task_taxonomy)

    method_taxonomy_generator = MethodTaxonomyGenerator(llm_factory)
    method_taxonomy = method_taxonomy_generator.generate_taxonomy('test',output_path='output/method_taxonomy.json')
    print(method_taxonomy)