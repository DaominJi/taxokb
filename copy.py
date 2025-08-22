from ruamel.yaml import YAML
from ruamel.yaml.scalarstring import LiteralScalarString

yaml = YAML()

prompt_path_1 = '/Users/daominji/Project/Research Paper Analysis/taxonomy_by_methodology_new/prompts/6_extract_method.yaml'
prompt_path_2 = '/Users/daominji/Project/Research Paper Analysis/taxonomy_by_methodology_new/prompts/14_summarize_pros_cons_methods.yaml'
prompt_path_3 = '/Users/daominji/Project/Research Paper Analysis/taxonomy_by_methodology_new/prompts/generate_taxonomy_tree.md'

file = 'prompts.yaml'

with open(file, 'r') as file:
    prompts = yaml.load(file)

with open(prompt_path_1, 'r') as file:
    prompt_1 = yaml.load(file)

with open(prompt_path_2, 'r') as file:
    prompt_2 = yaml.load(file)

with open(prompt_path_3, 'r') as file:
    prompt_3 = file.read()

prompts['taxonomy_generator']['method_taxonomy']['extract_method_summary'] = prompt_1['prompt']
prompts['taxonomy_generator']['method_taxonomy']['extract_pros_and_cons'] = prompt_2['prompt']
prompts['taxonomy_generator']['method_taxonomy']['taxonomy_generation'] = prompt_3

saved_file = 'prompts_new.yaml'
with open(saved_file, 'w') as file:
    yaml.dump(prompts, file)