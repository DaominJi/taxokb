import os
from pathlib import Path
from helper import *
import re
import json
import argparse
from llm_factory import LLMFactory
import yaml

roman_numerals = {
    'I': 1,
    'II': 2,
    'III': 3,
    'IV': 4,
    'V': 5,
    'VI': 6,
    'VII': 7,
    'VIII': 8,
    'IX': 9,
    'X': 10,
    'XI': 11,
    'XII': 12,
    'XIII': 13,
    'XIV': 14,
    'XV': 15,
    'XVI': 16,
    'XVII': 17,
    'XVIII': 18,
}

def chunking(md_file_path, json_output_path):
    section_dict = {}

    # Read the markdown content
    with open(md_file_path, 'r', encoding='utf-8') as f:
        md_content = f.read()

    # Match any heading starting with #
    pattern = r'^(#{1,6})\s+(.+)$'
    matches = list(re.finditer(pattern, md_content, flags=re.MULTILINE))

    # Add dummy end to simplify slicing
    spans = [m.span() for m in matches] + [(len(md_content), len(md_content))]
    titles = [m.group(0).strip() for m in matches]  # full heading line

    # Extract content under each heading
    for i in range(len(matches)):
        title = titles[i].replace(':', '_').strip()
        start = spans[i][1]  # start after heading line
        end = spans[i+1][0]
        content = md_content[start:end].strip()
        section_dict[title] = content

    # Save to JSON
    with open(json_output_path, 'w', encoding='utf-8') as json_file:
        json.dump(section_dict, json_file, ensure_ascii=False, indent=2)

    print(f"✅ Sections saved to: {json_output_path}")
    return section_dict

def labeling_section(section_dict, json_output_path, llm_factory):
    # Load the YAML configuration file
    with open('prompts.yaml', 'r', encoding='utf-8') as f:
        prompts = yaml.safe_load(f)
    
    prompt = prompts['chunker']['section_labeling']
    prompt = prompt.replace('[Title List]', ''.join(list(section_dict.keys())))

    response = llm_factory.generate(
        prompt,
        model = 'gpt-4o-mini',
        max_tokens = 1000,
        temperature = 0
    )

    response_json = json.loads(response.content)    
    with open(json_output_path, 'w', encoding='utf-8') as json_file:
        json.dump(response_json, json_file, ensure_ascii=False, indent=2)

    print(f"✅ Sections labeled and saved to: {json_output_path}")
    return response_json


        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess the paper files.")
    parser.add_argument("--basic", type=str, default="data/cross_modal_retrieval", 
                       help="Directory containing raw research papers.")
    args = parser.parse_args()
    
    input_path = Path(args.basic)
    source_path = input_path/ 'md_files'
    target_path = input_path /'md_files_chunking'
    if not source_path.exists():
        raise FileNotFoundError(f"Input directory {source_path} does not exist.")

    if not target_path.exists():
        print(f"Creating output directory: {target_path}")
        target_path.mkdir(parents=True, exist_ok=True)

    agent = LLMAgent(api_key = api_key)
    # Prompts/Step_0_Keycontent_extraction.md
    base_dir = Path(__file__).parent
    prompt_path = base_dir / "Prompts/Step_0_Keycontent_extraction.md"
    with open(prompt_path, 'r', encoding='utf-8') as f:
        prompt_template = f.read()

    for md_file in source_path.glob('*'):
        md_file_name = md_file.name.replace('.md', '.json')
        output_file = target_path / f"Section_division_{md_file_name}"

        md_content = open(md_file, 'r').read()
        title = '# ' + extract_title(md_content) + '\n'

        results = chunking(md_file, target_path/md_file_name)

        section_keys = list(results.keys())
        formatted_titles = ', '.join(section_keys)
        prompt_str = prompt_template.replace("[Title List]", str(section_keys))
        response = agent(prompt_str)

        response_json = json.loads(response) 
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(response_json, f, ensure_ascii=False, indent=2)
        print(f"✅ Saved LLM response to: {output_file}")

    related_work_path = target_path
    related_work_content = []
    for file_name in related_work_path.glob('Section_division_*'):
        sub_path = target_path / file_name.name.removeprefix('Section_division_')
        with open(sub_path, 'r', encoding='utf-8') as file:
            content = json.load(file)
        with open(file_name, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if 'Related Work' in data:
            for section in data['Related Work']:
                related_work_content.append(content[section])
    with open(related_work_path / 'related_work_content.md', 'w', encoding='utf-8') as f:
        f.write('\n'.join(related_work_content))    
    print(f"✅ Saved related work content to: {related_work_path / 'related_work_content.md'}")




