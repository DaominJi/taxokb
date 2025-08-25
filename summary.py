from llm_factory import LLMFactory
import json
import yaml
import pandas as pd

class SummaryGenerator:
    def __init__(self, llm_factory: LLMFactory):
        self.llm_factory = llm_factory
        with open("config.yaml", "r") as f:
            config = yaml.safe_load(f)
        self.aspects = config["summary_generator"]["sections"]
        self.summary = {}
        for aspect in self.aspects:
            self.summary[aspect] = {}

    def generate_summary(self, paper_id: str):
        section_dict_path = f"test/{paper_id}_sections.json"
        categorized_path = f"test/{paper_id}_categorized.json"

        with open(section_dict_path, 'r') as f:
            section_dict = json.load(f)

        with open(categorized_path, 'r') as f:
            categorized = json.load(f)

        for aspect in self.aspects:
            self.summary[aspect]['content'] = ""
            self.summary[aspect]['title'] = categorized[aspect]
            if aspect in categorized:
                for section_tilte in categorized[aspect]:
                    self.summary[aspect]['content'] += section_dict[section_tilte] + "\n"
            prompt_template = yaml.safe_load(open("prompts.yaml", "r"))["summary_generator"]
            prompt = prompt_template.replace('[Section Content]', self.summary[aspect]['content'])
            print (prompt)
            self.summary[aspect]['summary'] = self.llm_factory.generate(
                prompt,
                model='gpt-4.1',
                max_tokens=30000,
                temperature=0.7
            ).content
        return self.summary

if __name__ == "__main__":
    llm_factory = LLMFactory()
    summary_generator = SummaryGenerator(llm_factory)
    paper_register = pd.read_csv("data/paper_register.csv")
    paper_ids = paper_register['id']
    for paper_id in paper_ids:
        summary = summary_generator.generate_summary(paper_id)
        path = f"output/{paper_id}_summary.json"
        with open(path, "w") as f:
            json.dump(summary, f, indent=4)
        
        