from chunker import chunking, labeling_section
from llm_factory import LLMFactory

if __name__ == "__main__":
    llm_factory = LLMFactory()
    section_dict = chunking('test/Auto-Join.md', 'test/section_dict.json    ')
    labeling_section(section_dict, 'test/section_dict_labeled.json', llm_factory)