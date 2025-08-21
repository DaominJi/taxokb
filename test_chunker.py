from chunker import PaperChunker
from llm_factory import LLMFactory

if __name__ == "__main__":
    llm_factory = LLMFactory()
    chunker = PaperChunker(llm_factory)
    section_dict, categorization = chunker.process_paper('test/PEXESO.md', 'test/')
    #print(section_dict)
    #print(categorization)
    