from langchain_community.graphs import Neo4jGraph
from langchain_openai import ChatOpenAI
from langchain.chains import GraphCypherQAChain
from langchain.prompts import PromptTemplate
import os

# Connect to Neo4j
graph = Neo4jGraph(
    url="bolt://localhost:7687",
    username="neo4j",
    password="xiaodaomin517"
)

# Initialize LLM (OpenAI example)
llm = ChatOpenAI(
    model="gpt-4.1",  # Updated to match your quick_query function
    temperature=0,
    api_key=os.getenv("OPENAI_API_KEY")
)

# Create the Cypher generation chain
cypher_chain = GraphCypherQAChain.from_llm(
    llm=llm,
    graph=graph,
    verbose=True,
    return_intermediate_steps=True,
    allow_dangerous_requests=True
)

def quick_query(question):
    """Quick one-liner query function with safety measures"""
    try:
        # Validate input
        if not question or not question.strip():
            return "Error: Empty question provided"
        
        # Check for OpenAI API key
        if not os.getenv("OPENAI_API_KEY"):
            return "Error: OPENAI_API_KEY environment variable not set"
        
        graph = Neo4jGraph(
            url="bolt://localhost:7687",
            username="neo4j",
            password="xiaodaomin517"
        )
        
        llm = ChatOpenAI(model="gpt-5", temperature = 1, api_key=os.getenv("OPENAI_API_KEY"))
        
        # Note: allow_dangerous_requests=True is needed for GraphCypherQAChain
        # This allows the LLM to generate and execute Cypher queries
        # Make sure your Neo4j user has limited permissions in production
        chain = GraphCypherQAChain.from_llm(
            llm=llm,
            graph=graph,
            verbose=True,
            allow_dangerous_requests=True
        )
        
        result = chain.invoke(question)
        return result
        
    except Exception as e:
        print(f"Query failed: {e}")
        return f"Error: {e}"

# Use it
answer = quick_query("What are the challenges of entity resolution?")
print(answer)