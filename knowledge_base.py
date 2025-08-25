from neo4j import GraphDatabase
import json

# Neo4j connection setup
class Neo4jLoader:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
    
    def close(self):
        self.driver.close()
    
    def clear_database(self):
        """Clear existing data"""
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
    
    def create_constraints(self):
        """Create uniqueness constraints and indexes"""
        with self.driver.session() as session:
            # Constraints for unique IDs
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (p:Paper) REQUIRE p.paper_id IS UNIQUE")
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (m:MethodTaxonomy) REQUIRE m.node_id IS UNIQUE")
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (t:TaskTaxonomy) REQUIRE t.node_id IS UNIQUE")
            
            # Indexes for better query performance
            session.run("CREATE INDEX IF NOT EXISTS FOR (p:Paper) ON (p.title)")
            session.run("CREATE INDEX IF NOT EXISTS FOR (m:MethodTaxonomy) ON (m.name)")
            session.run("CREATE INDEX IF NOT EXISTS FOR (t:TaskTaxonomy) ON (t.name)")
    
    def load_papers(self, method_data, task_data):
        """Load paper nodes from both JSON files"""
        papers = {}
        
        # Extract paper info from method_summaries
        for paper_file, details in method_data.get('method_summaries', {}).items():
            paper_id = details.get('paper_id')
            if paper_id:
                papers[paper_id] = {
                    'paper_id': paper_id,
                    'file_name': paper_file,
                    'methodology_summary': details.get('methodology_summary', '')
                }
        
        # Enrich with task problem definitions
        for paper_def in task_data.get('problem_definitions', []):
            paper_id = paper_def.get('paper_id')
            if paper_id in papers:
                papers[paper_id]['title'] = paper_def.get('paper_title', '')
                papers[paper_id]['problem_simple'] = paper_def.get('problem_formulation', {}).get('simple_description', '')
                papers[paper_id]['problem_formal'] = json.dumps(paper_def.get('problem_formulation', {}).get('formal_definition', {}))
        
        # Create paper nodes
        with self.driver.session() as session:
            for paper_id, paper_info in papers.items():
                query = """
                CREATE (p:Paper {
                    paper_id: $paper_id,
                    title: $title,
                    file_name: $file_name,
                    problem_simple: $problem_simple,
                    problem_formal: $problem_formal,
                    methodology_summary: $methodology_summary,
                    authors: $authors,
                    publish_year: $publish_year,
                    venue: $venue
                })
                """
                session.run(query, 
                    paper_id=paper_id,
                    title=paper_info.get('title', ''),
                    file_name=paper_info.get('file_name', ''),
                    problem_simple=paper_info.get('problem_simple', ''),
                    problem_formal=paper_info.get('problem_formal', ''),
                    methodology_summary=paper_info.get('methodology_summary', ''),
                    authors='',  # Not in provided data
                    publish_year='',  # Not in provided data
                    venue=''  # Not in provided data
                )
    
    def load_method_taxonomy(self, taxonomy_node, parent_id=None):
        """Recursively load method taxonomy nodes"""
        with self.driver.session() as session:
            # Generate unique node ID
            if parent_id:
                node_id = f"{parent_id}:{taxonomy_node.get('name', '').replace(' ', '_')}"
            else:
                node_id = "METHOD_ROOT"
            
            # Create node
            query = """
            CREATE (m:MethodTaxonomy {
                node_id: $node_id,
                name: $name,
                description: $description
            })
            """
            session.run(query,
                node_id=node_id,
                name=taxonomy_node.get('name', ''),
                description=taxonomy_node.get('content', '')
            )
            
            # Create parent-child relationship if parent exists
            if parent_id:
                rel_query = """
                MATCH (parent:MethodTaxonomy {node_id: $parent_id})
                MATCH (child:MethodTaxonomy {node_id: $child_id})
                CREATE (parent)-[:HAS_CHILD]->(child)
                """
                session.run(rel_query, parent_id=parent_id, child_id=node_id)
            
            # Link papers to this taxonomy node
            if 'index' in taxonomy_node:
                paper_ids = taxonomy_node['index'] if isinstance(taxonomy_node['index'], list) else [taxonomy_node['index']]
                for paper_id in paper_ids:
                    paper_query = """
                    MATCH (p:Paper {paper_id: $paper_id})
                    MATCH (m:MethodTaxonomy {node_id: $node_id})
                    CREATE (p)-[:USES_METHOD]->(m)
                    """
                    session.run(paper_query, paper_id=paper_id, node_id=node_id)
            
            # Process children recursively
            for child in taxonomy_node.get('children', []):
                self.load_method_taxonomy(child, node_id)
    
    def load_task_taxonomy(self, taxonomy_data):
        """Load task taxonomy from JSON"""
        taxonomy_json = json.loads(taxonomy_data.get('taxonomy', '{}'))
        taxonomy = taxonomy_json.get('taxonomy', {})
        
        def process_task_node(node, parent_id=None):
            with self.driver.session() as session:
                node_id = node.get('task_id', '')
                
                # Create task node
                query = """
                CREATE (t:TaskTaxonomy {
                    node_id: $node_id,
                    name: $name,
                    description: $description,
                    input_class: $input_class,
                    output_class: $output_class
                })
                """
                session.run(query,
                    node_id=node_id,
                    name=node.get('task_name', ''),
                    description=node.get('explanation', ''),
                    input_class=node.get('input_class', ''),
                    output_class=node.get('output_class', '')
                )
                
                # Create parent-child relationship
                if parent_id:
                    rel_query = """
                    MATCH (parent:TaskTaxonomy {node_id: $parent_id})
                    MATCH (child:TaskTaxonomy {node_id: $child_id})
                    CREATE (parent)-[:HAS_SUBTASK]->(child)
                    """
                    session.run(rel_query, parent_id=parent_id, child_id=node_id)
                
                # Link papers to this task
                for paper_id in node.get('papers', []):
                    paper_query = """
                    MATCH (p:Paper {paper_id: $paper_id})
                    MATCH (t:TaskTaxonomy {node_id: $node_id})
                    CREATE (p)-[:ADDRESSES_TASK]->(t)
                    """
                    session.run(paper_query, paper_id=paper_id, node_id=node_id)
                
                # Process children
                for child in node.get('children', []):
                    process_task_node(child, node_id)
        
        process_task_node(taxonomy)

# Main loading script
def load_graph_to_neo4j(method_json_path, task_json_path, uri, user, password):
    """Main function to load all data into Neo4j"""
    
    # Load JSON files
    with open(method_json_path, 'r') as f:
        method_data = json.load(f)
    
    with open(task_json_path, 'r') as f:
        task_data = json.load(f)
    
    # Initialize Neo4j connection
    loader = Neo4jLoader(uri, user, password)
    
    try:
        # Clear and setup database
        print("Clearing database...")
        loader.clear_database()
        
        print("Creating constraints...")
        loader.create_constraints()
        
        # Load data
        print("Loading papers...")
        loader.load_papers(method_data, task_data)
        
        print("Loading method taxonomy...")
        loader.load_method_taxonomy(method_data['taxonomy'])
        
        print("Loading task taxonomy...")
        loader.load_task_taxonomy(task_data)
        
        print("Graph loaded successfully!")
        
    finally:
        loader.close()

# Usage
if __name__ == "__main__":
    # Neo4j connection parameters
    NEO4J_URI = "bolt://localhost:7687"  # Update with your Neo4j URI
    NEO4J_USER = "neo4j"  # Update with your username
    NEO4J_PASSWORD = "xiaodaomin517"  # Update with your password
    
    # File paths
    METHOD_JSON = "output/data_with_paper_ids.json"
    TASK_JSON = "output/task_taxonomy.json"
    
    # Load the graph
    load_graph_to_neo4j(METHOD_JSON, TASK_JSON, NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)