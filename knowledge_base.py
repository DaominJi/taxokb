from neo4j import GraphDatabase
import json
import pandas as pd
import os
from pdf_to_md_converter import title_to_id

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
    
    def load_papers(self):
        """Load paper nodes from both JSON files"""
        papers = {}
        
        # # Extract paper info from method_summaries
        # for paper_file, details in method_data.get('method_summaries', {}).items():
        #     paper_id = details.get('paper_id')
        #     if paper_id:
        #         papers[paper_id] = {
        #             'paper_id': paper_id,
        #             'file_name': paper_file,
        #             'methodology_summary': details.get('methodology_summary', '')
        #         }
        
        # # Enrich with task problem definitions
        # for paper_def in task_data.get('problem_definitions', []):
        #     paper_id = paper_def.get('paper_id')
        #     if paper_id in papers:
        #         papers[paper_id]['title'] = paper_def.get('paper_title', '')
        #         papers[paper_id]['problem_simple'] = paper_def.get('problem_formulation', {}).get('simple_description', '')
        #         papers[paper_id]['problem_formal'] = json.dumps(paper_def.get('problem_formulation', {}).get('formal_definition', {}))
        
        paper_register = pd.read_csv('data/paper_register.csv')
        for index, row in paper_register.iterrows():
            paper_id = row['id']
            title = row['title']
            authors = row['authors']
            publish_year = row['year']
            venue = row['venue']
            citation_count = row['citation_count']
            semantic_scholar_id = row['semantic_scholar_id']
            #id,path,title,section_ path,categorization_path,year,authors,venue,citation_count,semantic_scholar_id
            papers[paper_id] = {
                'paper_id': paper_id,
                'title': title,
                'authors': authors,
                'publish_year': publish_year,
                'venue': venue,
                'citation_count': citation_count,
                'semantic_scholar_id': semantic_scholar_id
            }
        # Create paper nodes
        with self.driver.session() as session:
            for paper_id, paper_info in papers.items():
                query = """
                CREATE (p:Paper {
                    paper_id: $paper_id,
                    title: $title,
                    authors: $authors,
                    publish_year: $publish_year,
                    venue: $venue,
                    citation_count: $citation_count,
                    semantic_scholar_id: $semantic_scholar_id
                })
                """
                session.run(query, 
                    paper_id=paper_id,
                    title=paper_info.get('title', ''),
                    authors=paper_info.get('authors', ''),
                    publish_year=paper_info.get('publish_year', ''),
                    venue=paper_info.get('venue', ''),
                    citation_count=paper_info.get('citation_count', ''),
                    semantic_scholar_id=paper_info.get('semantic_scholar_id', '')
                )

    def load_chunks(self, path):
        with open(path, 'r') as f:
            chunks = json.load(f)
        paper_register = pd.read_csv('data/paper_register.csv')
        ids = paper_register['id'].tolist()
        titles = paper_register['title'].tolist()
        with self.driver.session() as session:
            for paper_id, paper_title in zip(ids, titles):
                path = f'output/{id}_summary.json'
                if os.path.exists(path):
                    with open(path, 'r') as f:
                        summary = json.load(f)
                    for key, value in summary.items():
                        query = """
                        CREATE (p:Chunk {
                            chunk_id: $chunk_id,
                            title: $title,
                            paper_id: $paper_id,
                            paper_title: $paper_title,
                            content: $content,
                            summary: $summary,
                        })
                    """
                    chunk_id = title_to_id(paper_title + value.get('title', ''))
                    session.run(query,
                        chunk_id=chunk_id,
                        title=value.get('title', ''),
                        paper_id=paper_id,
                        paper_title=paper_title,
                        content=value.get('content', ''),
                        summary=value.get('summary', '')
                    )

                    # Create has_chunk relationship
                    query = """
                    MATCH (p:Paper {paper_id: $paper_id})
                    MATCH (c:Chunk {chunk_id: $chunk_id})
                    CREATE (p)-[:HAS_CHUNK]->(c)
                    """
                    session.run(query, paper_id=paper_id, chunk_id=chunk_id)
                else:
                    continue
    
    def load_dataset(self, path):
        with open(path, 'r') as f:
            datasets = json.load(f).get('canonical_datasets', [])
        with self.driver.session() as session:
            for dataset_dict in datasets:
                dataset_name = dataset_dict.get('canonical_name', '')
                dataset_id = title_to_id(dataset_name)
                query = """
                CREATE (d:Dataset {
                    dataset_id: $dataset_id,
                    dataset_name: $dataset_name,
                    aliases: $aliases,
                    description: $description,
                    task_type: $task_type,
                    supporting_papers: $supporting_papers,
                    usage_frequency: $usage_frequency
                })
                """
                session.run(query,
                    dataset_id=dataset_id,
                    dataset_name=dataset_name,
                    aliases=dataset_dict.get('aliases', []),
                    description=dataset_dict.get('description', ''),
                    task_type=dataset_dict.get('task_type', []),
                    supporting_papers=dataset_dict.get('supporting_papers', []),
                    usage_frequency=dataset_dict.get('usage_frequency', '')
                )
                
                # Create evaluated-on relationship if parent exists
                if 'supporting_papers' in dataset_dict:
                    for paper_id in dataset_dict['supporting_papers']:
                        paper_query = """
                        MATCH (p:Paper {paper_id: $paper_id})
                        MATCH (d:Dataset {dataset_id: $dataset_id})
                        CREATE (p)-[:EVALUATED_ON]->(d)
                        """
                        session.run(paper_query, paper_id=paper_id, dataset_id=dataset_id)

    def load_metric(self, path):
        with open(path, 'r') as f:
            metrics = json.load(f).get('canonical_metrics', [])
        with self.driver.session() as session:
            for metric_dict in metrics:
                metric_name = metric_dict.get('canonical_name', '')
                metric_id = title_to_id(metric_name)
                query = """
                CREATE (m:Metric {
                    metric_id: $metric_id,
                    metric_name: $metric_name,
                    aliases: $aliases,
                    description: $description,
                    formulation: $formulation,
                    supporting_papers: $supporting_papers,
                    usage_frequency: $usage_frequency
                })
                """
                session.run(query,
                    metric_id=metric_id,
                    metric_name=metric_name,
                    aliases=metric_dict.get('aliases', []),
                    description=metric_dict.get('description', ''),
                    formulation=metric_dict.get('formulation', ''),
                    supporting_papers=metric_dict.get('supporting_papers', []),
                    usage_frequency=metric_dict.get('usage_frequency', '')
                )
                
                # Create evaluated-by relationship if parent exists
                if 'supporting_papers' in metric_dict:
                    for paper_id in metric_dict['supporting_papers']:
                        paper_query = """
                        MATCH (p:Paper {paper_id: $paper_id})
                        MATCH (m:Metric {metric_id: $metric_id})
                        CREATE (p)-[:EVALUATED_BY]->(m)
                        """
                        session.run(paper_query, paper_id=paper_id, metric_id=metric_id)
    
    def load_baseline(self, path):
        with open(path, 'r') as f:
            baselines = json.load(f).get('canonical_baselines', [])
        with self.driver.session() as session:
            for baseline_dict in baselines:
                baseline_name = baseline_dict.get('canonical_name', '')
                baseline_id = title_to_id(baseline_name)
                query = """
                CREATE (b:Baseline {
                    baseline_id: $baseline_id,
                    baseline_name: $baseline_name,
                    aliases: $aliases,
                    description: $description,
                    supporting_papers: $supporting_papers,
                    usage_frequency: $usage_frequency
                })
                """
                session.run(query,
                    baseline_id=baseline_id,
                    baseline_name=baseline_name,
                    aliases=baseline_dict.get('aliases', []),
                    description=baseline_dict.get('description', ''),
                    supporting_papers=baseline_dict.get('supporting_papers', []),
                    usage_frequency=baseline_dict.get('usage_frequency', '')
                )
                
                # Create compared-against relationship if parent exists
                if 'supporting_papers' in baseline_dict:
                    for paper_id in baseline_dict['supporting_papers']:
                        paper_query = """
                        MATCH (p:Paper {paper_id: $paper_id})
                        MATCH (b:Baseline {baseline_id: $baseline_id})
                        CREATE (p)-[:COMPARED_AGAINST]->(b)
                        """
                        session.run(paper_query, paper_id=paper_id, baseline_id=baseline_id)
    
    def load_method_taxonomy(self, taxonomy_node, parent_id=None):
        """Recursively load method taxonomy nodes"""
        with self.driver.session() as session:
            # Generate unique node ID
            # if parent_id:
            #     node_id = f"{parent_id}:{taxonomy_node.get('name', '').replace(' ', '_')}"
            # else:
            #     node_id = "METHOD_ROOT"
            
            # Create node
            query = """
            CREATE (m:MethodTaxonomy {
                node_id: $node_id,
                name: $name,
                description: $description
            })
            """
            name = taxonomy_node.get('name', '')
            if not parent_id:
                name = "entity_resolution"
            node_id = title_to_id(name)
            session.run(query,
                node_id=node_id,
                name=name,
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
                    paper_id = paper_id[1:]
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
                node_name = node.get('task_name', '')
                node_id = title_to_id(node_name)
                
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
                    name=node_name,
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
        loader.load_papers()
        
        print("Loading chunks...")
        chunks_path = 'output/'
        for file in os.listdir(chunks_path):
            if file.endswith('_summary.json'):
                loader.load_chunks(os.path.join(chunks_path, file))
        
        print("Loading datasets...")
        loader.load_dataset('experiment_settings.json')
        
        print("Loading metrics...")
        loader.load_metric('experiment_settings.json')
        
        print("Loading baselines...")
        loader.load_baseline('experiment_settings.json')
        
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
    METHOD_JSON = "output/method_taxonomy.json"
    TASK_JSON = "output/task_taxonomy.json"
    
    # Load the graph
    load_graph_to_neo4j(METHOD_JSON, TASK_JSON, NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)