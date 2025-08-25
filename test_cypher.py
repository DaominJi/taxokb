from neo4j import GraphDatabase

driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "xiaodaomin517"))


def run_query(query, parameters=None):
    with driver.session() as session:
        result = session.run(query, parameters or {})
        return [record for record in result]

# Example query
query = """
MATCH (m:MethodTaxonomy {node_id: $node_id})
MATCH (p:Paper)-[:USES_METHOD]->(m)
OPTIONAL MATCH (p)-[:EVALUATED_ON]->(d:Dataset)
OPTIONAL MATCH (p)-[:EVALUATED_BY]->(metric:Metric)
OPTIONAL MATCH (p)-[:COMPARED_AGAINST]->(b:Baseline)
WITH m, 
     COLLECT(DISTINCT d) AS datasets,
     COLLECT(DISTINCT metric) AS metrics,
     COLLECT(DISTINCT b) AS baselines
RETURN m.name AS taxonomy_node,
       datasets,
       metrics,
       baselines
"""

records = run_query(query, {"node_id": "ca7a6e3c"})

for record in records:
    print(record["datasets"])