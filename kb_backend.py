"""
Flask Backend for Entity Resolution Taxonomy Viewer
Handles Neo4j connections and provides REST API for frontend
"""

from flask import Flask, jsonify, request
from flask_cors import CORS
from neo4j import GraphDatabase
import os
from typing import Dict, List, Any, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Neo4j configuration
class Neo4jConnection:
    def __init__(self):
        self.driver = None
        self.uri = None
        self.user = None
        self.password = None
    
    def connect(self, uri: str, user: str, password: str):
        """Establish connection to Neo4j database"""
        try:
            if self.driver:
                self.driver.close()
            
            self.driver = GraphDatabase.driver(uri, auth=(user, password))
            # Test connection
            with self.driver.session() as session:
                session.run("RETURN 1")
            
            self.uri = uri
            self.user = user
            self.password = password
            logger.info(f"Connected to Neo4j at {uri}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {str(e)}")
            raise
    
    def close(self):
        """Close Neo4j connection"""
        if self.driver:
            self.driver.close()
            self.driver = None
    
    def execute_query(self, query: str, parameters: Dict = None):
        """Execute a Cypher query and return results"""
        if not self.driver:
            raise Exception("Not connected to Neo4j")
        
        with self.driver.session() as session:
            result = session.run(query, parameters or {})
            return [record.data() for record in result]

# Global Neo4j connection instance
neo4j_conn = Neo4jConnection()

# Helper functions
def build_method_tree(node_id: str) -> Optional[Dict]:
    """Recursively build method taxonomy tree"""
    try:
        # Get node details and paper count
        node_query = """
        MATCH (m:MethodTaxonomy {node_id: $nodeId})
        OPTIONAL MATCH (p:Paper)-[:USES_METHOD]->(m)
        RETURN m.node_id as node_id,
               m.name as name,
               m.description as description,
               COUNT(DISTINCT p) as paper_count
        """
        
        node_results = neo4j_conn.execute_query(node_query, {"nodeId": node_id})
        
        if not node_results:
            return None
        
        node = node_results[0]
        node['paper_count'] = int(node['paper_count'])
        
        # Get children
        children_query = """
        MATCH (parent:MethodTaxonomy {node_id: $parentId})-[:HAS_CHILD]->(child:MethodTaxonomy)
        RETURN child.node_id as node_id, child.name as name
        ORDER BY child.name
        """
        
        children_results = neo4j_conn.execute_query(children_query, {"parentId": node_id})
        
        node['children'] = []
        for child_data in children_results:
            child_node = build_method_tree(child_data['node_id'])
            if child_node:
                node['children'].append(child_node)
        
        # Calculate total papers including children
        if node['children']:
            child_papers = sum(child.get('total_papers', child.get('paper_count', 0)) 
                             for child in node['children'])
            node['total_papers'] = node['paper_count'] + child_papers
        else:
            node['total_papers'] = node['paper_count']
        
        return node
    
    except Exception as e:
        logger.error(f"Error building method tree for {node_id}: {str(e)}")
        return None

def calculate_total_papers(node: Dict) -> int:
    """Calculate total papers for a node including all descendants"""
    if not node:
        return 0
    
    total = node.get('paper_count', 0)
    if 'children' in node and node['children']:
        for child in node['children']:
            total += calculate_total_papers(child)
    
    node['total_papers'] = total
    return total

# API Routes
@app.route('/api/connect', methods=['POST'])
def connect():
    """Connect to Neo4j database"""
    try:
        data = request.json
        uri = data.get('uri', 'bolt://localhost:7687')
        user = data.get('user', 'neo4j')
        password = data.get('password')
        
        if not password:
            return jsonify({'success': False, 'error': 'Password is required'}), 400
        
        neo4j_conn.connect(uri, user, password)
        
        return jsonify({
            'success': True,
            'message': 'Connected successfully to Neo4j'
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/disconnect', methods=['POST'])
def disconnect():
    """Disconnect from Neo4j database"""
    try:
        neo4j_conn.close()
        return jsonify({
            'success': True,
            'message': 'Disconnected from Neo4j'
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/taxonomy/method', methods=['GET'])
def get_method_taxonomy():
    """Get method taxonomy tree"""
    try:
        # Find root node
        root_query = """
        MATCH (m:MethodTaxonomy)
        WHERE m.name = 'entity_resolution' OR m.node_id = 'entity_resolution'
        RETURN m.node_id as node_id
        LIMIT 1
        """
        
        root_results = neo4j_conn.execute_query(root_query)
        
        if not root_results:
            # Try to find any method taxonomy node without parent
            fallback_query = """
            MATCH (m:MethodTaxonomy)
            WHERE NOT (m)<-[:HAS_CHILD]-(:MethodTaxonomy)
            RETURN m.node_id as node_id
            LIMIT 1
            """
            root_results = neo4j_conn.execute_query(fallback_query)
            
            if not root_results:
                return jsonify({
                    'success': False,
                    'error': 'No method taxonomy root found'
                }), 404
        
        root_id = root_results[0]['node_id']
        taxonomy = build_method_tree(root_id)
        
        if not taxonomy:
            return jsonify({
                'success': False,
                'error': 'Failed to build method taxonomy'
            }), 500
        
        return jsonify({
            'success': True,
            'data': taxonomy
        })
    
    except Exception as e:
        logger.error(f"Error getting method taxonomy: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/taxonomy/task', methods=['GET'])
def get_task_taxonomy():
    """Get task taxonomy tree"""
    try:
        # Get all task nodes with relationships
        query = """
        MATCH (t:TaskTaxonomy)
        OPTIONAL MATCH (t)<-[:HAS_SUBTASK]-(parent:TaskTaxonomy)
        OPTIONAL MATCH (p:Paper)-[:ADDRESSES_TASK]->(t)
        RETURN t.node_id as node_id,
               t.name as name,
               t.description as description,
               t.input_class as input_class,
               t.output_class as output_class,
               parent.node_id as parent_id,
               COUNT(DISTINCT p) as paper_count
        ORDER BY t.name
        """
        
        results = neo4j_conn.execute_query(query)
        
        if not results:
            return jsonify({
                'success': False,
                'error': 'No task taxonomy found'
            }), 404
        
        # Build tree structure
        nodes = {}
        root_nodes = []
        
        # First pass: create all nodes
        for record in results:
            node_id = record['node_id']
            
            if node_id not in nodes:
                nodes[node_id] = {
                    'node_id': node_id,
                    'name': record['name'],
                    'description': record['description'],
                    'input_class': record.get('input_class'),
                    'output_class': record.get('output_class'),
                    'paper_count': int(record['paper_count']),
                    'children': []
                }
            else:
                # Update paper count if we have more info
                nodes[node_id]['paper_count'] = max(
                    nodes[node_id].get('paper_count', 0),
                    int(record['paper_count'])
                )
        
        # Second pass: build hierarchy
        for record in results:
            node_id = record['node_id']
            parent_id = record.get('parent_id')
            
            if parent_id and parent_id in nodes:
                if not any(c['node_id'] == node_id for c in nodes[parent_id]['children']):
                    nodes[parent_id]['children'].append(nodes[node_id])
            elif not parent_id:
                if not any(r['node_id'] == node_id for r in root_nodes):
                    root_nodes.append(nodes[node_id])
        
        # Create response based on root nodes
        if len(root_nodes) > 1:
            # Multiple roots: create virtual root
            taxonomy = {
                'node_id': 'task_root',
                'name': 'Entity Resolution Tasks',
                'description': 'Root node for all entity resolution tasks',
                'paper_count': sum(node.get('paper_count', 0) for node in root_nodes),
                'children': root_nodes
            }
        elif len(root_nodes) == 1:
            taxonomy = root_nodes[0]
        else:
            # No root found, return empty taxonomy
            taxonomy = {
                'node_id': 'task_root',
                'name': 'Entity Resolution Tasks',
                'description': 'No task taxonomy found in database',
                'paper_count': 0,
                'children': []
            }
        
        # Calculate total papers
        calculate_total_papers(taxonomy)
        
        return jsonify({
            'success': True,
            'data': taxonomy
        })
    
    except Exception as e:
        logger.error(f"Error getting task taxonomy: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/node/<node_id>/papers', methods=['GET'])
def get_node_papers(node_id: str):
    """Get papers associated with a specific node"""
    try:
        taxonomy_type = request.args.get('type', 'method')
        
        if taxonomy_type == 'method':
            query = """
            MATCH (m:MethodTaxonomy {node_id: $nodeId})
            OPTIONAL MATCH (p:Paper)-[:USES_METHOD]->(m)
            RETURN p.paper_id as paper_id,
                   p.title as title,
                   p.authors as authors,
                   p.publish_year as year,
                   p.venue as venue,
                   p.citation_count as citations
            ORDER BY p.citation_count DESC
            LIMIT 20
            """
        else:
            query = """
            MATCH (t:TaskTaxonomy {node_id: $nodeId})
            OPTIONAL MATCH (p:Paper)-[:ADDRESSES_TASK]->(t)
            RETURN p.paper_id as paper_id,
                   p.title as title,
                   p.authors as authors,
                   p.publish_year as year,
                   p.venue as venue,
                   p.citation_count as citations
            ORDER BY p.citation_count DESC
            LIMIT 20
            """
        
        results = neo4j_conn.execute_query(query, {"nodeId": node_id})
        
        papers = [r for r in results if r.get('paper_id')]
        
        return jsonify({
            'success': True,
            'data': papers
        })
    
    except Exception as e:
        logger.error(f"Error getting papers for node {node_id}: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/statistics', methods=['GET'])
def get_statistics():
    """Get overall statistics about the taxonomies"""
    try:
        stats_query = """
        MATCH (p:Paper)
        WITH COUNT(p) as total_papers
        MATCH (m:MethodTaxonomy)
        WITH total_papers, COUNT(m) as method_nodes
        MATCH (t:TaskTaxonomy)
        WITH total_papers, method_nodes, COUNT(t) as task_nodes
        OPTIONAL MATCH (d:Dataset)
        WITH total_papers, method_nodes, task_nodes, COUNT(d) as datasets
        OPTIONAL MATCH (me:Metric)
        WITH total_papers, method_nodes, task_nodes, datasets, COUNT(me) as metrics
        OPTIONAL MATCH (b:Baseline)
        RETURN total_papers, method_nodes, task_nodes, datasets, metrics, COUNT(b) as baselines
        """
        
        results = neo4j_conn.execute_query(stats_query)
        
        if results:
            stats = results[0]
            return jsonify({
                'success': True,
                'data': {
                    'total_papers': stats.get('total_papers', 0),
                    'method_nodes': stats.get('method_nodes', 0),
                    'task_nodes': stats.get('task_nodes', 0),
                    'datasets': stats.get('datasets', 0),
                    'metrics': stats.get('metrics', 0),
                    'baselines': stats.get('baselines', 0)
                }
            })
        
        return jsonify({
            'success': True,
            'data': {
                'total_papers': 0,
                'method_nodes': 0,
                'task_nodes': 0,
                'datasets': 0,
                'metrics': 0,
                'baselines': 0
            }
        })
    
    except Exception as e:
        logger.error(f"Error getting statistics: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/search', methods=['GET'])
def search_nodes():
    """Search for nodes by name"""
    try:
        query_param = request.args.get('q', '')
        taxonomy_type = request.args.get('type', 'all')
        
        if not query_param:
            return jsonify({'success': True, 'data': []})
        
        search_query = """
        MATCH (n)
        WHERE (
            (n:MethodTaxonomy AND ($type = 'all' OR $type = 'method')) OR
            (n:TaskTaxonomy AND ($type = 'all' OR $type = 'task'))
        )
        AND toLower(n.name) CONTAINS toLower($search)
        RETURN n.node_id as node_id,
               n.name as name,
               n.description as description,
               labels(n)[0] as type
        LIMIT 20
        """
        
        results = neo4j_conn.execute_query(
            search_query,
            {"search": query_param, "type": taxonomy_type}
        )
        
        return jsonify({
            'success': True,
            'data': results
        })
    
    except Exception as e:
        logger.error(f"Error searching nodes: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    try:
        # Check if Neo4j is connected
        if neo4j_conn.driver:
            with neo4j_conn.driver.session() as session:
                session.run("RETURN 1")
            db_status = "connected"
        else:
            db_status = "disconnected"
        
        return jsonify({
            'status': 'healthy',
            'database': db_status
        })
    except:
        return jsonify({
            'status': 'healthy',
            'database': 'disconnected'
        })

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({'success': False, 'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'success': False, 'error': 'Internal server error'}), 500

# Cleanup on shutdown
def cleanup():
    if neo4j_conn:
        neo4j_conn.close()

if __name__ == '__main__':
    import atexit
    atexit.register(cleanup)
    
    # Configuration from environment variables or defaults
    port = int(os.environ.get('PORT', 5001))
    debug = os.environ.get('DEBUG', 'True').lower() == 'true'
    
    app.run(host='0.0.0.0', port=port, debug=debug)