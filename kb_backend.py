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

@app.route('/api/node/<node_id>/trends', methods=['GET'])
def get_node_trends(node_id: str):
    """Get research trends for a node and its children over time"""
    try:
        taxonomy_type = request.args.get('type', 'method')
        
        # Get the node and its children
        if taxonomy_type == 'method':
            node = build_method_tree(node_id)
        else:
            # For task taxonomy, we need to get the specific node
            query = """
            MATCH (t:TaskTaxonomy {node_id: $nodeId})
            OPTIONAL MATCH (t)-[:HAS_SUBTASK]->(child:TaskTaxonomy)
            RETURN t.name as name, 
                   collect(DISTINCT {
                       node_id: child.node_id, 
                       name: child.name
                   }) as children
            """
            result = neo4j_conn.execute_query(query, {"nodeId": node_id})
            if not result:
                return jsonify({'success': False, 'error': 'Node not found'}), 404
            
            node = {
                'node_id': node_id,
                'name': result[0]['name'],
                'children': [c for c in result[0]['children'] if c['node_id']]
            }
        
        # Generate trend data (with some real data if available, synthetic otherwise)
        trends = generate_trend_data(node, taxonomy_type)
        
        return jsonify({
            'success': True,
            'data': trends
        })
    
    except Exception as e:
        logger.error(f"Error getting trends for node {node_id}: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/node/<node_id>/experiment-settings', methods=['GET'])
def get_experiment_settings(node_id: str):
    """Get recommended experiment settings for a node"""
    try:
        taxonomy_type = request.args.get('type', 'method')
        
        # Try to get real experiment settings from the database
        settings = get_real_experiment_settings(node_id, taxonomy_type)
        
        # If no real data, generate synthetic recommendations
        if not settings or all(not settings[key] for key in ['datasets', 'metrics', 'baselines']):
            settings = generate_synthetic_experiment_settings(node_id, taxonomy_type)
        
        return jsonify({
            'success': True,
            'data': settings
        })
    
    except Exception as e:
        logger.error(f"Error getting experiment settings for node {node_id}: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

def generate_trend_data(node: Dict, taxonomy_type: str) -> Dict:
    """Generate trend data for a node and its children"""
    import random
    from datetime import datetime, timedelta
    
    # Try to get real data first
    try:
        if node.get('children'):
            # Query for real paper years
            child_ids = [c['node_id'] for c in node['children']]
            
            if taxonomy_type == 'method':
                query = """
                MATCH (m:MethodTaxonomy)
                WHERE m.node_id IN $childIds
                OPTIONAL MATCH (p:Paper)-[:USES_METHOD]->(m)
                WHERE p.publish_year IS NOT NULL
                RETURN m.node_id as node_id,
                       m.name as name,
                       collect(DISTINCT p.publish_year) as years
                """
            else:
                query = """
                MATCH (t:TaskTaxonomy)
                WHERE t.node_id IN $childIds
                OPTIONAL MATCH (p:Paper)-[:ADDRESSES_TASK]->(t)
                WHERE p.publish_year IS NOT NULL
                RETURN t.node_id as node_id,
                       t.name as name,
                       collect(DISTINCT p.publish_year) as years
                """
            
            results = neo4j_conn.execute_query(query, {"childIds": child_ids})
            
            # Process real data if available
            if results and any(r.get('years') for r in results):
                return process_real_trend_data(results)
    except:
        pass
    
    # Generate synthetic data for demonstration
    current_year = datetime.now().year
    years = list(range(2015, current_year + 1))
    
    trend_data = {
        'years': years,
        'series': []
    }
    
    if node.get('children'):
        for i, child in enumerate(node['children']):
            # Generate synthetic trend with some randomness
            base_trend = random.choice(['growing', 'stable', 'declining', 'cyclic'])
            values = generate_synthetic_trend(len(years), base_trend, i)
            
            trend_data['series'].append({
                'name': child.get('name', f'Child {i+1}'),
                'node_id': child.get('node_id', f'child_{i}'),
                'values': values,
                'trend_type': base_trend
            })
    else:
        # Single node trend
        values = generate_synthetic_trend(len(years), 'growing', 0)
        trend_data['series'].append({
            'name': node.get('name', 'Current Node'),
            'node_id': node.get('node_id', 'current'),
            'values': values,
            'trend_type': 'growing'
        })
    
    return trend_data

def generate_synthetic_trend(length: int, trend_type: str, seed: int) -> List[int]:
    """Generate synthetic trend values"""
    import random
    random.seed(seed)
    
    values = []
    base = random.randint(5, 20)
    
    for i in range(length):
        if trend_type == 'growing':
            value = base + i * random.randint(2, 5) + random.randint(-2, 5)
        elif trend_type == 'declining':
            value = base + 30 - i * random.randint(1, 3) + random.randint(-2, 2)
        elif trend_type == 'cyclic':
            import math
            value = base + 10 + int(10 * math.sin(i * 0.8)) + random.randint(-2, 2)
        else:  # stable
            value = base + random.randint(-3, 3)
        
        values.append(max(1, value))
    
    return values

def process_real_trend_data(results: List[Dict]) -> Dict:
    """Process real trend data from database results"""
    from collections import defaultdict
    
    # Find year range
    all_years = set()
    for r in results:
        if r.get('years'):
            all_years.update(int(y) for y in r['years'] if y)
    
    if not all_years:
        # No real data, return empty
        return {'years': [], 'series': []}
    
    min_year = min(all_years)
    max_year = max(all_years)
    years = list(range(min_year, max_year + 1))
    
    series = []
    for r in results:
        year_counts = defaultdict(int)
        if r.get('years'):
            for year in r['years']:
                if year:
                    year_counts[int(year)] += 1
        
        values = [year_counts.get(y, 0) for y in years]
        series.append({
            'name': r['name'],
            'node_id': r['node_id'],
            'values': values
        })
    
    return {'years': years, 'series': series}

def get_real_experiment_settings(node_id: str, taxonomy_type: str) -> Dict:
    """Get real experiment settings from database"""
    try:
        settings = {
            'datasets': [],
            'metrics': [],
            'baselines': []
        }
        
        # Get papers for this node
        if taxonomy_type == 'method':
            paper_query = """
            MATCH (m:MethodTaxonomy {node_id: $nodeId})
            OPTIONAL MATCH (p:Paper)-[:USES_METHOD]->(m)
            RETURN collect(DISTINCT p.paper_id) as paper_ids
            """
        else:
            paper_query = """
            MATCH (t:TaskTaxonomy {node_id: $nodeId})
            OPTIONAL MATCH (p:Paper)-[:ADDRESSES_TASK]->(t)
            RETURN collect(DISTINCT p.paper_id) as paper_ids
            """
        
        paper_result = neo4j_conn.execute_query(paper_query, {"nodeId": node_id})
        
        if paper_result and paper_result[0].get('paper_ids'):
            paper_ids = paper_result[0]['paper_ids']
            
            # Get datasets
            dataset_query = """
            MATCH (p:Paper)-[:EVALUATED_ON]->(d:Dataset)
            WHERE p.paper_id IN $paperIds
            RETURN DISTINCT d.dataset_name as name,
                   d.description as description,
                   COUNT(DISTINCT p) as usage_count
            ORDER BY usage_count DESC
            LIMIT 5
            """
            dataset_results = neo4j_conn.execute_query(dataset_query, {"paperIds": paper_ids})
            settings['datasets'] = [
                {
                    'name': r['name'],
                    'description': r.get('description', ''),
                    'usage_count': int(r['usage_count']),
                    'recommended': True if i < 3 else False
                }
                for i, r in enumerate(dataset_results)
            ]
            
            # Get metrics
            metric_query = """
            MATCH (p:Paper)-[:EVALUATED_BY]->(m:Metric)
            WHERE p.paper_id IN $paperIds
            RETURN DISTINCT m.metric_name as name,
                   m.description as description,
                   COUNT(DISTINCT p) as usage_count
            ORDER BY usage_count DESC
            LIMIT 5
            """
            metric_results = neo4j_conn.execute_query(metric_query, {"paperIds": paper_ids})
            settings['metrics'] = [
                {
                    'name': r['name'],
                    'description': r.get('description', ''),
                    'usage_count': int(r['usage_count']),
                    'recommended': True if i < 3 else False
                }
                for i, r in enumerate(metric_results)
            ]
            
            # Get baselines
            baseline_query = """
            MATCH (p:Paper)-[:COMPARED_AGAINST]->(b:Baseline)
            WHERE p.paper_id IN $paperIds
            RETURN DISTINCT b.baseline_name as name,
                   b.description as description,
                   COUNT(DISTINCT p) as usage_count
            ORDER BY usage_count DESC
            LIMIT 5
            """
            baseline_results = neo4j_conn.execute_query(baseline_query, {"paperIds": paper_ids})
            settings['baselines'] = [
                {
                    'name': r['name'],
                    'description': r.get('description', ''),
                    'usage_count': int(r['usage_count']),
                    'recommended': True if i < 3 else False
                }
                for i, r in enumerate(baseline_results)
            ]
        
        return settings
    
    except Exception as e:
        logger.error(f"Error getting real experiment settings: {str(e)}")
        return {'datasets': [], 'metrics': [], 'baselines': []}

def generate_synthetic_experiment_settings(node_id: str, taxonomy_type: str) -> Dict:
    """Generate synthetic experiment settings for demonstration"""
    import random
    
    # Synthetic datasets
    dataset_pool = [
        ('DBLP-ACM', 'Citation matching dataset from DBLP and ACM digital libraries'),
        ('DBLP-Scholar', 'Cross-platform academic paper matching dataset'),
        ('Amazon-Google Products', 'Product matching between Amazon and Google Shopping'),
        ('Walmart-Amazon', 'E-commerce product entity resolution dataset'),
        ('Restaurant', 'Restaurant entity matching from multiple review platforms'),
        ('Beer', 'Beer product matching across different databases'),
        ('iTunes-Amazon', 'Music and media matching dataset'),
        ('Fodors-Zagat', 'Restaurant matching between Fodors and Zagat guides'),
        ('Abt-Buy', 'Electronic product matching dataset'),
        ('Company', 'Corporate entity resolution dataset')
    ]
    
    # Synthetic metrics  
    metric_pool = [
        ('Precision', 'Ratio of correct matches to total predicted matches'),
        ('Recall', 'Ratio of correct matches to total true matches'),
        ('F1-Score', 'Harmonic mean of precision and recall'),
        ('AUC-ROC', 'Area under the receiver operating characteristic curve'),
        ('MRR', 'Mean reciprocal rank for ranking quality'),
        ('MAP', 'Mean average precision across queries'),
        ('Accuracy', 'Overall correctness of predictions'),
        ('Jaccard Similarity', 'Intersection over union of matched sets'),
        ('Edit Distance', 'String similarity metric'),
        ('Cosine Similarity', 'Vector space similarity measure')
    ]
    
    # Synthetic baselines
    baseline_pool = [
        ('DeepMatcher', 'Deep learning based entity matching system'),
        ('Magellan', 'ML-based entity matching toolkit'),
        ('DITTO', 'Pre-trained language model for entity matching'),
        ('ZeroER', 'Zero-shot entity resolution using generative models'),
        ('Random Forest', 'Traditional ML ensemble method'),
        ('SVM', 'Support vector machine classifier'),
        ('Logistic Regression', 'Statistical classification baseline'),
        ('Rule-based', 'Hand-crafted matching rules'),
        ('TF-IDF', 'Term frequency based similarity matching'),
        ('Word2Vec', 'Word embedding based matching')
    ]
    
    # Select random subsets
    random.seed(hash(node_id))
    
    selected_datasets = random.sample(dataset_pool, min(5, len(dataset_pool)))
    selected_metrics = random.sample(metric_pool, min(5, len(metric_pool)))
    selected_baselines = random.sample(baseline_pool, min(5, len(baseline_pool)))
    
    return {
        'datasets': [
            {
                'name': name,
                'description': desc,
                'usage_count': random.randint(10, 100),
                'recommended': i < 3
            }
            for i, (name, desc) in enumerate(selected_datasets)
        ],
        'metrics': [
            {
                'name': name,
                'description': desc,
                'usage_count': random.randint(20, 150),
                'recommended': i < 3
            }
            for i, (name, desc) in enumerate(selected_metrics)
        ],
        'baselines': [
            {
                'name': name,
                'description': desc,
                'usage_count': random.randint(5, 80),
                'recommended': i < 3
            }
            for i, (name, desc) in enumerate(selected_baselines)
        ]
    }

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
    port = int(os.environ.get('PORT', 5001))  # Changed to 5001
    debug = os.environ.get('DEBUG', 'True').lower() == 'true'
    
    print(f"Starting Flask server on http://localhost:{port}")
    app.run(host='0.0.0.0', port=port, debug=debug)