import json
import re

# Create the mapping dictionary
p_to_id = {
    'P1': 'f32a10ff',
    'P2': '53cfb6e5',
    'P3': '79bc4b32',
    'P4': 'baa3b0fc',
    'P5': '54593c94',
    'P6': '98c01c4b',
    'P7': 'b24a912c',
    'P8': 'd6c37522',
    'P9': '9f0c2daf',
    'P10': '10669016',
    'P11': '3e96d15e',
    'P12': '6ab21388',
    'P13': 'ec4709c1',
    'P14': '256e52ab',
    'P15': '07e4e216',
    'P16': '705f793e'
}

def replace_p_references(text):
    """Replace P references with paper IDs, starting with largest numbers first"""
    if not isinstance(text, str):
        return text
    
    # Sort keys by number in descending order (P16, P15, ..., P1)
    sorted_keys = sorted(p_to_id.keys(), key=lambda x: int(x[1:]), reverse=True)
    
    result = text
    for p_ref in sorted_keys:
        # Use word boundaries to ensure we only match complete P references
        pattern = r'\b' + re.escape(p_ref) + r'\b'
        result = re.sub(pattern, p_to_id[p_ref], result)
    
    return result

def transform_taxonomy(node):
    """Recursively transform P references in taxonomy tree"""
    if isinstance(node, dict):
        # Transform string fields
        for key in ['content', 'name']:
            if key in node and isinstance(node[key], str):
                node[key] = replace_p_references(node[key])
        
        # Transform index field
        if 'index' in node:
            if isinstance(node['index'], list):
                node['index'] = [replace_p_references(p) if isinstance(p, str) else p 
                                for p in node['index']]
            elif isinstance(node['index'], str):
                node['index'] = replace_p_references(node['index'])
        
        # Recursively process children
        if 'children' in node and isinstance(node['children'], list):
            for child in node['children']:
                transform_taxonomy(child)
    
    return node

# Load and transform the data
with open('output/method_taxonomy.json', 'r') as f:
    data = json.load(f)

# Transform the taxonomy
data['taxonomy'] = transform_taxonomy(data['taxonomy'])

# Also transform the pros_cons_analysis if needed
if 'pros_cons_analysis' in data and isinstance(data['pros_cons_analysis'], str):
    data['pros_cons_analysis'] = replace_p_references(data['pros_cons_analysis'])

# Save the result
with open('output/data_with_paper_ids.json', 'w') as f:
    json.dump(data, f, indent=2)

print("Transformation complete!")