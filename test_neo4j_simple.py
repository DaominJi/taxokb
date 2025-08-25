#!/usr/bin/env python3
"""
Simple Neo4j connection test without APOC dependencies.
This script tests the basic Neo4j connection and queries without requiring APOC procedures.
"""

from neo4j import GraphDatabase
import sys

def test_neo4j_connection():
    """Test basic Neo4j connection and run simple queries"""
    
    # Connection parameters
    uri = "bolt://localhost:7687"
    username = "neo4j"
    password = "xiaodaomin517"
    
    print(f"Testing Neo4j connection to {uri}...")
    
    try:
        # Create driver
        driver = GraphDatabase.driver(uri, auth=(username, password))
        
        # Test basic connection
        with driver.session() as session:
            result = session.run("RETURN 1 as test_value")
            test_value = result.single()["test_value"]
            
            if test_value == 1:
                print("✅ Basic connection successful!")
            else:
                print("❌ Connection test failed - unexpected result")
                return False
        
        # Test database content
        with driver.session() as session:
            # Check if there are any nodes
            result = session.run("MATCH (n) RETURN count(n) as node_count")
            node_count = result.single()["node_count"]
            print(f"📊 Database contains {node_count} nodes")
            
            # Check node types
            result = session.run("MATCH (n) RETURN DISTINCT labels(n) as labels LIMIT 10")
            labels = [record["labels"] for record in result]
            print(f"📋 Node types found: {labels}")
            
            # Test a simple query on our data
            if node_count > 0:
                result = session.run("""
                MATCH (p:Paper) 
                RETURN p.title as title, p.paper_id as id 
                LIMIT 3
                """)
                
                papers = [{"title": record["title"], "id": record["id"]} for record in result]
                if papers:
                    print("📚 Sample papers found:")
                    for paper in papers:
                        print(f"  - {paper['title']} (ID: {paper['id']})")
                else:
                    print("ℹ️  No papers found in database")
        
        driver.close()
        print("✅ All tests passed! Neo4j is working correctly.")
        return True
        
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        print("\n🔧 Troubleshooting tips:")
        print("1. Make sure Neo4j is running: `neo4j start` or `brew services start neo4j`")
        print("2. Check if the password is correct")
        print("3. Verify Neo4j is listening on bolt://localhost:7687")
        return False

if __name__ == "__main__":
    success = test_neo4j_connection()
    sys.exit(0 if success else 1)


