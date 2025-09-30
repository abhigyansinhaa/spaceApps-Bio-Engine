# export_to_neo4j.py
import json
import socket
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Python 3.13 compatibility fix
if not hasattr(socket, 'EAI_ADDRFAMILY'):
    socket.EAI_ADDRFAMILY = -9

from neo4j import GraphDatabase
from tqdm import tqdm

# ---------- CONFIG ----------
TRIPLES_FILE = "kg_triples_validated.json"
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
BATCH_SIZE = 1000  # Process in batches for better performance
# ----------------------------

# Validate password is set
if not NEO4J_PASSWORD:
    raise ValueError(
        "NEO4J_PASSWORD not found in environment variables.\n"
        "Please create a .env file with:\n"
        "NEO4J_URI=bolt://localhost:7687\n"
        "NEO4J_USER=neo4j\n"
        "NEO4J_PASSWORD=your_password"
    )

class Neo4jExporter:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
    
    def close(self):
        self.driver.close()
    
    def create_constraints(self):
        """Create constraints for better performance"""
        with self.driver.session() as session:
            try:
                # Create uniqueness constraint on Entity.name
                session.run("CREATE CONSTRAINT entity_name IF NOT EXISTS FOR (e:Entity) REQUIRE e.name IS UNIQUE")
                print("Constraint created successfully")
            except Exception as e:
                print(f"Constraint creation error (may already exist): {e}")
    
    def create_triple_batch(self, tx, triples_batch):
        """Create multiple triples in a single transaction"""
        query = """
        UNWIND $triples AS triple
        MERGE (s:Entity {name: triple.subject})
        MERGE (o:Entity {name: triple.object})
        MERGE (s)-[r:RELATION {type: triple.predicate}]->(o)
        SET r.title = triple.title,
            r.chunk_id = triple.chunk_id
        """
        tx.run(query, triples=triples_batch)
    
    def export_triples(self, triples):
        """Export triples to Neo4j in batches"""
        # Prepare batch data
        valid_triples = []
        for t in triples:
            subj = t.get("subject", "").strip()
            pred = t.get("predicate", "").strip()
            obj = t.get("object", "").strip()
            
            if subj and pred and obj:
                valid_triples.append({
                    "subject": subj,
                    "predicate": pred,
                    "object": obj,
                    "title": t.get("title", ""),
                    "chunk_id": t.get("chunk_id", "")
                })
        
        print(f"Exporting {len(valid_triples)} valid triples to Neo4j...")
        
        # Process in batches
        with self.driver.session() as session:
            for i in tqdm(range(0, len(valid_triples), BATCH_SIZE), desc="Exporting batches"):
                batch = valid_triples[i:i + BATCH_SIZE]
                session.execute_write(self.create_triple_batch, batch)
        
        print(f"Export complete! {len(valid_triples)} triples added to Neo4j")
    
    def get_statistics(self):
        """Get statistics about the graph"""
        with self.driver.session() as session:
            # Count nodes
            node_count = session.run("MATCH (n:Entity) RETURN count(n) as count").single()["count"]
            
            # Count relationships
            rel_count = session.run("MATCH ()-[r:RELATION]->() RETURN count(r) as count").single()["count"]
            
            # Get relationship types
            rel_types = session.run("""
                MATCH ()-[r:RELATION]->()
                RETURN r.type as type, count(*) as count
                ORDER BY count DESC
                LIMIT 10
            """).data()
            
            print("\n" + "="*60)
            print("NEO4J GRAPH STATISTICS")
            print("="*60)
            print(f"Total entities (nodes): {node_count}")
            print(f"Total relationships: {rel_count}")
            print("\nTop 10 relationship types:")
            for item in rel_types:
                print(f"  {item['type']}: {item['count']}")
            print("="*60)

def load_triples(file_path):
    """Load triples from JSON file"""
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

def test_connection(uri, user, password):
    """Test Neo4j connection"""
    try:
        driver = GraphDatabase.driver(uri, auth=(user, password))
        with driver.session() as session:
            result = session.run("RETURN 1 as test")
            result.single()
        driver.close()
        print("Neo4j connection successful!")
        return True
    except Exception as e:
        print(f"Neo4j connection failed: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure Neo4j is running")
        print("2. Check URI (default: bolt://localhost:7687)")
        print("3. Verify username and password")
        print("4. Install neo4j driver: pip install neo4j")
        return False

if __name__ == "__main__":
    print("="*60)
    print("EXPORTING KNOWLEDGE GRAPH TO NEO4J")
    print("="*60)
    
    # Test connection first
    print("\nTesting Neo4j connection...")
    if not test_connection(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD):
        print("\nPlease fix the connection issues and try again.")
        exit(1)
    
    # Load triples
    print(f"\nLoading triples from {TRIPLES_FILE}...")
    triples = load_triples(TRIPLES_FILE)
    print(f"Loaded {len(triples)} triples")
    
    # Export to Neo4j
    exporter = Neo4jExporter(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    
    try:
        # Create constraints for performance
        print("\nCreating database constraints...")
        exporter.create_constraints()
        
        # Export triples
        print("\nExporting triples...")
        exporter.export_triples(triples)
        
        # Show statistics
        exporter.get_statistics()
        
        print("\n" + "="*60)
        print("SUCCESS!")
        print("="*60)
        print("Open Neo4j Browser at: http://localhost:7474")
        print("\nSample Cypher queries to try:")
        print("1. View all nodes and relationships:")
        print("   MATCH (n)-[r]->(m) RETURN n,r,m LIMIT 50")
        print("\n2. Find all relationships for 'microgravity':")
        print("   MATCH (n:Entity {name: 'microgravity'})-[r]->(m) RETURN n,r,m")
        print("\n3. Find paths between two entities:")
        print("   MATCH p=(a:Entity {name: 'microgravity'})-[*1..3]-(b:Entity {name: 'bone loss'})")
        print("   RETURN p LIMIT 5")
        
    except Exception as e:
        print(f"\nError during export: {e}")
    finally:
        exporter.close()
