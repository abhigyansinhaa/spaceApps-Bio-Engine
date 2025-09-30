# Workaround for Python 3.13 compatibility
import socket
if not hasattr(socket, 'EAI_ADDRFAMILY'):
    socket.EAI_ADDRFAMILY = -9  # Add missing constant

from neo4j import GraphDatabase

uri = "bolt://localhost:7687"  # Default URI
username = "neo4j"
password = "bioengine911"     # Your password

driver = GraphDatabase.driver(uri, auth=(username, password))

with driver.session() as session:
    result = session.run("RETURN 'Hello, Neo4j!' AS msg")
    print(result.single()["msg"])