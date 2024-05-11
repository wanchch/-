from typing import List, Optional, Tuple, Dict

from neo4j import GraphDatabase

from logger import logging


class Neo4jDatabase:
    def __init__(self, host: str = "neo4j://localhost:7687",
                 user: str = "neo4j",
                 password: str = "pleaseletmein"):

        self.driver = GraphDatabase.driver(host, auth=(user, password))

    def query(self, cypher_query: str, params: Optional[Dict] = {}) -> List[Dict[str, str]]:
        logging.debug(cypher_query)
        with self.driver.session() as session:
            result = session.run(cypher_query, params)
            return set([r.values()[0] for r in result][:50])


if __name__ == "__main__":
    database = Neo4jDatabase(host="neo4j://localhost:7687",
                             user="neo4j", password="wcc771106")

    a = database.query("""MATCH (n:Disease) return n.name limit 25""")

    print(a)
