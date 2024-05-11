from __future__ import annotations
from database import Neo4jDatabase
from langchain.chains.base import Chain

from typing import Any, Dict, List

from pydantic import Field
from logger import logging
 

fulltext_search = """
CALL db.index.fulltext.queryNodes("name", '%s') 
YIELD node, score
WITH node, score LIMIT 5
CALL {
  WITH node
  MATCH (node)-[r]->(target)
  RETURN coalesce(node.name) + " " + type(r) + " " + coalesce(target.name) AS result
  UNION
  WITH node
  MATCH (node)<-[r]-(target)
  RETURN coalesce(target.name) + " " + type(r) + " " + coalesce(node.name) AS result
}
RETURN result LIMIT 100
"""


def generate_params(input_str):
    """
    Generate full text parameters using the Lucene syntax
    """
    if type(input_str)==list:
        input_str = ''.join(input_str)
    disease_sentences = [disease.strip() for disease in input_str.split(' ')]
    disease_sentence = ['"' + disease + '"' for disease in disease_sentences]
    # transformed_str = ' OR '.join(disease_sentences)
    # Return the transformed string
    # return transformed_str
    return disease_sentence


class LLMKeywordGraphChain(Chain):
    """Chain for keyword question-answering against a graph."""

    graph: Neo4jDatabase = Field(exclude=True)
    input_key: str = "query"  #: :meta private:
    output_key: str = "result"  #: :meta private:

    @property
    def input_keys(self) -> List[str]:
        """Return the input keys.
        :meta private:
        """
        return [self.input_key]

    @property
    def output_keys(self) -> List[str]:
        """Return the output keys.
        :meta private:
        """
        _output_keys = [self.output_key]
        return _output_keys

    def _call(self, inputs: Dict[str, str]) -> Dict[str, Any]:
        """Extract entities, look up info and answer question."""
        question = inputs[self.input_key]
        params = generate_params(question)
        context = []
        # print(f"Keyword search params: {params}")
        # context = self.graph.query(fulltext_search % params)
        # print(f"Keyword search context: {context}")
        for par in params:
            print(f"Keyword search params: {par}")
            context.append(self.graph.query(fulltext_search % par))

        return {self.output_key: context}


if __name__ == '__main__':
    from langchain_openai import ChatOpenAI
    with open('key.txt') as f:
        key = f.read()
    llm = ChatOpenAI(temperature=0.0,model_name="gpt-3.5-turbo",openai_api_key=key)

    database = Neo4jDatabase(host="neo4j://localhost:7687",user="neo4j", password="wcc771106")
    chain = LLMKeywordGraphChain(llm=llm, verbose=True, graph=database)

    while True:
        question = input('问题：')
        output = chain.invoke(question)
        print(output)