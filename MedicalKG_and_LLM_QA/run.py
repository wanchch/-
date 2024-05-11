import contextlib
import io
from typing import Dict

from logger import logging


def get_result_and_thought_using_graph(langchain_object, message: str,):
    try:
        if hasattr(langchain_object, "verbose"):
            langchain_object.verbose = True
        chat_input = None
        memory_key = ""
        if hasattr(langchain_object, "memory") and langchain_object.memory is not None:
            memory_key = langchain_object.memory.memory_key

        for key in langchain_object.input_keys:
            if key not in [memory_key, "chat_history"]:
                chat_input = {key: message}

        with io.StringIO() as output_buffer, contextlib.redirect_stdout(output_buffer):
            try:
                output = langchain_object(chat_input)
            except ValueError as exc:
                # make the error message more informative
                logging.debug(f"Error: {str(exc)}")
                output = langchain_object.run(chat_input)
            thought = output_buffer.getvalue().strip()
            logging.info(thought)
    
    except Exception as exc:
        raise ValueError(f"Error: {str(exc)}") from exc

    return {"response": output["output"], "thought": thought}

if __name__ == "__main__":
    neo4j_host = "neo4j://localhost:7687"
    neo4j_user = "neo4j"
    neo4j_password = "wcc771106"
    model_name = 'gpt-3.5-turbo'
    modelpath = ""
    from agent import FinAgent
    from database import Neo4jDatabase
    fin_graph = Neo4jDatabase(host=neo4j_host, user=neo4j_user, password=neo4j_password)
    agent_fin = FinAgent.initialize(fin_graph=fin_graph, model_name=model_name,modelpath=modelpath,handle_parsing_errors=True)
    # while True:
    #     ques = input('用户：')
    #     result = get_result_and_thought_using_graph(langchain_object=agent_fin,message=ques)
    #     print(result['response'])
    ques = '急性肺炎有什么症状？'
    output = get_result_and_thought_using_graph(langchain_object=agent_fin,message=ques)
    print(output['response'])