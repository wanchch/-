
import os
import logging
from flask import Flask
from flask import render_template
from flask import request
from flask import redirect
from flask import url_for

from agent import FinAgent
from database import Neo4jDatabase
from run import get_result_and_thought_using_graph

app = Flask(__name__)

neo4j_host = "neo4j://localhost:7687"
neo4j_user = "neo4j"
neo4j_password = "wcc771106"
model_name = 'gpt-3.5-turbo'
modelpath = ""

fin_graph = Neo4jDatabase(
    host=neo4j_host, user=neo4j_user, password=neo4j_password)
agent_fin = FinAgent.initialize(
    fin_graph=fin_graph, model_name=model_name,modelpath=modelpath)



@app.route('/')
def hello_world():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    data = request.get_json()
    search = data['search']

    res = get_result_and_thought_using_graph(agent_fin, search)
    return {
        "code": 200,
        "data": {
            "search": search,
            "answer": res['response'],
            "tags": [],
        },
    }


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=6006,debug=True)
