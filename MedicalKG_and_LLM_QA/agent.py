from langchain.agents.agent import AgentExecutor
from langchain.agents.tools import Tool
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory, ReadOnlySharedMemory
from langchain.agents import initialize_agent
from langchain.agents import AgentType

from cypher_database_tool import LLMCypherGraphChain
from keyword_neo4j_tool import LLMKeywordGraphChain
from chatglm import GLM


class FinAgent(AgentExecutor):
    """FinKG agent"""

    @staticmethod
    def function_name():
        return "FinAgent"

    @classmethod
    def initialize(cls, fin_graph, model_name, modelpath = '',*args, **kwargs):
        if model_name in ['gpt-3.5-turbo', 'gpt-4']:
            with open('key.txt') as f:
                key = f.read()
            llm = ChatOpenAI(temperature=0, model_name=model_name,
                             openai_api_key=key)
        elif model_name=='ChatGLM':
            llm = GLM()
            llm.temperature = 0.0
            llm.load_model(model_name_or_path = modelpath)
        else:
            raise Exception(f"Model {model_name} is currently not supported")
        
        memory = ConversationBufferMemory(
            memory_key="chat_history", return_messages=True)
        readonlymemory = ReadOnlySharedMemory(memory=memory)

        cypher_tool = LLMCypherGraphChain(llm=llm, graph=fin_graph, verbose=True, memory=readonlymemory)
        fulltext_tool = LLMKeywordGraphChain(llm=llm, graph=fin_graph, verbose=True)

        tools = [
            Tool(
                name="Cypher search",
                func=cypher_tool.run,
                description="""
                    利用此工具在疾病数据库中进行搜索，该数据库专门用于回答与疾病相关的问题。
                    这个专门的工具提供了简化的搜索功能，可以帮助您轻松找到所需的疾病信息。
                    输入应该是完整的问题。
                    请把搜索出的疾病信息全部融入回答之中。
                    最后请用中文来回答问题。""",
            ),
            Tool(
                name="Keyword search",
                func=fulltext_tool.run,
                description="""
                    当明确告知使用关键字搜索时，请使用此工具。
                    输入应该是从问题中推断出的相关疾病的列表。
                    最后请用中文来回答问题。""",
            ),

        ]

        agent_chain = initialize_agent(
            tools, llm, agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION, verbose=True, memory=memory)

        return agent_chain

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def run(self, *args, **kwargs):
        return super().run(*args, **kwargs)
