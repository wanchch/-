from database import Neo4jDatabase
from pydantic import BaseModel, Extra
from langchain.prompts.base import BasePromptTemplate
from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
from langchain.chains.base import Chain

from langchain.memory import ReadOnlySharedMemory
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from typing import Dict, List, Any
from logger import logging

examples = """
# 小儿肺炎有什么症状？
MATCH (n:Disease) -[r:has_symptom]->(m) where n.name=‘小儿肺炎’ return m as result
# 肝病不能吃什么？
MATCH (m:Disease) -[r:avoid_food]->(n)  where m.name='肝病' return n as result
# 怎么样才能防止肾虚？
MATCH (n:Disease) where n.name ='肾虚' return n.prevent as result
# 痔疮有什么并发症？
MATCH (n:Disease)-[r:has_complication]->(m) where n.name='痔疮' return m.name as result
# 感冒的护理方式？
MATCH (n:Disease) where n.name='感冒' return n.nursing as result
# 最近老是流鼻涕是因为什么？
MATCH (n:Disease)-[r:has_symptom]->(m) 
WHERE m.name='流鼻涕'
RETURN n.name as result
# 如果我得了肺炎，我会有什么症状？我不应该吃什么食物？
MATCH (n:Disease)-[r:avoid_food | has_symptom]->(m) where n.name='肺炎' return m as result
# 阿莫西林胶囊能治什么病？
MATCH (n:Disease) -[r:common_drug]->(m) where m.name='阿莫西林胶囊' return n.name as result
"""

SYSTEM_TEMPLATE = """
你是一个具有根据示例Cypher查询生成Cypher查询能力的助手。
目前知识图谱存在的实体类型是：【Disease,Department,Check,Drug,Food,Symptom】
关系类型是：【belongs_to,common_drug,good_food,avoid_food,check_item,recommand_recipes,has_complication,has_symptom】
Disease的属性是：【name,desc,cause,prevent,treat_cycle,treat_way,
cure_prob,susceptible_people,medical_insurance,transmission_way,treat_cost,nursing】
示例Cypher查询是：\n"""+examples+"""\n
除了Cypher查询，不要回应任何解释或任何其他信息。
你永远不要道歉，并严格根据提供的Cypher示例生成cypher语句。
不要提供任何无法从Cypher示例中推断出的Cypher语句。
当你无法由于对话的上下文缺乏而推断出cypher语句时，告知用户并说明缺少的上下文是什么。
输入:{question}
"""


prompt = '''请将输入转换为neo4j的Cypher查询语句。只返回Cypher查询语句，不要回应任何解释或任何其他信息。
目前知识图谱存在的实体类型是：【Disease,Department,Check,Drug,Food,Symptom】
关系类型是：【belongs_to,common_drug,good_food,avoid_food,check_item,recommand_recipes,has_complication,has_symptom】
Disease的属性是：【name,desc,cause,prevent,treat_cycle,treat_way,cure_prob,susceptible_people,medical_insurance,transmission_way,treat_cost,nursing】
如输入：小儿肺炎有什么症状？
输出：MATCH (n:Disease {name:"小儿肺炎"}) return n.name,n.desc as result
输入：{question}
输出：'''

SYSTEM_CYPHER_PROMPT = SystemMessagePromptTemplate.from_template(SYSTEM_TEMPLATE) #创建一个系统消息提示模板

HUMAN_TEMPLATE = "{inputs}"
HUMAN_PROMPT = HumanMessagePromptTemplate.from_template(HUMAN_TEMPLATE) #创建了一个人类消息提示模板

class LLMCypherGraphChain(Chain):
    """Chain that interprets a prompt and executes python code to do math.
    """
    llm: Any
    """LLM wrapper to use."""
    system_prompt: BasePromptTemplate = SYSTEM_CYPHER_PROMPT
    human_prompt: BasePromptTemplate = HUMAN_PROMPT
    input_key: str = "question"  #question为默认值
    output_key: str = "answer"  #answer为默认值
    graph: Neo4jDatabase
    memory: ReadOnlySharedMemory
    
    #定义了一个内部的 Config 类，用于配置 Pydantic 对象的行为。
    # Pydantic 是一个 Python 库，用于数据验证和数据序列化，通常在定义数据模型时用到
    class Config:
        """Configuration for this pydantic object."""
        extra = 'allow'
        arbitrary_types_allowed = True

    #定义了一个属性 input_keys，它是一个只读的属性，使用 @property 装饰器来标记。
    # 这意味着 input_keys 是一个通过方法调用而不是直接访问属性值来获取的属性。
    @property
    def input_keys(self) -> List[str]:
        """Expect input key.
        :meta private:
        """
        return [self.input_key]

    @property
    def output_keys(self) -> List[str]:
        """Expect output key.
        :meta private:
        """
        return [self.output_key]

    def _call(self, inputs: Dict[str, str]) -> Dict[str, str]:
        logging.debug(f"Cypher generator inputs: {inputs}")
        chat_prompt =  PromptTemplate(input_variables=["question"], template=SYSTEM_TEMPLATE)
        cypher_executor = LLMChain(prompt=chat_prompt, llm=self.llm, callback_manager=self.callback_manager)
        cypher_statement = cypher_executor.predict(question=inputs[self.input_key], stop=["Output:"])
        print(cypher_statement)
        if not "MATCH" in cypher_statement:
            return {'answer': 'Missing context to create a Cypher statement'}
        context = self.graph.query(cypher_statement)
        logging.debug(f"Cypher generator context: {context}")
        print(context)

        return {'answer': context}


if __name__ == "__main__":
    from langchain_openai import ChatOpenAI
    with open('key.txt') as f:
        key = f.read()

    llm = ChatOpenAI(temperature=0.3, model_name="gpt-3.5-turbo", openai_api_key=key)

    
    database = Neo4jDatabase(host="neo4j://localhost:7687", user="neo4j", password="wcc771106")
    
    from langchain.memory import ConversationBufferMemory
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    readonlymemory = ReadOnlySharedMemory(memory=memory)
    chain = LLMCypherGraphChain(llm=llm, verbose=True, graph=database,memory = readonlymemory)
    # while True:
    #     question = input("问题：")
    #     output = chain.invoke(
    #         question
    #     )
    #     print(output)
    question = '肺炎有什么症状？'
    output = chain.invoke(question)
    print(output)