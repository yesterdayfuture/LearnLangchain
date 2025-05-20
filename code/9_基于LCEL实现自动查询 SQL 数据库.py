
from langchain_community.chat_models import ChatOllama
from langchain.embeddings import OllamaEmbeddings

local_llm = ChatOllama(
    model="llama3.2:3b"
)

local_embedding = OllamaEmbeddings(
    model="mxbai-embed-large:335m"
)


database_url = "mysql+pymysql://root:12345678@localhost:3306/rag_test"

from langchain_community.utilities import SQLDatabase

db = SQLDatabase.from_uri(database_url)

#获取表信息, _代表占位符
def get_database_info(_):
    return db.get_table_info()

#执行 sql 语句
def query_database(query: str):
    return db.run(query)

#测试
print(get_database_info("占位符"))
print(query_database("select * from city_stats"))

from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough,RunnableParallel

template_sql="请通过写sql代码来回答对应问题，并且需要基于如下数据库信息：{info} \n 需要回答问题是：{question} \n 注意仅需要通过sql代码回答，不需要文字 \n 代码形式如下: ```sql\n...\n```"
t_sql=PromptTemplate.from_template(template_sql)

#获取 大模型生成的 sql 语句，并进行提取
def get_sql(x):
    return x.split("```sql")[1].split("```")[0]

chain_sql=({"info":get_database_info,"question":RunnablePassthrough()}| t_sql | local_llm | StrOutputParser() | RunnableLambda(get_sql))


template_sql0="请通过综合如下的数据库信息，问题，sql代码，sql代码的执行结果给出问题的自然语言回答：\n 数据库信息{info} \n 需要回答问题是：{question} \n sql代码：{query} \n sql代码执行结果: {res}"
t_sql0=PromptTemplate.from_template(template_sql0)

#完整流程,获取数据库信息，生成sql语句，执行sql语句，获取结果，生成回答
chain_sql0=( {"info":get_database_info,"question":RunnablePassthrough(),"query":chain_sql}
            |RunnablePassthrough.assign(res=lambda x: query_database(x["query"])) 
            | t_sql0 | local_llm |StrOutputParser())

#执行完整流程
response = chain_sql0.invoke("请告诉我数据库中人口最多的两个城市是哪些，分别是多少人？")
print(response)
