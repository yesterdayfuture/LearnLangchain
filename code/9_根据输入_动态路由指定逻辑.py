


from langchain_community.chat_models import ChatOllama
from langchain.embeddings import OllamaEmbeddings

local_llm = ChatOllama(
    model="llama3.2:3b"
)

local_embedding = OllamaEmbeddings(
    model="mxbai-embed-large:335m"
)


from langchain.utils.math import cosine_similarity
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough


physics_template = """你是一个非常聪明的物理学教授。
你擅长以简洁易懂的方式回答物理问题。
当你不知道答案时，你会承认不知道。

这是一个问题：
{query}"""

math_template = """你是一个非常优秀的数学家。
你擅长回答数学问题。
你之所以优秀是因为你能够将难题分解为组成部分，
回答组成部分，然后把它们组合起来回答更广泛的问题。

这是一个问题：
{query}"""

#提示词 列表
prompt_list = [physics_template, math_template]

#根据提示词列表，生成embedding向量
embedding_prompt = local_embedding.embed_documents(prompt_list)

#根据输入，动态路由指定逻辑
def prompt_router(input):
    # 根据输入，生成embedding向量
    query_embedding = local_embedding.embed_query(input["type"])

    # 计算输入向量与提示词向量之间的相似度
    similarity = cosine_similarity([query_embedding], embedding_prompt)[0]

    # 找到最相似的提示词
    most_similar = prompt_list[similarity.argmax()]
    print("使用数学" if most_similar == math_template else "使用物理")

    # 根据最相似的提示词，生成对应的提示词模板
    return PromptTemplate.from_template(most_similar)


prompt_router_template = """
你是一个非常聪明的教授。
你擅长分析问题，并根据问题的类型返回当前问题是属于什么学科。
示例；
 问题：1+1=？
 返回：数学
 问题：黑洞是什么
 返回：物理
 问题：DNA是什么
 返回：生物

问题：
{query}

请返回属于哪个学科，不添加任何关于学科的描述。
"""

prompt_router_prompt = PromptTemplate.from_template(prompt_router_template)

chain0 = {"query": RunnablePassthrough()} | prompt_router_prompt | local_llm | StrOutputParser()

#创建一个链,可以根据输入，动态路由指定逻辑
chain = (
    {"query": RunnablePassthrough(), "type":chain0}
    | RunnableLambda(prompt_router)
    | local_llm
    | StrOutputParser()
)

response = chain.invoke("高等数学是什么")

print(response)