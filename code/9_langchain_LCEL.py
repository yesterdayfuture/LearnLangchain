'''
langchain 表达式语言 LCEL
自定义实现链
'''



from langchain_community.chat_models import ChatOllama,ChatOpenAI

local_llm = ChatOllama(
    model="llama3.2:3b"
)
# print(local_llm.invoke("你好"))

online_llm = ChatOpenAI(
    api_key="sk-8f299497aaa74c64ad2899c85c2dcaa5",
    model="deepseek-chat",
    base_url="https://api.deepseek.com/v1"
)

# print(online_llm.invoke("你好"))

'''
自定义简单链 prompt + model + output parse
'''
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


prompt = PromptTemplate.from_template(
    "请帮我写一首关于{topic}的诗"
)

output_parser = StrOutputParser()

custom_chains1 = prompt | local_llm | output_parser

# print(custom_chains1.invoke("春天"))


'''
自定义简单 RAG链
'''

from langchain_community.vectorstores import Chroma
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_community.embeddings import OllamaEmbeddings

#定义向量化 模型
llm_embedding = OllamaEmbeddings(model="mxbai-embed-large:335m")

prompt2 = PromptTemplate.from_template(
    "请根据以下内容回答问题：\n\n{content}\n\n问题：{question}\n\n答案："
)

vectors = Chroma.from_texts(
    texts=["小明在百度工作"],
    embedding=llm_embedding
)

retriveal = vectors.as_retriever()

'''
 | 符号前后都可以看作一个 Runnable

RunnableParallel 的作用是 将输入组成一个字典，然后传递给 prompt
有三种等价形势：
1、RunnableParallel({"content":retriveal, "question": RunnablePassthrough()})
2、{"content":retriveal, "question": RunnablePassthrough()}
3、RunnableParallel(content=retriveal, question=RunnablePassthrough())
'''

custom_chains2 = {"content":retriveal, "question": RunnablePassthrough()} | prompt2 | local_llm | output_parser

# print(custom_chains2.invoke("小明在做什么"))


'''
多变量 RAG 链
'''

from operator import itemgetter

prompt3 = PromptTemplate.from_template(
    "请根据以下内容回答问题：\n\n{content}\n\n问题：{question}\n\n使用{language}回答\n\n答案："
)

# custom_chains3 = {"content": itemgetter("question") | retriveal, "question": itemgetter("question") , "language": itemgetter("language")} | prompt3 | local_llm | output_parser

# print(custom_chains3.invoke({"question": "小明在做什么", "language": "英文"}))

#使用 Runnable 实现


custom_chains4 = RunnableParallel(
    content=retriveal,
    question=lambda X: X["question"],
    language=lambda X: X["language"]
) | prompt3 | local_llm | output_parser

print(custom_chains4.invoke({"question": "小明在做什么", "language": "英文"}))

#测试多变量 能否 正常传递到 prompt提示模版中
# ceshi = RunnableParallel(
#     content=retriveal,
#     question=lambda X: X["question"],
#     language=lambda X: X["language"]
# )| prompt3
# print(ceshi.invoke({"question": "小明在做什么", "language": "英文"}))

'''
RunnableLambda:将一个函数变成一个可运行的链，输出一个Runnable
RunnableParallel:将多个Runnable并行运行，输出一个Runnable
用例：
RunnableLambda(lambda X: X["question"]) | retriveal
{"content":retriveal, "question": lambda X: X["question"] | RunnableLambda(function)} | prompt3

'''










