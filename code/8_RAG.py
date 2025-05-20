

from langchain_community.chat_models import ChatOllama

from langchain_community.embeddings import OllamaEmbeddings

llm = ChatOllama(
    model="llama3.2:3b"
)

# print(llm.invoke("你是谁？"))


from langchain.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma

#加载文档
documents = PyPDFLoader("/Users/zhangtian/Downloads/大模型学习路径.pdf").load()

#打印 文档长度
print(len(documents))

#文档类型
print(type(documents[0]))

#文档内容
print(documents[0].page_content)

#文档元数据
print(documents[0].metadata)


#文档分割器
from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

texts = text_splitter.split_documents(documents)

#打印分割后的文档 块数
print(len(texts))

#打印分割后的文档内容
print(texts[0].page_content)
#打印分割后的文档元数据
print(texts[0].metadata)



#定义向量化 模型
llm_embedding = OllamaEmbeddings(model="mxbai-embed-large:335m")

#向量化查询
print(len(llm_embedding.embed_query("你好")))
'''
#第一次使用向量数据库，存入文档向量
#使用向量存储
chroma_client = Chroma.from_documents(documents=texts, embedding=llm_embedding, persist_directory="./ceshiStore")

#持久化
chroma_client.persist()

'''

#加载持久化向量存储
chroma_client = Chroma(persist_directory="./ceshiStore", embedding_function=llm_embedding)

#查询
# print(chroma_client.similarity_search("你好"))


#打印向量存储 集合 数量
print(chroma_client._collection.count())



'''
构造检索式问答链：stuff、refine、map reduce、map re-rank
'''

'''
stuff
'''
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain.prompts import ChatPromptTemplate

#构造检索式问答链
prompt = ChatPromptTemplate.from_template(
    "你是一个搜索引擎，请根据以下文档回答问题：\n\n{context}\n\n问题：{input}"
)

stuff_chain = create_stuff_documents_chain(
    llm=llm,
    prompt=prompt,
    document_variable_name="context"
)

#向量库 转为检索引擎
retrieval = chroma_client.as_retriever()

retrieval_chain = create_retrieval_chain(
    # llm=llm,
    combine_docs_chain=stuff_chain,
    retriever=retrieval,
    # prompt=prompt,
    # document_variable_name="context"
)

# response = retrieval_chain.invoke({"input": "llm是什么"})
# print(response["answer"])


'''
对话式问答链
'''
from langchain.memory import ConversationSummaryMemory,ConversationBufferMemory
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain

#对话式问答链
memory = ConversationSummaryMemory(llm=llm, memory_key="chat_history",return_messages=True)
conversation_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retrieval,
    memory=memory,
    verbose=True
)

inputs_list = [
    "llm是什么",
    "然语言处理是什么",
    "大的语言模型是什么"
]

for input in inputs_list:
    response = conversation_chain.invoke({"question": input})
    print(response["answer"])
    print("============================")

print(conversation_chain.memory.chat_history.messages)

