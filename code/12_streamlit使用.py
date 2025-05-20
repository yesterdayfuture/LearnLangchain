
# streamlit:一个用于快速创建数据驱动的web应用的python库
# pip install streamlit

from langchain_community.chat_models import ChatOpenAI, ChatOllama
from langchain_community.llms.ollama import Ollama
from langchain.prompts import PromptTemplate

local_llm = ChatOllama(
    model="llama3.2:3b"
)

local_llm2 = Ollama(
    model="llama3.2:3b",
    
)

template = """
你叫小一一，是张先生的助理，是一个万能助手，可以回答用户各类问题。
用户问题：{user_input}
"""


prompt = PromptTemplate(
    input_variables=["user_input"],
    template=template
)

chains = prompt | local_llm2

# response = chains.invoke("你好，你是谁？")
# print(response)


# streamlit:一个用于快速创建数据驱动的web应用的python库
import streamlit as st
import logging
import time

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
    )

# 标题
st.title("LangChain 示例")

# 页面中，下方的输入框
# st.chat_input("请输入您的问题：", key="user_input")

#将输入的内容 展示 在页面上
# a:= ：将输入的内容赋值给a
if a:= st.chat_input("请输入您的问题：", key="user_input"):

    #无区分输出
    # st.write(a)

    # 区分输出
    # st.chat_message("user").write(a)
    # st.chat_message("assistant").write("response")
    # st.chat_message("ai").write("ai")
    # st.chat_message("human").write("human")
    
    logging.warning(f"用户输入：{a}")
    st.chat_message("human").write(a)

    response = chains.invoke(a)

    st.chat_message("assistant").write(response)
    logging.info(f"大模型回复：{response}")



