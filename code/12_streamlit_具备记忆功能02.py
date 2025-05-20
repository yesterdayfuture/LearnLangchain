

from langchain_community.chat_models import ChatOpenAI, ChatOllama
from langchain_community.llms.ollama import Ollama
from langchain.prompts import PromptTemplate,ChatPromptTemplate

from langchain.memory import ConversationBufferMemory
from langchain.chains.conversation.base import ConversationChain
import streamlit as st
import logging
import time


st.title("具备记忆的聊天机器人")


local_llm2 = Ollama(
    model="llama3.2:3b",
    
)

template = """
ai:你叫小一一，是张先生的助理，是一个万能助手，可以回答用户各类问题。
history:{history}
human：{input}
"""


# prompt = PromptTemplate(
#     input_variables=["user_input"],
#     template=template
# )


if "history" not in st.session_state:
    st.session_state["history"] = ConversationBufferMemory()


chains = ConversationChain(
    llm=local_llm2,
    memory= st.session_state["history"],
    prompt=ChatPromptTemplate.from_template(template=template)
)

#展示所有的历史记录
for i in chains.memory.chat_memory.messages:
    st.chat_message(i.type).write(i.content)



if input_text := st.chat_input("请输入你的问题"):

    st.chat_message("human").write(input_text)

    response = chains.invoke(input_text)

    st.chat_message("ai").write(response["response"])

    print(chains.memory.chat_memory.messages)

