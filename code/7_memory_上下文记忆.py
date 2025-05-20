

from langchain_community.chat_models import ChatOllama, ChatOpenAI

llm1 = ChatOllama(
    model="llama3.2:3b"
)

from langchain.memory import ConversationBufferWindowMemory
from langchain.chains.conversation.base import ConversationChain

memory = ConversationBufferWindowMemory(k=3,return_messages=True)

conversation = ConversationChain(
    llm=llm1,
    memory=memory,
    verbose=True,
)

# 1. 上下文记忆
# 2. 上下文记忆的长度
# 3. 上下文记忆的格式
# 4. 上下文记忆的存储方式
# 5. 上下文记忆的更新方式
# 6. 

text_list = [
    "你好",
    "你是谁",
    # "自然语言处理如何进行学习"
]

for text in text_list:
    print(conversation.predict(input=text))

#上下文记忆的查询方式
print(conversation.memory.chat_memory.messages)


from langchain.schema import messages_to_dict, messages_from_dict

dict_memory = messages_to_dict(conversation.memory.chat_memory.messages)
print("---------------dict_memory---------------")
print(dict_memory)


#上下文记忆存入本地文件
import pickle
with open("memory.pkl", "wb") as f:
    pickle.dump(dict_memory, f)
print("上下文记忆已存入本地文件")


#清除上下文记忆
conversation.memory.clear()
print("上下文记忆已清除")
print(conversation.memory.chat_memory.messages)


#上下文记忆从本地文件读取
with open("memory.pkl", "rb") as f:
    dict_memory2 = pickle.load(f)
print(dict_memory2)
print("上下文记忆已从本地文件读取")

messages = messages_from_dict(dict_memory2)
print("---------------messages---------------")
# print(messages)

#上下文记忆的加载方式
conversation.memory.chat_memory.messages = messages
print(conversation.memory.chat_memory.messages)


# #上下文记忆的更新方式
# conversation.memory.buffer.append_message(messages[0])
# print(conversation.memory.buffer)
