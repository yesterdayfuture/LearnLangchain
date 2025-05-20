
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    api_key="sk-3e8d696b9b3a403caa30db95c801f875", 
    model="qwen-plus", 
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    # api_base="https://dashscope.aliyuncs.com/compatible-mode/v1/chat",
    temperature=0.1
)

result = llm.invoke("你是谁？")
print(result.content)