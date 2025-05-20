

import os
#Literal 是一个特殊的类型提示，用于表示一组固定的字符串值
from typing import Literal,Type
#MemorySaver: 用来保存内存中的数据，比如保存用户输入的prompt，保存工具的返回结果等
from langgraph.checkpoint.memory import MemorySaver
from langchain_huggingface import HuggingFacePipeline
from langchain_openai import ChatOpenAI
# from langchain_community.chat_models import ChatOllama,ChatOpenAI
from langgraph.graph import END, StateGraph,MessagesState
from langchain_core.messages import HumanMessage, AIMessage
from pydantic import BaseModel,Field
from langchain.tools import BaseTool
from langgraph.prebuilt import ToolNode
from langchain_community.agent_toolkits import load_tools
from langchain.agents import AgentType, initialize_agent

from langchain_experimental.llms.ollama_functions import OllamaFunctions

from langchain_openai import ChatOpenAI

import uvicorn
from fastapi import FastAPI,WebSocket

app =FastAPI(version="0.1.0",title="Langchain",description="Langchain API")

online_llm = ChatOpenAI(
    api_key="sk-8f299497aaa74c64ad2899c85c2dcaa5",
    model="deepseek-chat",
    base_url="https://api.deepseek.com/v1"
)



'''
造车机器人，并且这个车需要与某个账户绑定，只能这个账户才能使用这个车
'''

class CreateCarInput(BaseModel):
    engine: str = Field(description="发动机信息")
    chassis: str = Field(description="底盘信息")
    transmission: str = Field(description="变速箱信息")


class CreateCar(BaseTool):

    name:str = Field(default="create_car")

    description:str = Field(default="这是一个生成车的信息的方法，需要用户提供发动机、底盘、变速箱信息，才能进行造车；" \
    "如果用户没有提供发动机、底盘、变速箱信息，或者缺少某些信息，则提示用户提供对应的信息，" \
    "直到信息完整，才能进行造车，并把造车的信息返回给用户")

    args_schema : Type[BaseModel] = CreateCarInput

    def _run(self, engine: str, chassis: str, transmission: str) -> str:
        print("同步造车方法")
        return f"engine: {engine}, chassis: {chassis}, transmission: {transmission}"

    async def _arun(self, engine: str, chassis: str, transmission: str) -> str:
        print("异步造车方法")
        return f"engine: {engine}, chassis: {chassis}, transmission: {transmission}"


# 创建账户
class CreateAccountInput(BaseModel):
    account_name: str = Field(description="账户名称")

class CreateAccount(BaseTool):

    name:str = Field(default="create_account")

    description:str = Field(default="创建账户，需要用户提供账户名称，才能创建账户；" \
    "如果用户没有提供账户名称，则提示用户提供账户名称，" \
    "直到信息完整，才能进行创建账户，并把创建账户的信息返回给用户")

    args_schema : Type[BaseModel] = CreateAccountInput

    def _run(self, account_name: str) -> str:
        print("同步创建账户方法")
        return f"account_name: {account_name}"

    async def _arun(self, account_name: str) -> str:
        print("异步创建账户方法")
        return f"account_name: {account_name}"



#绑定账户和车辆
class BindCarAccountInput(BaseModel):
    account_name: str = Field(description="账户名称")
    car_name: str = Field(description="车信息")

class BindCarAccount(BaseTool):

    name:str = Field(default="bind_car_account")

    description:str = Field(default="绑定账户和车辆，需要用户提供账户名称和车信息，才能进行绑定；" \
    "如果用户没有提供账户名称或车信息，则提示用户提供账户名称或车信息，" \
    "直到信息完整，才能进行绑定，并把绑定信息返回给用户")

    args_schema : Type[BaseModel] = BindCarAccountInput
    
    def _run(self, account_name: str, car_name: str) -> str:
        print("同步绑定账户和车辆方法")
        return f"account_name: {account_name}, car_name: {car_name}"

    async def _arun(self, account_name: str, car_name: str) -> str:
        print("异步绑定账户和车辆方法")
        return f"account_name: {account_name}, car_name: {car_name}"


class langGraphLearn(object):

    def __init__(self):
        
        #工具列表
        self.tools = [
            CreateCar(),
            CreateAccount(),
            BindCarAccount()
        ]

        #将 tool工具 转为 工具字典
        self.tools_node = ToolNode(self.tools)

        self.model = online_llm.bind_tools(self.tools)

        #保存对话历史
        self.messages = []

    #设定状态
    def should_continue(self, state: MessagesState) -> Literal["tools", END]:
        '''
        判断是否需要调用工具，如果需要调用工具，返回 "tools"，否则返回 END
        '''
        messages = state["messages"]
        last_message = messages[-1]

        # 如果 last_message 中有 tool_calls，则返回 "tools"
        if last_message.tool_calls:
            return "tools"
        return END


    #调用大模型
    def call_model(self, state: MessagesState) -> MessagesState:
        '''
        调用大模型，获取回复
        '''
        messages = state["messages"]
        response = self.model.invoke(messages)
        # messages.append(response)
        return {"messages":[response]}


    def __call__(self, query):

        print("=============调用call=============")

        #定义工作流
        workflow = StateGraph(MessagesState)

        #添加节点
        workflow.add_node("agent", self.call_model)
        workflow.add_node("tools", self.tools_node)

        #set_entry_point: 设置入口节点
        workflow.set_entry_point("agent")

        #添加条件边
        #add_conditional_edges: 添加条件边，如果 should_continue 返回 "tools"，则从 "agent" 节点跳转到 "tools" 节点
        # 如果 should_continue 返回 END，则从 "agent" 节点跳转到 END 节点
        workflow.add_conditional_edges("agent", self.should_continue)

        #添加普通边
        #add_edge: 添加普通边，从 "tools" 节点跳转到 "agent" 节点
        workflow.add_edge("tools", "agent")

        #MemorySaver: 保存工作流状态
        checkpoint = MemorySaver()

        #compile: 编译工作流，返回一个可调用的函数
        app = workflow.compile(checkpoint)

        
        #添加用户消息
        self.messages.append(HumanMessage(content=query))

        response = app.invoke({"messages":self.messages}, config={"configurable":{"thread_id":42}})
        
        
        #添加 AI 消息
        self.messages.append(AIMessage(content=response["messages"][-1].content))

        return response["messages"][-1].content



#实例化 langGraphLearn 对象
langGraphLearn = langGraphLearn()



@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    while True:
        data = await websocket.receive_text()

        response = langGraphLearn(data)

        await websocket.send_text(f"大模型回复为：\n {response}" )



if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)


