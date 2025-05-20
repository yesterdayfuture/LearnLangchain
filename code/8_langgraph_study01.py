'''
langchain 
  agent: 通过用户的输入或prompt 来进行调用工具， 比如搜索、调用api等，它可以对用户提出的问题进行编排
  例如问题：问某某人的老公是谁？主持过什么节目
  步骤： 1、先定义工具，比如搜索工具，api工具
        2、agent 去做任务编排，通过任务调用对应tool工具获取答案    
'''

'''
langgraph:  通过图的形式，将多个工具串联起来，形成一个完整的流程

    state: 状态，--用来判断是否进行调用工具或调用结束  表示当前任务的状态，比如搜索中，搜索完成，搜索失败等
    node: 节点，--用来表示工具，比如搜索工具，api工具
    edge: 边，--用来表示节点之间的连接关系，通过tool 指向大模型 或 另外一个tool， 比如搜索工具和api工具之间的连接关系
    graph：图，--用来表示整个流程，通过节点和边来表示整个流程

    状态变换：每次执行完一个节点（tool工具）后，判断是否需要调用其他node，如果需要调用，则将状态设置为调用其他node，否则设置为结束
'''

import os
#Literal 是一个特殊的类型提示，用于表示一组固定的字符串值
from typing import Literal,Type
#MemorySaver: 用来保存内存中的数据，比如保存用户输入的prompt，保存工具的返回结果等
from langgraph.checkpoint.memory import MemorySaver

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



#工具列表
tools = [
    CreateCar(),
    CreateAccount(),
    BindCarAccount()
]


#将 tool工具 转为 工具字典
tools_node = ToolNode(tools)

#定义模型
# local_llm= OllamaFunctions(
#     model="llama3.2:3b",
#     temperature=0.7
# ).bind_tools(tools=tools)

os.environ["DEEPSEEK_API_KEY"] = "sk-8f299497aaa74c64ad2899c85c2dcaa5"

local_llm = ChatOpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com/v1",
    model="deepseek-chat"
).bind_tools(tools=tools)


#设定状态
def should_continue(state: MessagesState) -> Literal["tools", END]:
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
def call_model(state: MessagesState) -> MessagesState:
    '''
    调用大模型，获取回复
    '''
    messages = state["messages"]
    response = local_llm.invoke(messages)
    # messages.append(response)
    return {"messages":[response]}



#定义工作流
workflow = StateGraph(MessagesState)

#添加节点
workflow.add_node("agent", call_model)
workflow.add_node("tools", tools_node)

#set_entry_point: 设置入口节点
workflow.set_entry_point("agent")

#添加条件边
#add_conditional_edges: 添加条件边，如果 should_continue 返回 "tools"，则从 "agent" 节点跳转到 "tools" 节点
# 如果 should_continue 返回 END，则从 "agent" 节点跳转到 END 节点
workflow.add_conditional_edges("agent", should_continue)

#添加普通边
#add_edge: 添加普通边，从 "tools" 节点跳转到 "agent" 节点
workflow.add_edge("tools", "agent")

#MemorySaver: 保存工作流状态
checkpoint = MemorySaver()

#compile: 编译工作流，返回一个可调用的函数
app = workflow.compile(checkpoint)


#多轮对话
messages = []
while True:
    user_input = input("请输入：")
    if user_input == "exit":
        break
    messages.append(HumanMessage(content=user_input))

    response = app.invoke({"messages":messages}, config={"configurable":{"thread_id":42}})
    
    print(response["messages"][-1].content)

    messages.append(AIMessage(content=response["messages"][-1].content))

    



