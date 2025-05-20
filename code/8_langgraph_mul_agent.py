


from langchain.agents import create_openai_tools_agent
from langchain_openai.chat_models import ChatOpenAI
from pydantic import BaseModel, Field
from langchain_community.tools import BaseTool

from typing import Union, Type

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




online_llm = ChatOpenAI(
    api_key="sk-8f299497aaa74c64ad2899c85c2dcaa5",
    model="deepseek-chat",
    base_url="https://api.deepseek.com/v1",
    verbose=True,
)



tools = [CreateCar(), CreateAccount(), BindCarAccount()]



#create_openai_tools_agent函数创建一个工具代理，该代理使用OpenAI API作为语言模型，并使用提供的工具列表。
from langchain.agents import create_openai_tools_agent, AgentExecutor

#MessagesPlaceholder:提示模板中定义占位符名称（如"history"），并在后续格式化时传入对应消息列表
from langchain_core.prompts import MessagesPlaceholder,ChatPromptTemplate
from langchain_core.messages import HumanMessage

#创建agent
def create_agent(llm, tools, system_prompt):
    #提示词
    prompt = ChatPromptTemplate.from_messages([
        ("system",system_prompt),
        MessagesPlaceholder(variable_name="messages"),
        MessagesPlaceholder(variable_name="agent_scratchpad")
    ])
    #创建代理
    agent = create_openai_tools_agent(llm, tools, prompt)
    executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    return executor

#
def create_node(state, agent: AgentExecutor, name):
    result = agent.invoke(state)

    return {
        "messages":[HumanMessage(content=result["output"], name=name)]
    }


#创建agent 和 对应的节点
get_createCar_agent = create_agent(online_llm, [CreateCar()], system_prompt="你是一个汽车专家，请根据用户需求，提供专业的汽车建造建议。")

#测试 agent 是否正常运行
# response = get_createCar_agent.invoke({"messages":[HumanMessage(content="我想要一辆豪华的汽车")]})
# print(response)

import functools
create_car_node = functools.partial(create_node, agent = get_createCar_agent, name="create_car")

create_account_agent = create_agent(online_llm, [CreateAccount()], system_prompt="你是一个用户专家，请根据用户需求，提供账户创建建议。")
create_account_node = functools.partial(create_node, agent = create_account_agent, name="create_account_node")

bind_car_account_agnet = create_agent(online_llm, [BindCarAccount()], system_prompt="你是一个信息专家，请根据用户需求，提供账户绑定建议。")
bind_car_account_node = functools.partial(create_node, agent = bind_car_account_agnet, name="bind_car_account_node")


'''
上述建立好 三个agent，并使用 functools.partial 创建了对应的函数进行包装
'''

#supervisor 监督员


agent_member = ["get_createCar_agent", "create_account_agent", "bind_car_account_agnet"] #创建一个包含所有agent的列表

system_prompt = f"""
你是一名任务管理员，请根据用户需求，选择合适的任务执行者，并分配任务。下面是你的工作者{agent_member}，给定以下请求，与工作者一起响应，并采取下一步行动。
每个工作者将执行一个任务并回复执行后的结果和状态，若全部执行完后，用FINISH回应。禁止添加任何自然语言解释！
"""

options = agent_member + ["FINISH"]

function_def = {
    "name":"route",
    "description":"选择下一个工作者",
    "parameters":{
        # "title": "routeSchema",
        "type": "object",
        "properties": {
            "next": {
                "type": "string",
                "enum": options
            }
        },
        "required": ["next"]
    }
}


##创建一个函数调用模板
prompt = ChatPromptTemplate.from_messages(
    [
        ("system",system_prompt),
        MessagesPlaceholder(variable_name="messages"),
        ("system",f"基于上述的对话，接下来应该是谁来采取行动？或者告诉我们应该完成吗？请在以下选项中进行选择：{options}"),
        ("system", "请严格输出 JSON 格式，包含 'next' 字段，值为选项之一：{options}")
    ]
).partial(options=str(options), agent_member=",".join(agent_member))


from langchain_core.output_parsers import JsonOutputToolsParser
from langchain_community.output_parsers.ernie_functions import JsonOutputFunctionsParser
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnableLambda

supervisor_chain = prompt | online_llm.bind_functions(functions=[function_def], function_call="route") | JsonOutputParser()


# print(supervisor_chain.invoke({"messages": [HumanMessage(content="你好,请造一辆车")]}))



from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage
import operator
from langgraph.graph import StateGraph, END

class AgentState(TypedDict):
    
    messages : Annotated[Sequence[BaseMessage], operator.add]

    next : str



work_flow = StateGraph(AgentState)

#添加节点
work_flow.add_node("get_createCar_agent", create_car_node)
work_flow.add_node("create_account_agent",create_account_node)
work_flow.add_node("bind_car_account_agnet",bind_car_account_node)

work_flow.add_node("supervisor",supervisor_chain)


#添加边
#表示调用 agent之后都去调用supervisor，判断下一步是谁来执行
for name in agent_member:
    work_flow.add_edge(name, "supervisor")


#定义条件 map
conditions_map = {
    "get_createCar_agent":"get_createCar_agent",
    "create_account_agent":"create_account_agent",
    "bind_car_account_agnet":"bind_car_account_agnet",
    "FINISH": END,
    # "supervisor": "supervisor" ,
}

work_flow.add_conditional_edges("supervisor", lambda x: x["next"], conditions_map)

#设置入口
work_flow.set_entry_point("supervisor")

graph = work_flow.compile()


#调用
# res = graph.invoke({"messages": [HumanMessage(content="你好，请造一辆车")]}, config={"configurable":{"thread_id":42}}, debug=True)

# print(res)


#本地保存图
# from IPython.display import Image

# img = graph.get_graph().draw_png()

# with open("graph.png", "wb") as f:
#     f.write(img)

#绘制图
from PIL import Image
import io

img = graph.get_graph().draw_png()

Image.open(io.BytesIO(img)).show()


