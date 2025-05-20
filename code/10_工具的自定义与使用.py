

from langchain_community.chat_models import ChatOllama
from langchain.embeddings import OllamaEmbeddings

#定义模型与向量化模型
local_llm = ChatOllama(
    model="llama3.2:3b"
)

local_embedding = OllamaEmbeddings(
    model="mxbai-embed-large:335m"
)


'''
工具本质含义是一个函数，具备特定功能

装饰器：在不改变原有函数代码的情况下，增加额外的功能
'''

#tool 是一个装饰器，用于将函数包装成工具
from langchain_community.tools import tool


@tool
def multiply(first_int: int, second_int: int) -> int:
    """将两个整数相乘。"""
    return first_int * second_int

@tool
def add_sum(first_int: int, second_int: int) -> int:
    """将两个整数相加。"""
    return first_int + second_int

print(multiply.name)
print(multiply.description)
print(multiply.args_schema)

#运行工具
response_tool = multiply.invoke({"first_int": 3, "second_int": 4})
# print(response_tool)


#获取工具的描述
from langchain.tools.render import render_text_description,render_text_description_and_args

#获取工具的描述
render_tools = render_text_description([add_sum, multiply])
# print(render_tools)

#获取工具的描述与参数
render_tools_args = render_text_description_and_args([add_sum, multiply])
# print(render_tools_args)



#定义对应提示词
from langchain_core.prompts import ChatPromptTemplate

system_prompt = f"""您是一名助理，可以使用以下工具集。 以下是每个工具的名称和说明:

{render_tools}

根据用户输入，返回要使用的工具的名称和输入。 将您的响应作为带有“name”和“arguments”键的 JSON blob 返回.此JSON blob必须是如下格式：```json
...
```"""

prompt = ChatPromptTemplate.from_messages(
    [("system", system_prompt), ("user", "{input}")]
)


#定义输出解析器
from langchain_core.output_parsers import JsonOutputParser, JsonOutputToolsParser, SimpleJsonOutputParser
from langchain.output_parsers import StructuredOutputParser
from operator import itemgetter

chains = prompt | local_llm | JsonOutputParser() | itemgetter("arguments") | add_sum

#运行
# response = chains.invoke("3 + 4是多少？")
# print(response)



'''
在多工具中，根据条件选择工具使用
'''

tools = [add_sum, multiply]

#根据 模型输出的 json ，返回对应工具
def tool_chain(model_output):

    tool_map ={tool.name: tool for tool in tools}

    choosen_tool = tool_map[model_output["name"]]
    
    return itemgetter("arguments") | choosen_tool

system_prompt = f"""您是一名助理，可以使用以下工具集。 以下是每个工具的名称和说明:

{render_tools}

根据用户输入，返回要使用的工具的名称和输入。 将您的响应作为带有“name”和“arguments”键的 JSON blob 返回.此JSON blob必须是如下格式：```json
...
```"""

prompt = ChatPromptTemplate.from_messages(
    [("system", system_prompt), ("user", "{input}")])

chains = prompt | local_llm | JsonOutputParser() | tool_chain

# response = chains.invoke("3 + 4是多少？")
# print(response)
# response = chains.invoke("3 * 4是多少？")
# print(response)




'''
在多个工具中，根据大模型输出自动选择多个工具进行并行使用
方式一：openai 相关模型 使用bind_tools函数
'''
#多类型声明‌ Union 允许你定义一个值可以属于多个类型（例如 int 或 str）
from typing import Union

from langchain_core.runnables import (
    Runnable,
    RunnableLambda,
    RunnableMap,
    RunnablePassthrough,
)

system_prompt = f"""您是一名助理，为了完美回答用户的问题，可以使用以下工具集中的一个或多个工具来完成任务。 以下是每个工具的名称和说明:

{render_tools}

根据用户输入，返回解决用户问题所需的所有工具的名称和输入，按照执行顺序。将您的响应作为多个带有“name”和“arguments”键的 JSON blob 返回,
“arguments”键对应的值应该是所选函数的输入参数的字典，字典里不要有任何说明,此JSON blob必须是如下格式：```json
...
```"""

prompt = ChatPromptTemplate.from_messages(
    [("system", system_prompt), ("user", "{input}")])

#工具列表
tools = [add_sum, multiply]

#工具名称与工具的映射
tools_map = {tool.name: tool for tool in tools}


#工具调用函数
def call_tool(tool_invocation: dict) -> Union[str, Runnable]:

    tool_name = tool_invocation["type"]
    tool = tools_map[tool_name]

    args = tool_invocation["args"]
    return RunnablePassthrough.assign(output = itemgetter("args") | tool)


call_tool_list = RunnableLambda(call_tool).map()



import os
from langchain_openai import ChatOpenAI

os.environ["DEEPSEEK_API_KEY"] = "sk-8f299497aaa74c64ad2899c85c2dcaa5"

online_llm = ChatOpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com/v1",
    model="deepseek-chat"
).bind_tools(tools=tools)


chain0 =  online_llm  | JsonOutputToolsParser() | call_tool_list
# response = online_llm.invoke("3 + 4是多少？3 * 5是多少？").tool_calls #可以输出工具调用的列表
response = chain0.invoke("3 + 4是多少？3 * 5是多少？")
print(response)




'''
在多个工具中，根据大模型输出自动选择多个工具进行并行使用
方式二：非openai 相关模型 调用
'''

system_prompt = f"""您是一名助理，为了完美回答用户的问题，可以使用以下工具集中的一个或多个工具来完成任务。 以下是每个工具的名称和说明:

{render_tools}

根据用户输入，返回解决用户问题所需的所有工具的名称和输入，按照执行顺序。将您的响应作为多个带有“name”和“arguments”键的 JSON blob 返回,返回结果只包含json，不需要有工具信息之外的任何信息，
“arguments”键对应的值应该是所选函数的输入参数的字典，字典里不要有任何说明,此JSON blob必须是如下格式：```json
...
```"""

prompt = ChatPromptTemplate.from_messages(
    [("system", system_prompt), ("user", "{input}")])

#工具调用函数
def call_tool(tool_invocation: dict) -> Union[str, Runnable]:

    tool_name = tool_invocation["name"]
    tool = tools_map[tool_name]

    args = tool_invocation["arguments"]
    return RunnablePassthrough.assign(output = itemgetter("arguments") | tool)

call_tool_list = RunnableLambda(call_tool).map()


chain0 = {"input"  :RunnablePassthrough()} | prompt | local_llm | JsonOutputToolsParser() 
#| call_tool_list
# response = chain0.invoke("3 + 4,3 * 5分别是多少？")
# print(response)


