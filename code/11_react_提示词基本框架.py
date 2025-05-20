
import os
#通义千问api_key
os.environ["DASHSCOPE_API_KEY"] = "sk-3e8d696b9b3a403caa30db95c801f875"

from dashscope import Generation


#自定义模版
query="量子力学有什么用"

physics_template = f"""你是一个非常聪明的物理学教授。
你擅长以简洁易懂的方式回答物理问题。
当你不知道答案时，你会承认不知道。

这是一个问题：
{query}"""

# print(physics_template)

'''
与langchain工具的结合使用
'''

from langchain_core.tools import tool

@tool
def multiply(first_int: int, second_int: int) -> int:
    """将两个整数相乘。"""
    return first_int * second_int*2

@tool
def add(first_int: int, second_int: int) -> int:
    "将两个整数相加。"
    return first_int + second_int

@tool
def exponentiate(base: int, exponent: int) -> int:
    "对底数求指数幂。"
    return base**exponent

tools = [multiply, add, exponentiate]

#convert_to_openai_tool: 将函数转换为OpenAI工具
#即将tool工具函数转为字典
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_core.output_parsers import JsonOutputParser,JsonOutputKeyToolsParser
from langchain.tools.render import render_text_description,render_text_description_and_args

#通义千问中 添加 工具函数
def get_chatglm_response2(messages):
    response = Generation.call(
        # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
        api_key=os.getenv('DASHSCOPE_API_KEY'),
        model="qwen-turbo",
        messages=messages,
        tools=[convert_to_openai_tool(i) for i in tools], #添加工具函数
        result_format='message'
        )
    return response


# response1 = get_chatglm_response2([{'role': 'user', 'content': "请帮我计算一下 2*3"}])
# print(response1.output.choices[0].message.tool_calls[0]["function"])

# render_tools = render_text_description(tools)
render_tools = [i.name for i in tools]

#定义模版
def prompt_react(query):
    PREFIX = """Answer the following questions as best you can. You have access to the following tools:"""
    p0= f"""Use the following format:

        Question: the input question you must answer
        Thought: you should always think about what to do
        Action: the action to take in order, should be one of {render_tools}
        Action Input: the input to the action
        Observation: the result of the action
        ... (this Thought/Action/Action Input/Observation can repeat N times)
        Thought: I now know the final answer
        Final Answer: the final answer to the original input question

        Begin!

        Question: {query}
        Thought:"""
    return [{"role":"system","content":PREFIX},{"role":"user","content":p0}]
 

# print(render_tools)


response = get_chatglm_response2(prompt_react("4加5乘以6然后再算4次方然后再加35"))
print(response.output.choices[0].message.content)





