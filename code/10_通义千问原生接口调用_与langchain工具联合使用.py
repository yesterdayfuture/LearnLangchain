
import os
#通义千问api_key
os.environ["DASHSCOPE_API_KEY"] = "sk-3e8d696b9b3a403caa30db95c801f875"

from dashscope import Generation

def get_chatglm_response(messages):
    # messages = [
    # {'role': 'system', 'content': 'You are a helpful assistant.'},
    # {'role': 'user', 'content': f'{prompt}'}
    # ]
    response = Generation.call(
        # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
        api_key=os.getenv('DASHSCOPE_API_KEY'),
        model="qwen-turbo",
        messages=messages,
        result_format='message'
        )
    return response


#调用通义千问api
# response = get_chatglm_response([{'role': 'user', 'content': '你是谁'}])
# print(response.output.choices[0].message.content)

#自定义模版
query="量子力学有什么用"

physics_template = f"""你是一个非常聪明的物理学教授。
你擅长以简洁易懂的方式回答物理问题。
当你不知道答案时，你会承认不知道。

这是一个问题：
{query}"""

# print(physics_template)

#测试模版
# response = get_chatglm_response([{'role': 'user', 'content': physics_template}])
# print(response.output.choices[0].message.content)



'''
与langchain工具的结合使用
'''

from langchain_core.tools import tool

@tool
def multiply(first_int: int, second_int: int) -> int:
    """将两个整数相乘。"""
    return first_int * second_int

@tool
def add(first_int: int, second_int: int) -> int:
    "将两个整数相加。"
    return first_int + second_int

@tool
def exponentiate(base: int, exponent: int) -> int:
    "对底数求指数幂。"
    return base**exponent

#convert_to_openai_tool: 将函数转换为OpenAI工具
#即将tool工具函数转为字典
from langchain_core.utils.function_calling import convert_to_openai_tool

#通义千问中 添加 工具函数
def get_chatglm_response2(messages):
    response = Generation.call(
        # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
        api_key=os.getenv('DASHSCOPE_API_KEY'),
        model="qwen-turbo",
        messages=messages,
        tools=[convert_to_openai_tool(i) for i in [multiply, add]], #添加工具函数
        result_format='message'
        )
    return response


response1 = get_chatglm_response2([{'role': 'user', 'content': "请帮我计算一下 2*3+4"}])
print(response1)
print(response1.output.choices[0].message.tool_calls[0]["function"])



