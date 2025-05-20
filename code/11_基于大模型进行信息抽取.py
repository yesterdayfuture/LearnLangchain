
from langchain_community.chat_models import ChatOpenAI
from pydantic import BaseModel,Field
#Optional: 可选类型
from typing import Optional
from langchain.tools import BaseTool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from langchain_core.utils.function_calling import convert_to_openai_tool

from dashscope import Generation
import dashscope

import os
#通义千问api_key
os.environ["DASHSCOPE_API_KEY"] = "sk-3e8d696b9b3a403caa30db95c801f875"


online_llm = ChatOpenAI(
    api_key="sk-8f299497aaa74c64ad2899c85c2dcaa5",
    model="deepseek-chat",
    base_url="https://api.deepseek.com/v1",
    verbose=True,
)


'''
单个对象 进行信息抽取
'''

class Person(BaseModel):
    """一个人的信息."""
    name: Optional[str] = Field(default=None, description="人的名字")
    hair_color: Optional[str] = Field(default=None, description="头发颜色")
    height_in_meters: Optional[str] = Field(default=None, description="身高，单位为米")



prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert extraction algorithm. "
            "Only extract relevant information from the text. "
            "If you do not know the value of an attribute asked to extract, "
            "return null for the attribute's value.",
        ),
        ("human", "{text}"),
    ]
)


# print(convert_to_openai_tool(Person))



#通义千问中 添加 工具函数
def get_chatglm_response2(messages):
    response = Generation.call(
        # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
        api_key=os.getenv('DASHSCOPE_API_KEY'),
        model="qwen-turbo",
        messages=messages,
        tools=[convert_to_openai_tool(Person)], #添加工具函数
        result_format='message'
        )
    return response


def prompt_ty(input):
    pro_sys="你是提取算法的专家。仅从文本中提取相关信息。如果您不知道要求提取的属性的值，请为该属性的值返回null。"
    return [{"role":"system","content":pro_sys},{"role":"user","content":input}]

p_y=prompt_ty("无良身高大约1米7，黄色短发.")
# res=get_chatglm_response2(p_y)

# print(res.output.choices[0].message.tool_calls[0])

'''
多个对象 进行信息抽取
'''

from typing import List

class PersonData(BaseModel):
    '''抽取关于人的信息'''
    people: List[Person] = Field(default=[], description="可以同时抽取多个实体的信息")



#通义千问中 添加 工具函数
def get_chatglm_response2(messages):
    response = Generation.call(
        # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
        api_key=os.getenv('DASHSCOPE_API_KEY'),
        model="qwen-turbo",
        messages=messages,
        tools=[convert_to_openai_tool(PersonData)], #添加工具函数
        result_format='message'
        )
    return response


def prompt_ty(input):
    pro_sys="你是提取算法的专家。仅从文本中提取相关信息。如果您不知道要求提取的属性的值，请为该属性的值返回null。"
    return [{"role":"system","content":pro_sys},{"role":"user","content":input}]

p_y=prompt_ty("王五和李四身高大约1米7，王五是黄色短发，李四是黑色长发.")
res=get_chatglm_response2(p_y)

print(res.output.choices[0].message.tool_calls[0])


