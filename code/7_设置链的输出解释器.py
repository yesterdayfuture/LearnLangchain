import os
from langchain_community.chat_models import ChatOllama, ChatOpenAI

llm1 = ChatOllama(
    model="llama3.2:3b"
)

# os.environ["DEEPSEEK_API_KEY"] = "sk-8f299497aaa74c64ad2899c85c2dcaa5"

# llm2 = ChatOpenAI(
#     api_key=os.getenv("DEEPSEEK_API_KEY"),
#     base_url="https://api.deepseek.com/v1",
#     model="deepseek-chat"
# )


#导入输出解析器， 定义大模型输出的格式
'''
输出解析器共有的方法：
1. parse()：接收一个字符串，解析大模型输出，返回解析后的结果。
2. get_format_instructions()：返回格式说明，用于指导大模型输出格式。
'''
from langchain.output_parsers import CommaSeparatedListOutputParser, StructuredOutputParser, ResponseSchema
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate


#定义输出解析器

# #将输出结果转为列表
# output_parser = CommaSeparatedListOutputParser()
# print(output_parser.get_format_instructions())

# format_instructions = output_parser.get_format_instructions()



'''
将输出结果转为结构化数据
'''

#定义输出结构
schema = [
        ResponseSchema(
            name="name",
            description="人物名字"
        ),
        ResponseSchema(
            name="age",
            description="人物年龄"
        )
]

#定义输出解析器
output_parser = StructuredOutputParser.from_response_schemas(schema)

prompt = PromptTemplate(
    input_variables=["text"],
    template="请列出一个{text}.注意不要有键的描述信息、不要描述人物故事等信息，人物要真实存在。{output_format}",
    partial_variables={"output_format": output_parser.get_format_instructions()}
)

#定义链
chain = LLMChain(llm=llm1, prompt=prompt, verbose=True)

# chain = prompt | llm1 | output_parser

#调用链
response = chain.run("感动中国人物的信息")
print(response)



