
#langchain 中prompt模版的使用
from langchain import PromptTemplate

template = """
我希望你可以当新公司的命名顾问。
一个生产{content}的公司的好名字是什么？
"""


#方式一
prompt = PromptTemplate(
    input_variables=["content"],
    template=template
)

content = "汽车"

print(prompt.format(content=content))



#方式二
prompt = PromptTemplate.from_template(template)

print(prompt.format(content=content))

#方式三
prompt = PromptTemplate.from_template(template).format(content=content)

print(prompt)



#多个变量
template = """
请写一本关于{content}的书籍，字数{word_count}，要求是{requirement}
"""

prompt = PromptTemplate.from_template(template)

content = "汽车"
word_count = 10000
requirement = "通俗易懂"

print(prompt.format(content=content, word_count=word_count, requirement=requirement))

#多个变量，使用字典
dictionary = {
    "content": "汽车",
    "word_count": 10000,
    "requirement": "通俗易懂"
}

print(prompt.format(**dictionary))



#定义示例模版
from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate

#少量示例数据
exmaples = [
    {
        "input": "汽车",
        "output": "汽车之家"
    },
    {
        "input": "手机",
        "output": "手机之家"
    }
]

template = """
输入：{input}
输出：{output}
"""

template_prompt = PromptTemplate(input_variables=["input", "output"], template=template)

prompt = FewShotPromptTemplate(
    #示例数据
    examples=exmaples,
    #模版
    example_prompt=template_prompt,
    #前缀 提示词前面的内容
    prefix="请根据输入，输出对应的网站名称",
    #后缀 提示词后面的内容
    suffix="输入：{input}\n输出：",
    #输入变量
    input_variables=["input"],
    #将 前缀 示例数据 后缀 连接的字符
    example_separator="\n\n"

)

print(prompt.format(input="电脑"))
