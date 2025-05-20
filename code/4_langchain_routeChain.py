
from langchain_community.llms.ollama import Ollama
from langchain_core.prompts import PromptTemplate,ChatPromptTemplate
from langchain.chains import LLMChain, ConversationChain
from langchain.chains.router.llm_router import RouterOutputParser
from langchain.chains.router import MultiPromptChain, LLMRouterChain
from langchain.chains.router.multi_prompt_prompt import MULTI_PROMPT_ROUTER_TEMPLATE
from langchain.prompts import PromptTemplate

import warnings
warnings.filterwarnings("ignore")

'''
RouterOutputParser是LLMChain的子类，用于解析路由输出
ConversationChain是LLMChain的子类，用于处理对话,可以设置历史对话
'''

llm = Ollama(base_url="http://localhost:11434", model="deepseek-r1:1.5b", temperature=0.7)

# prompt = PromptTemplate(
#     input_variables=["content"],
#     template="你好，请你介绍一下以下问题，内容：{input}"
# )

#目标链的prompt
math_prompt = """
你是一位数学家，请根据以下问题，生成对应的数学公式.内容：{input}
"""

weather_prompt = """你是一位天气预报员，请根据以下问题，生成对应的天气信息.内容：{input}"""
news_prompt = """你是一位新闻工作者，请根据以下问题，生成对应的新闻内容.内容：{input}"""

prompt_info = [
    {
        "name": "math",
        "description": "数学问题",
        "prompt": math_prompt
    },
    {
        "name": "weather",
        "description": "天气问题",
        "prompt": weather_prompt
    },
    {
        "name": "news",
        "description": "新闻问题",
        "prompt": news_prompt
    }
]

#定义目标链
# math_chain = LLMChain(llm=llm, prompt=PromptTemplate(input_variables=["content"], template=math_prompt))
# weather_chain = LLMChain(llm=llm, prompt=PromptTemplate(input_variables=["content"], template=weather_prompt))
# news_chain = LLMChain(llm=llm, prompt=PromptTemplate(input_variables=["content"], template=news_prompt))

chains = {}
for info in prompt_info:
    chain = LLMChain(llm=llm, prompt=PromptTemplate.from_template(template=info["prompt"]))
    chains[info["name"]] = chain

print(chains)



#定义默认链
default_chain = ConversationChain(llm=llm,output_key="text")

#路由配置
prompt_info = [
    "math:你是一位数学家，请根据以下问题，生成对应的数学公式",
    "weather:你是一位天气预报员，请根据以下问题，生成对应的天气信息",
    "news:你是一位新闻工作者，请根据以下问题，生成对应的新闻内容",
]

#构建路由模版


router_template = MULTI_PROMPT_ROUTER_TEMPLATE.format(destinations = "\n".join(prompt_info))
print("路由模版是：",router_template)

router_prompt = PromptTemplate(
    template=router_template,
    # template="\n".join(prompt_info),
    input_variables=["input"],
    output_parser=RouterOutputParser()
    )
print("路由模版router_prompt 是：",router_prompt)

#构建路由链
router_chain = LLMRouterChain.from_llm(llm=llm, prompt=router_prompt)

#构建最终路由链，将路由链和默认链组合
expert_chain = MultiPromptChain(
    router_chain=router_chain, #路由链
    destination_chains=chains, #路由链对应的链
    default_chain=default_chain, #默认链
    verbose=True
)

#调用最终链

question = [
    "今天北京天气怎么样？",
    "请给我一个数学题，简单一点的",
    "最近有什么新闻吗？",
    "你叫什么名字？",
    "请你写一首诗，歌颂祖国"
]

for q in question:
    print("问题：",q)
    print("回答：",expert_chain.invoke({"input":q}))