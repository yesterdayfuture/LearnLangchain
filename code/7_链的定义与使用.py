
import os
from langchain_community.chat_models import ChatOllama, ChatOpenAI

# llm1 = ChatOllama(
#     model="llama3.2:3b"
# )

os.environ["DEEPSEEK_API_KEY"] = "sk-8f299497aaa74c64ad2899c85c2dcaa5"

llm2 = ChatOpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com/v1",
    model="deepseek-chat"
)



# response = llm2.invoke("你叫什么？可以做什么？")

# print(response.content)

from langchain.prompts import PromptTemplate, FewShotPromptTemplate

template_text = """
我希望你能充当新公司的命名顾问。一个生成{product}的公司好的中文名字是什么，请给出五个选项？
"""

prompt = PromptTemplate(
    input_variables=["product"],
    template=template_text
)


from langchain.chains.llm import LLMChain

#单个输入
# chains_ceshi = prompt | llm2
# response = chains_ceshi.invoke("智能手表")
# print(response.content)


input_list = [
    {"product": "智能手表"},
    {"product": "智能手环"},
    {"product": "智能眼镜"}
]

#多个输入
chains_ceshi = LLMChain(llm=llm2, prompt=prompt)

response = chains_ceshi.run(input_list[0])
print(response)
# response2 = chains_ceshi.apply(input_list)
# print(response2)


