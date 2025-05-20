
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from typing import List, Optional

online_llm = ChatOpenAI(
    api_key="sk-8f299497aaa74c64ad2899c85c2dcaa5",
    model="deepseek-chat",
    base_url="https://api.deepseek.com/v1",
    verbose=True,
)


print(online_llm.invoke("What is the capital of France?"))

class AnswerWithJustification(BaseModel):
            '''An answer to the user question along with justification for the answer.'''

            answer: str
            justification: Optional[str] = Field(
                default=..., description="A justification for the answer."
            )


structured_llm = online_llm.with_structured_output(AnswerWithJustification)

res = structured_llm.invoke(
            "What weighs more a pound of bricks or a pound of feathers"
        )

print(res)






