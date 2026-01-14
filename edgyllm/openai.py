from .nodes import LLMNode
from .states import LLMGraphState

from openai import OpenAI

from llm_ir import AIMessage, AIRoles, AIChunkText
from llm_ir.adapter import to_openai, OpenAIMessage

class LLMNodeOpenai(LLMNode):

    client: OpenAI

    def __init__(self, model: str, api_key: str, base_url: str ="https://api.openai.com/v1") -> None:
        super().__init__(model)

        self.client = OpenAI(api_key=api_key, base_url=base_url)


    def run(self, state: LLMGraphState) -> LLMGraphState:
        
        history = state.messages

        print(history)

        response = self.client.chat.completions.create(
            model=self.model,
            messages=self.format_messages(history), # type: ignore
        )

        print(response)

        state.messages.append(
            AIMessage(
                role=AIRoles.MODEL,
                chunks=[AIChunkText(
                    text=str(response.choices[0].message.content)
                    )
                ],
            )
        )

        return state
    
    def format_messages(self, messages: list[AIMessage]) -> list[OpenAIMessage]:
        return to_openai(messages)
    