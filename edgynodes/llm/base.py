from edgygraph import GraphNode, GraphState
from llm_ir import AIMessage
from pydantic import BaseModel
from typing import TypeVar, Generic


S = TypeVar('S', bound=object, default=object, covariant=True)

class LLMGraphState(GraphState[S], Generic[S]):
    messages: list[AIMessage] = []


class Supports(BaseModel):

    vision: bool = True
    audio: bool = False
    streaming: bool = True
    remote_image_urls: bool = True

T = TypeVar('T', bound=LLMGraphState)

class LLMNode(GraphNode[T], Generic[T]):

    model: str
    
    enable_streaming: bool

    supports: Supports = Supports()
    
    def __init__(self, model: str, enable_streaming: bool = False) -> None:
        self.model = model
        self.enable_streaming = enable_streaming