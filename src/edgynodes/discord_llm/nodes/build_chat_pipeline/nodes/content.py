
import edgygraph
from llmir.chunks import AIChunkText
from llmir import AIMessage, AIRoles

from ..core.states import StateProtocol, SharedProtocol


class BuildDiscordContentNode[T: StateProtocol = StateProtocol, S: SharedProtocol = SharedProtocol](edgygraph.Node[T, S]):
    """Build a text chunk from the content of a message."""

    required_packages = {"py-cord", "llmir"}

    async def __call__(self, state: T, shared: S) -> None:

        async with shared.lock:
            message = shared.discord_message
            bot = shared.discord_bot

        text = message.content
        role = AIRoles.MODEL if message.author == bot.user else AIRoles.USER

        if not state.ai_messages or not isinstance(state.ai_messages[-1], AIMessage):
            state.ai_messages.append(AIMessage.text(
                text=text,
                role=role, # TODO integrate AIRoles into AIMessage
            ))

        else:
            state.ai_messages[-1].chunks.append(AIChunkText(text=text))


