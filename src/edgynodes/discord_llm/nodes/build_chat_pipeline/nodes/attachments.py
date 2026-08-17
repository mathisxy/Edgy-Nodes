import edgygraph
import mimetypes
from llmir import AIChunkFile
from llmir.chunks import AIChunks
from llmir.messages import AIMessage
from llmir.messages import AIMessage
from llmir.roles import AIRoles

from ..core.states import StateProtocol, SharedProtocol

class BuildDiscordAttachmentsNode[T: StateProtocol = StateProtocol, S: SharedProtocol = SharedProtocol](edgygraph.Node[T, S]):

    """Build file chunks from the attachments of a message."""

    required_packages = {"py-cord", "llmir", "mimetypes"}

    async def __call__(self, state: T, shared: S) -> None:

        async with shared.lock:
            message = shared.discord_message
            bot = shared.discord_bot

        ai_chunks: list[AIChunks] = []
        role = AIRoles.MODEL if message.author == bot.user else AIRoles.USER

        for attachment in message.attachments:
            mimetype, _ = mimetypes.guess_type(attachment.filename)
            file_bytes = await attachment.read()

            ai_chunks.append(AIChunkFile(
                name=attachment.filename,
                mimetype=str(mimetype),
                bytes=file_bytes,
            ))

        if ai_chunks:
            if not state.ai_messages or not isinstance(state.ai_messages[-1], AIMessage):
                state.ai_messages.append(AIMessage(
                    chunks=ai_chunks,
                    role=role, # TODO integrate AIRoles into AIMessage
                ))
            else:
                state.ai_messages[-1].chunks.extend(ai_chunks)