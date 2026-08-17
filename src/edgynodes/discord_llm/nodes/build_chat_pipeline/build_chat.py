from edgygraph import Node, Graph, State, Shared, END, START
import discord
from discord.ext import commands
from llmir.messages import AIMessages

from ...core.states import StateProtocol, SharedProtocol
from .core.states import StateProtocol as BuildChatStateProtocol, SharedProtocol as BuildChatSharedProtocol


class BuildChatState(State):
    ai_messages: list[AIMessages]

class BuildChatShared(Shared):
    discord_message: discord.Message
    discord_bot: commands.Bot


class BuildChatNode(Node[StateProtocol, SharedProtocol]):
    """Add the last `limit` messages in the discord channel to the messages in the state.

    Attributes:
        limit: The number of messages to load from the discord channel.
        include_embeds: Whether to transfer embeds to the messages.
        include_attachments: Whether to transfer attachments to the messages.
    """

    dependencies = {"llmir", "py-cord"}

    limit: int
    nodes: tuple[Node[BuildChatStateProtocol, BuildChatSharedProtocol], ...]
    include_embeds: bool
    include_attachments: bool

    def __init__(self, limit: int = 20, include_embeds: bool = True, include_attachments: bool = True) -> None:
        super().__init__()

        self.limit = limit
        self.include_embeds = include_embeds
        self.include_attachments = include_attachments


    async def __call__(self, state: StateProtocol, shared: SharedProtocol) -> None:

        chat: list[AIMessages] = []

        async with shared.lock:
            channel = shared.discord.text_channel
            bot = shared.discord.bot

        graph = Graph[BuildChatStateProtocol, BuildChatSharedProtocol](
            edges=[(START,*self.nodes,END,)]
        )

        async for msg in channel.history(limit=self.limit, oldest_first=False):
            
            build_state = BuildChatState(ai_messages=[])
            build_shared = BuildChatShared(discord_message=msg, discord_bot=bot)

            build_state, build_shared = await graph(build_state, build_shared)

            chat.extend(build_state.ai_messages)


