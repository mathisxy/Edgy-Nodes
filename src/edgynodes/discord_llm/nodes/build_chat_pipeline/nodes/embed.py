import discord
import edgygraph
from llmir import AIChunkText, AIChunkImageURL, AIChunks
from llmir.messages import AIMessage
from llmir.roles import AIRoles

from ..core.states import StateProtocol, SharedProtocol

class BuildDiscordEmbedNode[T: StateProtocol = StateProtocol, S: SharedProtocol = SharedProtocol](edgygraph.Node[T, S]):
    """Build a text chunk from the embed of a message."""

    required_packages = {"py-cord", "llmir"}

    async def __call__(self, state: T, shared: S) -> None:

        async with shared.lock:
            message = shared.discord_message
            bot = shared.discord_bot

        ai_chunks: list[AIChunks] = []
        role = AIRoles.MODEL if message.author == bot.user else AIRoles.USER

        for embed in message.embeds:
            ai_chunks.extend(self.format_embed(embed))

        if ai_chunks:
            if not state.ai_messages or not isinstance(state.ai_messages[-1], AIMessage):
                state.ai_messages.append(AIMessage(
                    chunks=ai_chunks,
                    role=role, # TODO integrate AIRoles into AIMessage
                ))
            else:
                state.ai_messages[-1].chunks.extend(ai_chunks)
                

    def format_embed(self, embed: discord.Embed) -> list[AIChunks]:
        """Konvertiert ein Discord Embed in Text AI Chunks"""
        chunks: list[AIChunks] = []
        
        if embed.title:
            chunks.append(AIChunkText(text=f"**{embed.title}**"))
        
        if embed.description:
            chunks.append(AIChunkText(text=embed.description))
        
        if embed.fields:
            for field in embed.fields:
                chunks.append(AIChunkText(text=f"{field.name}: {field.value}"))
        
        if embed.footer and embed.footer.text:
            chunks.append(AIChunkText(text=f"__{embed.footer.text}__"))

        if embed.image:
            chunks.append(AIChunkImageURL(url=embed.image.url))

        if embed.video: # Not supported currently
            chunks.append(AIChunkText(text=embed.video.url))
        
        return chunks