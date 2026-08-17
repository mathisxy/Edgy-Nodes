from typing import Protocol, runtime_checkable

import discord
from discord.ext import commands
from edgygraph import StateProtocol as BaseStateProtocol, SharedProtocol as BaseSharedProtocol
from llmir import AIMessages


@runtime_checkable
class StateProtocol(BaseStateProtocol, Protocol):
    ai_messages: list[AIMessages]

@runtime_checkable
class SharedProtocol(BaseSharedProtocol, Protocol):
    discord_message: discord.Message
    discord_bot: commands.Bot
