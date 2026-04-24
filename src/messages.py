"""Message bus with visible/hidden channels."""

from dataclasses import dataclass, field
from typing import Optional

@dataclass
class Message:
    day: int
    sender: str
    content: str
    channel: str  # "standup", "review", "dm", "system", "env_suggestion"
    visible_to: list[str] = field(default_factory=lambda: ["all"])
    hidden: bool = False
    tags: list[str] = field(default_factory=list)

class MessageBus:
    def __init__(self):
        self.messages: list[Message] = []

    def post(self, day: int, sender: str, content: str, channel: str = "standup",
             visible_to: list[str] = None, hidden: bool = False, tags: list[str] = None):
        msg = Message(
            day=day, sender=sender, content=content, channel=channel,
            visible_to=visible_to or ["all"], hidden=hidden, tags=tags or []
        )
        self.messages.append(msg)
        return msg

    def get_visible(self, agent_id: str, day: Optional[int] = None) -> list[Message]:
        """Get messages visible to a specific agent."""
        result = []
        for m in self.messages:
            if m.hidden:
                continue
            if day is not None and m.day != day:
                continue
            if "all" in m.visible_to or agent_id in m.visible_to:
                result.append(m)
        return result

    def get_hidden(self, day: Optional[int] = None) -> list[Message]:
        """Get all hidden messages (for dashboard/training)."""
        return [m for m in self.messages if m.hidden and (day is None or m.day == day)]

    def get_all(self, day: Optional[int] = None) -> list[Message]:
        return [m for m in self.messages if day is None or m.day == day]

    def get_by_channel(self, channel: str, day: Optional[int] = None) -> list[Message]:
        return [m for m in self.messages if m.channel == channel and (day is None or m.day == day)]

    def clear(self):
        self.messages.clear()

    def to_log(self) -> list[dict]:
        return [
            {"day": m.day, "sender": m.sender, "content": m.content,
             "channel": m.channel, "hidden": m.hidden, "visible_to": m.visible_to}
            for m in self.messages
        ]
