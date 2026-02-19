"""Multimodal content types for rich message handling.

Provides ContentBlock types for text, images, documents, and tool results,
with backward-compatible string wrapping.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class TextBlock:
    """Plain text content block."""

    text: str
    type: str = "text"


@dataclass(frozen=True)
class ImageBlock:
    """Image content block with base64-encoded data or URL."""

    source: str  # base64 data or URL
    media_type: str = "image/png"
    source_type: str = "base64"  # "base64" or "url"
    type: str = "image"

    @classmethod
    def from_file(cls, path: str, media_type: str = "image/png") -> ImageBlock:
        """Create an ImageBlock from a file path."""
        with open(path, "rb") as f:
            data = base64.b64encode(f.read()).decode()
        return cls(source=data, media_type=media_type, source_type="base64")

    @classmethod
    def from_url(cls, url: str, media_type: str = "image/png") -> ImageBlock:
        """Create an ImageBlock from a URL."""
        return cls(source=url, media_type=media_type, source_type="url")


@dataclass(frozen=True)
class DocumentBlock:
    """Document content block (PDF, etc.)."""

    source: str  # base64 data or URL
    media_type: str = "application/pdf"
    name: str = ""
    type: str = "document"


@dataclass(frozen=True)
class ToolResultBlock:
    """Result from a tool invocation."""

    tool_use_id: str
    content: str
    is_error: bool = False
    type: str = "tool_result"


ContentBlock = TextBlock | ImageBlock | DocumentBlock | ToolResultBlock


@dataclass
class Message:
    """A message with role and multi-part content."""

    role: str
    content: list[ContentBlock] = field(default_factory=list)

    @classmethod
    def from_text(cls, role: str, text: str) -> Message:
        """Create a simple text message."""
        return cls(role=role, content=[TextBlock(text=text)])

    def text(self) -> str:
        """Extract concatenated text from all TextBlocks."""
        return "".join(block.text for block in self.content if isinstance(block, TextBlock))

    def to_dict(self) -> dict[str, Any]:
        """Convert to a provider-friendly dict format."""
        if all(isinstance(b, TextBlock) for b in self.content):
            return {"role": self.role, "content": self.text()}
        blocks = []
        for block in self.content:
            if isinstance(block, TextBlock):
                blocks.append({"type": "text", "text": block.text})
            elif isinstance(block, ImageBlock):
                if block.source_type == "url":
                    blocks.append(
                        {
                            "type": "image",
                            "source": {"type": "url", "url": block.source},
                        }
                    )
                else:
                    blocks.append(
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": block.media_type,
                                "data": block.source,
                            },
                        }
                    )
            elif isinstance(block, DocumentBlock):
                blocks.append(
                    {
                        "type": "document",
                        "source": {
                            "type": "base64",
                            "media_type": block.media_type,
                            "data": block.source,
                        },
                        "name": block.name,
                    }
                )
            elif isinstance(block, ToolResultBlock):
                blocks.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.tool_use_id,
                        "content": block.content,
                        "is_error": block.is_error,
                    }
                )
        return {"role": self.role, "content": blocks}


def wrap_input(message: str | Message) -> Message:
    """Wrap a plain string as a Message if needed. Backward compatible."""
    if isinstance(message, Message):
        return message
    return Message.from_text("user", message)
