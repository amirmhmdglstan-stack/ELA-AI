"""The single chat serialisation format used for fine-tuning and inference.

Keeping this in one place guarantees that the SFT data and the chat interface
agree on the exact token sequence, which is the most common cause of silent
quality loss in small instruction-tuned models.
"""

from __future__ import annotations

DEFAULT_SYSTEM = "You are a concise, helpful coding assistant."

# Special tokens (must exist in the tokenizer)
EOT = "<|endoftext|>"
USER = "<|user|>"
ASSISTANT = "<|assistant|>"
SYSTEM = "<|system|>"


def render_prompt(messages: list[dict], system: str | None = DEFAULT_SYSTEM,
                  add_system: bool = True) -> str:
    """Serialise a conversation up to (and including) the assistant header.

    ``messages`` is a list of {"role": "user"|"assistant", "content": str}.
    """
    parts = []
    if add_system and system:
        parts.append(f"{SYSTEM}{system}{EOT}")
    for m in messages:
        if m["role"] == "user":
            parts.append(f"{USER}{m['content'].strip()}{EOT}")
        elif m["role"] == "assistant":
            parts.append(f"{ASSISTANT}{m['content'].strip()}{EOT}")
        else:
            raise ValueError(f"unsupported role {m['role']!r}")
    if messages and messages[-1]["role"] == "assistant":
        return "".join(parts)
    return "".join(parts) + ASSISTANT


def render_example(messages: list[dict], system: str | None = DEFAULT_SYSTEM
                   ) -> tuple[str, str]:
    """Split a full conversation into (prompt, response) for masked SFT.

    The response (the final assistant turn) is what we compute loss on; it ends
    with the end-of-text token so the model learns to stop.
    """
    if not messages or messages[-1]["role"] != "assistant":
        raise ValueError("last message must be from the assistant")
    prompt = render_prompt(messages[:-1], system=system)
    response = messages[-1]["content"].strip() + EOT
    return prompt, response
