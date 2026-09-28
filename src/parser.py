"""
WhatsApp chat export parser.
Handles Android and iOS exports across locales, date formats, and WhatsApp versions.
"""
import re
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional


# Unicode directional markers prepended in some locales (Arabic, Hebrew, Hindi, etc.)
UNICODE_JUNK = re.compile(r"[‎‏‪‬﻿]+")

# Date component: 1-4 digits separated by / . or -
_DATE = r"\d{1,4}[\/.\-]\d{1,2}[\/.\-]\d{2,4}"
# Time: HH:MM or HH:MM:SS, optional AM/PM variants
_TIME = r"\d{1,2}:\d{2}(?::\d{2})?(?:\s?[AaPp]\.?\s?[Mm]\.?)?"

# Android format:  "12/31/23, 11:59 PM - Name: message"
#                  "31.12.2023, 23:59 - Name: message"
ANDROID_RE = re.compile(
    rf"^({_DATE}),\s({_TIME})\s[-–]\s([^:]+?):\s(.+)$"
)
# Android system: same timestamp but no "Name: " part
ANDROID_SYSTEM_RE = re.compile(
    rf"^({_DATE}),\s({_TIME})\s[-–]\s"
)

# iOS / newer Android bracket format: "[31/12/23, 23:59:05] Name: message"
IOS_RE = re.compile(
    rf"^\[({_DATE}),\s({_TIME})\]\s([^:]+?):\s(.+)$"
)
IOS_SYSTEM_RE = re.compile(
    rf"^\[({_DATE}),\s({_TIME})\]"
)

MEDIA_STRINGS = {
    "<media omitted>", "<image omitted>", "<video omitted>",
    "<audio omitted>", "<sticker omitted>", "<gif omitted>",
    "<document omitted>", "image omitted", "video omitted",
    "audio omitted", "sticker omitted", "gif omitted",
}


@dataclass
class Message:
    timestamp: pd.Timestamp
    sender: str
    text: str
    is_media: bool = False


def _clean(line: str) -> str:
    return UNICODE_JUNK.sub("", line).strip()


def _parse_ts(date_str: str, time_str: str) -> pd.Timestamp:
    # Normalize AM/PM variants: "a.m." → "AM", "p. m." → "PM"
    time_str = re.sub(r"[Aa]\.?\s?[Mm]\.?", "AM", time_str).strip()
    time_str = re.sub(r"[Pp]\.?\s?[Mm]\.?", "PM", time_str).strip()

    raw = f"{date_str} {time_str}"

    for dayfirst in (True, False):
        try:
            ts = pd.to_datetime(raw, dayfirst=dayfirst)
            if not pd.isna(ts):
                return ts
        except Exception:
            pass
    return pd.NaT


def parse(text: str) -> tuple[list[Message], bool]:
    """
    Parse WhatsApp export text.
    Returns (messages, is_group_chat).
    """
    messages: list[Message] = []
    current: Optional[Message] = None

    for raw_line in text.splitlines():
        line = _clean(raw_line)
        if not line:
            continue

        match = ANDROID_RE.match(line) or IOS_RE.match(line)

        if match:
            if current:
                messages.append(current)

            date_s, time_s, sender, body = match.groups()
            ts = _parse_ts(date_s, time_s)
            is_media = body.strip().lower() in MEDIA_STRINGS

            current = Message(
                timestamp=ts,
                sender=sender.strip(),
                text="" if is_media else body.strip(),
                is_media=is_media,
            )

        elif ANDROID_SYSTEM_RE.match(line) or IOS_SYSTEM_RE.match(line):
            # system message (e.g. "Messages and calls are end-to-end encrypted")
            if current:
                messages.append(current)
                current = None

        elif current:
            # continuation line of a multi-line message
            current.text += "\n" + line

    if current:
        messages.append(current)

    messages = [m for m in messages if not pd.isna(m.timestamp)]
    senders = {m.sender for m in messages}
    is_group = len(senders) > 2

    return messages, is_group


def to_dataframe(messages: list[Message]) -> pd.DataFrame:
    return pd.DataFrame([
        {
            "timestamp": m.timestamp,
            "sender": m.sender,
            "text": m.text,
            "is_media": m.is_media,
            "char_count": len(m.text),
            "word_count": len(m.text.split()),
        }
        for m in messages
    ])


def detect_format(text: str) -> str:
    """Returns a human-readable description of the detected format for debugging."""
    sample = _clean(text[:2000])
    if IOS_RE.search(sample):
        return "iOS / bracket format"
    if ANDROID_RE.search(sample):
        return "Android standard format"
    return "unknown — may fail to parse"
