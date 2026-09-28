"""
Feature extraction from parsed chat data.
All computation happens here — no LLM calls.
"""
import emoji
import pandas as pd
import numpy as np
from collections import defaultdict


DISTRESS_PRONOUNS = {"i", "me", "my", "myself", "mine"}
BONDING_PRONOUNS = {"we", "us", "our", "ours", "ourselves"}
BLAME_PRONOUNS = {"you", "your", "yours", "yourself"}

CATASTROPHIZE_WORDS = {"always", "never", "everything", "nothing", "worst", "ruined", "disaster", "impossible", "hopeless"}
MIND_READ_PHRASES = ["you think", "you feel", "you don't", "you never", "you always", "you obviously", "you clearly", "you don't care", "you hate"]
BLACK_WHITE_WORDS = {"either", "or", "completely", "totally", "absolutely", "perfect", "terrible", "hate", "love"}


def extract_all(df: pd.DataFrame, is_group: bool) -> dict:
    senders = df["sender"].unique().tolist()
    features = {
        "senders": senders,
        "is_group": is_group,
        "total_messages": len(df),
        "date_range": {
            "start": str(df["timestamp"].min().date()),
            "end": str(df["timestamp"].max().date()),
            "days": (df["timestamp"].max() - df["timestamp"].min()).days,
        },
        "per_person": {s: _person_features(df, s) for s in senders},
        "dynamics": _dynamics(df, senders, is_group),
    }
    if is_group:
        features["group"] = _group_features(df, senders)
    return features


def _person_features(df: pd.DataFrame, sender: str) -> dict:
    msgs = df[df["sender"] == sender].copy()
    text_msgs = msgs[~msgs["is_media"]]
    all_text = " ".join(text_msgs["text"].str.lower().tolist())
    words = all_text.split()
    word_count = len(words)

    pronoun_counts = {
        "i_me_my": sum(1 for w in words if w.strip(".,!?") in DISTRESS_PRONOUNS),
        "we_us_our": sum(1 for w in words if w.strip(".,!?") in BONDING_PRONOUNS),
        "you_your": sum(1 for w in words if w.strip(".,!?") in BLAME_PRONOUNS),
    }

    distortions = {
        "catastrophizing": sum(1 for w in words if w.strip(".,!?") in CATASTROPHIZE_WORDS),
        "mind_reading": sum(
            1 for msg in text_msgs["text"].str.lower()
            for phrase in MIND_READ_PHRASES if phrase in msg
        ),
        "black_white_thinking": sum(1 for w in words if w.strip(".,!?") in BLACK_WHITE_WORDS),
    }

    emojis_used = [ch for ch in all_text if ch in emoji.EMOJI_DATA]

    hour_dist = msgs["timestamp"].dt.hour.value_counts().sort_index().to_dict()

    return {
        "message_count": len(msgs),
        "media_count": int(msgs["is_media"].sum()),
        "avg_message_length": float(text_msgs["word_count"].mean()) if len(text_msgs) else 0,
        "longest_message_words": int(text_msgs["word_count"].max()) if len(text_msgs) else 0,
        "total_words": word_count,
        "pronoun_ratios": {
            k: round(v / max(word_count, 1) * 100, 2)
            for k, v in pronoun_counts.items()
        },
        "distortion_signals": distortions,
        "top_emojis": _top_emojis(emojis_used),
        "active_hours": hour_dist,
        "questions_asked": sum(1 for t in text_msgs["text"] if "?" in t),
    }


def _dynamics(df: pd.DataFrame, senders: list, is_group: bool) -> dict:
    df = df.sort_values("timestamp").reset_index(drop=True)

    initiations = defaultdict(int)
    double_texts = defaultdict(int)
    reply_times = defaultdict(list)
    left_on_read = defaultdict(int)

    SESSION_GAP = pd.Timedelta(hours=4)
    prev_sender = None
    prev_time = None
    session_start = None

    for i, row in df.iterrows():
        sender = row["sender"]
        ts = row["timestamp"]

        # session initiation
        if prev_time is None or (ts - prev_time) > SESSION_GAP:
            initiations[sender] += 1
            session_start = ts

        # double-text: same sender twice without reply
        if sender == prev_sender:
            double_texts[sender] += 1

        # reply time: how long before this sender responded to the previous different sender
        if prev_sender and sender != prev_sender and prev_time:
            gap = (ts - prev_time).total_seconds() / 60  # minutes
            if gap < 24 * 60:  # ignore gaps > 24h (not a reply, new session)
                reply_times[sender].append(gap)

        prev_sender = sender
        prev_time = ts

    reply_time_stats = {}
    for sender, times in reply_times.items():
        if times:
            reply_time_stats[sender] = {
                "median_minutes": round(float(np.median(times)), 1),
                "mean_minutes": round(float(np.mean(times)), 1),
                "fast_replies_pct": round(sum(1 for t in times if t < 5) / len(times) * 100, 1),
            }

    share = df["sender"].value_counts(normalize=True).mul(100).round(1).to_dict()

    return {
        "message_share_pct": share,
        "conversation_initiations": dict(initiations),
        "double_texts": dict(double_texts),
        "reply_time_stats": reply_time_stats,
    }


def _group_features(df: pd.DataFrame, senders: list) -> dict:
    """Reply network: who replies to whom."""
    df = df.sort_values("timestamp").reset_index(drop=True)
    reply_to = defaultdict(lambda: defaultdict(int))
    ignored_count = defaultdict(int)

    REPLY_WINDOW = pd.Timedelta(minutes=30)
    prev_sender = None
    prev_time = None

    for _, row in df.iterrows():
        sender = row["sender"]
        ts = row["timestamp"]

        if prev_sender and sender != prev_sender:
            gap = ts - prev_time
            if gap <= REPLY_WINDOW:
                reply_to[sender][prev_sender] += 1
            else:
                ignored_count[prev_sender] += 1

        prev_sender = sender
        prev_time = ts

    reply_matrix = {s: dict(v) for s, v in reply_to.items()}

    # role heuristics
    roles = {}
    total = len(df)
    for s in senders:
        sent = len(df[df["sender"] == s])
        received_replies = sum(reply_matrix.get(r, {}).get(s, 0) for r in senders if r != s)
        sent_replies = sum(reply_to.get(s, {}).values())
        share = sent / total

        if share < 0.05:
            role = "lurker"
        elif sent_replies / max(sent, 1) > 0.7:
            role = "connector"
        elif received_replies / max(sent, 1) > 0.5:
            role = "energizer"
        elif share > 0.3:
            role = "broadcaster"
        else:
            role = "participant"

        roles[s] = role

    return {
        "reply_network": reply_matrix,
        "participant_roles": roles,
        "ignored_messages_count": dict(ignored_count),
    }


def _top_emojis(emojis: list, n: int = 5) -> list:
    from collections import Counter
    return [e for e, _ in Counter(emojis).most_common(n)]
