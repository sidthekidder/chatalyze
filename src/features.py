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

FUNCTION_WORDS = sorted({
    "the", "a", "an", "and", "but", "or", "so", "if", "not", "no",
    "is", "are", "was", "were", "be", "been", "have", "has", "had",
    "do", "does", "did", "will", "would", "can", "could", "should",
    "in", "on", "at", "to", "for", "of", "with", "by", "from", "into",
    "i", "you", "it", "this", "that", "he", "she", "we", "they",
    "me", "him", "her", "us", "them", "my", "your", "his", "our", "their",
    "just", "like", "get", "know", "think", "what", "how", "when", "where",
})


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
        "trajectory": _trajectory(df, senders),
    }
    if not is_group and len(senders) == 2:
        features["accommodation"] = _language_accommodation(df, senders)
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


def _period_label(p: pd.Period) -> str:
    ts = p.to_timestamp()
    if p.freqstr.startswith("Q"):
        return f"Q{(ts.month - 1) // 3 + 1} {ts.year}"
    if p.freqstr.startswith("M"):
        return ts.strftime("%b %Y")
    return ts.strftime("Week of %b %d")


def _trajectory(df: pd.DataFrame, senders: list) -> list:
    """Split chat into time periods, compute per-person metrics for each."""
    days = (df["timestamp"].max() - df["timestamp"].min()).days
    if days < 60:
        freq = "W"
    elif days < 547:
        freq = "M"
    else:
        freq = "Q"

    df = df.copy()
    df["_period"] = df["timestamp"].dt.to_period(freq)

    buckets = []
    for period, group in df.groupby("_period"):
        if len(group) < 5:
            continue
        per_person = {}
        for s in senders:
            s_msgs = group[group["sender"] == s]
            text_msgs = s_msgs[~s_msgs["is_media"]]
            words = " ".join(text_msgs["text"].str.lower()).split()
            distortions = (
                sum(1 for w in words if w.strip(".,!?") in CATASTROPHIZE_WORDS)
                + sum(1 for m in text_msgs["text"].str.lower() for ph in MIND_READ_PHRASES if ph in m)
                + sum(1 for w in words if w.strip(".,!?") in BLACK_WHITE_WORDS)
            )
            per_person[s] = {
                "message_count": len(s_msgs),
                "message_share_pct": round(len(s_msgs) / max(len(group), 1) * 100, 1),
                "distortion_signals": distortions,
                "i_ratio": round(sum(1 for w in words if w.strip(".,!?") in DISTRESS_PRONOUNS) / max(len(words), 1) * 100, 2),
                "we_ratio": round(sum(1 for w in words if w.strip(".,!?") in BONDING_PRONOUNS) / max(len(words), 1) * 100, 2),
            }
        buckets.append({"label": _period_label(period), "per_person": per_person})
    return buckets


def _fw_vector(text: str) -> np.ndarray:
    words = [w.strip(".,!?;:\"'") for w in text.lower().split()]
    counts = np.array([words.count(fw) for fw in FUNCTION_WORDS], dtype=float)
    norm = np.linalg.norm(counts)
    return counts / norm if norm > 0 else counts


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 0 else 0.0


def _language_accommodation(df: pd.DataFrame, senders: list) -> dict:
    """How similar are the two speakers' function-word distributions? Tracks over time."""
    text_df = df[~df["is_media"]].copy()

    texts = {s: " ".join(text_df[text_df["sender"] == s]["text"].str.lower()) for s in senders}
    overall = round(_cosine(_fw_vector(texts[senders[0]]), _fw_vector(texts[senders[1]])), 3)

    days = (df["timestamp"].max() - df["timestamp"].min()).days
    freq = "W" if days < 60 else ("M" if days < 547 else "Q")
    text_df["_period"] = text_df["timestamp"].dt.to_period(freq)

    trend = []
    for period, group in text_df.groupby("_period"):
        vecs = {}
        for s in senders:
            t = " ".join(group[group["sender"] == s]["text"].str.lower())
            if t.strip():
                vecs[s] = _fw_vector(t)
        if len(vecs) == 2:
            score = round(_cosine(vecs[senders[0]], vecs[senders[1]]), 3)
            trend.append({"label": _period_label(period), "score": score})

    if overall > 0.85:
        interp = "Very high alignment — you've absorbed each other's communication style."
    elif overall > 0.70:
        interp = "Strong accommodation — language patterns are converging."
    elif overall > 0.55:
        interp = "Moderate alignment with distinct individual styles."
    else:
        interp = "Low accommodation — you communicate in quite different registers."

    return {"overall_score": overall, "trend": trend, "interpretation": interp}


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

    cohesion = _group_cohesion(df, senders, reply_matrix)
    topic_own = _topic_ownership(df, senders)
    subgroups = _subgroups(reply_matrix, senders)

    return {
        "reply_network": reply_matrix,
        "participant_roles": roles,
        "ignored_messages_count": dict(ignored_count),
        "cohesion": cohesion,
        "topic_ownership": topic_own,
        "subgroups": subgroups,
    }


def _group_cohesion(df: pd.DataFrame, senders: list, reply_network: dict) -> dict:
    shares = np.array([len(df[df["sender"] == s]) for s in senders], dtype=float)
    shares = shares / shares.sum()
    entropy = float(-np.sum(shares * np.log(shares + 1e-10)))
    max_entropy = np.log(len(senders))
    participation = entropy / max_entropy if max_entropy > 0 else 0.0

    total_replies = sum(sum(v.values()) for v in reply_network.values())
    reply_ratio = min(total_replies / max(len(df), 1), 1.0)
    overall = round((participation + reply_ratio) / 2, 3)

    if overall > 0.65:
        interp = "High cohesion — everyone participates and conversations are reciprocal."
    elif overall > 0.40:
        interp = "Moderate cohesion — some members carry the conversation more than others."
    else:
        interp = "Low cohesion — conversation is dominated or mostly one-directional."

    return {
        "participation_balance": round(participation, 3),
        "reply_ratio": round(reply_ratio, 3),
        "overall_cohesion": overall,
        "interpretation": interp,
    }


def _topic_ownership(df: pd.DataFrame, senders: list) -> dict:
    """Who starts exchanges that get a quick reply (within 5 min)?"""
    df = df.sort_values("timestamp").reset_index(drop=True)
    WINDOW = pd.Timedelta(minutes=5)
    traction = defaultdict(int)
    ignored = defaultdict(int)

    for i in range(len(df) - 1):
        row, nxt = df.iloc[i], df.iloc[i + 1]
        if nxt["sender"] != row["sender"] and (nxt["timestamp"] - row["timestamp"]) <= WINDOW:
            traction[row["sender"]] += 1
        else:
            ignored[row["sender"]] += 1

    result = {}
    for s in senders:
        t, ig = traction.get(s, 0), ignored.get(s, 0)
        result[s] = {
            "messages_with_traction": t,
            "traction_rate_pct": round(t / max(t + ig, 1) * 100, 1),
        }
    return result


def _subgroups(reply_network: dict, senders: list) -> list:
    """Detect cliques from reply patterns — only meaningful in groups of 4+."""
    if len(senders) < 4:
        return []

    edges = defaultdict(float)
    for replier, targets in reply_network.items():
        for target, count in targets.items():
            key = tuple(sorted([replier, target]))
            edges[key] += count

    subgroups: list[set] = []
    assigned: set = set()
    for (a, b), weight in sorted(edges.items(), key=lambda x: -x[1]):
        if weight < 3:
            continue
        found = next((sg for sg in subgroups if a in sg or b in sg), None)
        if found is not None:
            found.update([a, b])
        else:
            subgroups.append({a, b})
        assigned.update([a, b])

    loners = [s for s in senders if s not in assigned]
    return [sorted(sg) for sg in subgroups] + [[s] for s in loners]


def _top_emojis(emojis: list, n: int = 5) -> list:
    from collections import Counter
    return [e for e, _ in Counter(emojis).most_common(n)]
