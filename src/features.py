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

CATASTROPHIZE_WORDS = {"everything", "nothing", "worst", "ruined", "disaster", "impossible", "hopeless", "doomed", "destroyed", "catastrophe", "terrible", "unbearable"}
MIND_READ_PHRASES = ["you think", "you feel", "you don't", "you never", "you always", "you obviously", "you clearly", "you don't care", "you hate"]
BLACK_WHITE_WORDS = {"either", "or", "completely", "totally", "absolutely", "perfect", "terrible", "hate", "love"}

PERSONALIZATION_PHRASES = [
    "my fault", "i ruined", "because of me", "i caused", "i'm the reason",
    "i always mess", "i messed up", "all my fault", "i should have",
    "i'm the problem", "i'm sorry for being",
]
REPAIR_WORDS = {
    "sorry", "apologize", "apologies", "forgive", "my bad",
    "youre right", "you're right", "i was wrong", "my mistake",
    "i understand", "lets", "let's", "i hear you",
}
COGNITIVE_WORDS = {
    "realize", "realized", "understand", "understood", "figured",
    "learned", "discovered", "recognize", "noticed",
}

COMFORT_PHRASES = [
    "are you okay", "you okay", "how are you", "that sounds hard", "that must be",
    "i'm sorry to hear", "must be tough", "i'm here", "here for you",
    "you alright", "how are you feeling", "you doing okay", "tell me what happened",
    "that makes sense", "i understand how", "must be difficult",
]
VENTING_PHRASES = [
    "i can't deal", "i'm so stressed", "so exhausted", "so tired of",
    "i hate this", "this is too much", "i can't take", "overwhelmed",
    "i'm done", "can't anymore", "so frustrated", "nothing works",
    "i give up", "i'm breaking", "everything is wrong",
]

CONFLICT_EXTRA = {"wtf", "seriously", "unbelievable", "ridiculous", "whatever", "fine", "stop it",
                  "leave me", "forget it", "never mind", "omg", "shut up", "enough"}
POSITIVE_WORDS = {"love", "great", "amazing", "good", "happy", "excited", "thanks", "thank",
                  "appreciate", "wonderful", "awesome", "perfect", "haha", "lol", "nice", "cool",
                  "fun", "enjoy", "glad", "yes", "sure", "brilliant", "miss", "proud"}

FUTURE_WORDS = {
    "will", "gonna", "going", "would", "could", "should",
    "tomorrow", "soon", "later", "next", "eventually", "someday",
    "hope", "plan", "promise", "try", "shall",
}

# Emoji buckets — variation selectors stripped before comparison
EMOJI_AFFECTIVE = {
    "❤", "🥰", "😊", "✨", "😍", "💕", "💖", "💗", "💓", "💞", "💝",
    "🫶", "🤗", "😘", "🥺", "💛", "💚", "💙", "💜", "🖤", "🤍", "🤎",
    "❣", "💟", "😻", "🌟", "⭐", "🥹",
}
EMOJI_NEGATIVE = {
    "😭", "😤", "😡", "😢", "😠", "😩", "😫", "😟", "😔", "💔", "😞",
    "😖", "😣", "😰", "😥", "😓", "😿", "😾", "🙁", "☹", "😒",
    "😧", "😨", "🤬", "😱",
}
EMOJI_SOFTENING = {
    "😅", "😬", "😂", "🙏", "🤭", "😆", "😁", "🤣", "😜", "😝",
    "🤪", "🫠", "🙃", "😏", "🫢", "🤷",
}

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
        "conflict_events": _conflict_patterns(df, senders),
        "activity_patterns": _activity_patterns(df, senders),
        "rt_trend": _rt_trend(df, senders),
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

    word_count_100 = max(word_count / 100, 1)
    distortions = {
        "catastrophizing_per100": round(
            sum(1 for w in words if w.strip(".,!?") in CATASTROPHIZE_WORDS) / word_count_100, 2
        ),
        "black_white_per100": round(
            sum(1 for w in words if w.strip(".,!?") in BLACK_WHITE_WORDS) / word_count_100, 2
        ),
        "personalization": sum(
            1 for msg in text_msgs["text"].str.lower()
            if any(phrase in msg for phrase in PERSONALIZATION_PHRASES)
        ),
        "repair_attempts": sum(
            1 for msg in text_msgs["text"].str.lower()
            if any(w in msg.split() for w in REPAIR_WORDS)
        ),
        "cognitive_complexity": sum(
            1 for msg in text_msgs["text"].str.lower()
            if any(w in msg.split() for w in COGNITIVE_WORDS)
        ),
    }

    # Use emoji_list for proper multi-codepoint extraction
    emojis_used = [e["emoji"] for e in emoji.emoji_list(all_text)]
    emoji_prof = _emoji_profile(emojis_used)
    emoji_prof["emoji_to_word_ratio"] = round(len(emojis_used) / max(word_count, 1), 4)

    future_focus = sum(1 for w in words if w.strip(".,!?") in FUTURE_WORDS)

    hour_dist = msgs["timestamp"].dt.hour.value_counts().sort_index().to_dict()

    emotional_labor = {
        "comfort_given": sum(
            1 for msg in text_msgs["text"].str.lower()
            if any(phrase in msg for phrase in COMFORT_PHRASES)
        ),
        "venting_messages": sum(
            1 for msg in text_msgs["text"].str.lower()
            if any(phrase in msg for phrase in VENTING_PHRASES)
        ),
    }

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
        "emotional_labor": emotional_labor,
        "future_focus": future_focus,
        "emoji_profile": emoji_prof,
        "top_emojis": _top_emojis(emojis_used),
        "active_hours": hour_dist,
        "questions_asked": sum(1 for t in text_msgs["text"] if "?" in t),
    }


def _dynamics(df: pd.DataFrame, senders: list, is_group: bool) -> dict:
    df = df.sort_values("timestamp").reset_index(drop=True)

    initiations = defaultdict(int)
    double_texts = defaultdict(int)
    reply_times = defaultdict(list)
    reply_times_daytime = defaultdict(list)  # 8am-10pm only (context-normalized)
    left_on_read = defaultdict(int)

    SESSION_GAP = pd.Timedelta(hours=4)
    LEFT_ON_READ_GAP = pd.Timedelta(minutes=15)
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

        if sender == prev_sender and prev_time:
            # double-text: same sender twice without reply
            double_texts[sender] += 1
            # left on read: same sender follows up after a real wait (not just rapid-fire)
            if (ts - prev_time) > LEFT_ON_READ_GAP:
                left_on_read[sender] += 1

        # reply time: how long before this sender responded to the previous different sender
        if prev_sender and sender != prev_sender and prev_time:
            gap = (ts - prev_time).total_seconds() / 60  # minutes
            if gap < 24 * 60:  # ignore gaps > 24h (not a reply, new session)
                reply_times[sender].append(gap)
                if 8 <= prev_time.hour < 22:  # daytime-normalized: exclude night sends
                    reply_times_daytime[sender].append(gap)

        prev_sender = sender
        prev_time = ts

    reply_time_stats = {}
    for sender, times in reply_times.items():
        if times:
            med = float(np.median(times))
            daytime = reply_times_daytime.get(sender, [])
            reply_time_stats[sender] = {
                "median_minutes": round(med, 1),
                "mean_minutes": round(float(np.mean(times)), 1),
                "fast_replies_pct": round(sum(1 for t in times if t < 5) / len(times) * 100, 1),
                "variance_minutes": round(float(np.std(times)), 1),
                "spike_count": sum(1 for t in times if t > max(med * 3, 60)),
                "daytime_median_minutes": round(float(np.median(daytime)), 1) if daytime else None,
            }

    # RT asymmetry: for 1:1 chats, who waits longer to reply to whom
    rt_asymmetry = None
    if len(senders) == 2 and all(s in reply_time_stats for s in senders):
        s1, s2 = senders[0], senders[1]
        m1 = reply_time_stats[s1]["median_minutes"]
        m2 = reply_time_stats[s2]["median_minutes"]
        if abs(m1 - m2) > 2:
            slower = s1 if m1 > m2 else s2
            faster = s2 if m1 > m2 else s1
            ratio = round(max(m1, m2) / max(min(m1, m2), 0.1), 1)
            rt_asymmetry = f"{slower} takes {ratio}× longer to reply than {faster} on average."
        else:
            rt_asymmetry = "Both reply at similar speeds."

    share = df["sender"].value_counts(normalize=True).mul(100).round(1).to_dict()

    return {
        "message_share_pct": share,
        "conversation_initiations": dict(initiations),
        "double_texts": dict(double_texts),
        "reply_time_stats": reply_time_stats,
        "reply_time_asymmetry": rt_asymmetry,
        "left_on_read": dict(left_on_read),
    }


def _emoji_profile(emojis: list) -> dict:
    """Classify extracted emoji into affective, negative-expressive, softening buckets."""
    normed = [e.replace('️', '').replace('︎', '') for e in emojis]
    affective = sum(1 for e in normed if e in EMOJI_AFFECTIVE)
    negative = sum(1 for e in normed if e in EMOJI_NEGATIVE)
    softening = sum(1 for e in normed if e in EMOJI_SOFTENING)
    total = len(normed)
    return {
        "total": total,
        "affective": affective,
        "negative_expressive": negative,
        "softening_hedging": softening,
    }


def _rt_trend(df: pd.DataFrame, senders: list) -> list:
    """Reply time trend across early / mid / recent thirds of the conversation."""
    df = df.sort_values("timestamp").reset_index(drop=True)
    n = len(df)
    thirds = [
        ("early", df.iloc[:n // 3]),
        ("mid", df.iloc[n // 3: 2 * n // 3]),
        ("recent", df.iloc[2 * n // 3:]),
    ]
    result = []
    for label, chunk in thirds:
        chunk = chunk.reset_index(drop=True)
        rt_per_sender = defaultdict(list)
        prev_sender = None
        prev_time = None
        for _, row in chunk.iterrows():
            sender = row["sender"]
            ts = row["timestamp"]
            if prev_sender and sender != prev_sender and prev_time:
                gap = (ts - prev_time).total_seconds() / 60
                if gap < 24 * 60 and 8 <= prev_time.hour < 22:
                    rt_per_sender[sender].append(gap)
            prev_sender = sender
            prev_time = ts
        result.append({
            "period": label,
            "median_rt": {
                s: round(float(np.median(v)), 1) if v else None
                for s, v in rt_per_sender.items()
            },
        })
    return result


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
        all_words = " ".join(group[~group["is_media"]]["text"].str.lower()).split()
        clean = [w.strip(".,!?") for w in all_words]
        pos = sum(1 for w in clean if w in POSITIVE_WORDS)
        neg = sum(1 for w in clean if w in CATASTROPHIZE_WORDS | BLACK_WHITE_WORDS | CONFLICT_EXTRA)
        sentiment = round(pos / (pos + neg), 3) if (pos + neg) > 0 else 0.5

        per_person = {}
        for s in senders:
            s_msgs = group[group["sender"] == s]
            text_msgs = s_msgs[~s_msgs["is_media"]]
            words = " ".join(text_msgs["text"].str.lower()).split()
            distortions = (
                sum(1 for w in words if w.strip(".,!?") in CATASTROPHIZE_WORDS)
                + sum(1 for w in words if w.strip(".,!?") in BLACK_WHITE_WORDS)
            )
            per_person[s] = {
                "message_count": len(s_msgs),
                "message_share_pct": round(len(s_msgs) / max(len(group), 1) * 100, 1),
                "distortion_signals": distortions,
                "i_ratio": round(sum(1 for w in words if w.strip(".,!?") in DISTRESS_PRONOUNS) / max(len(words), 1) * 100, 2),
                "we_ratio": round(sum(1 for w in words if w.strip(".,!?") in BONDING_PRONOUNS) / max(len(words), 1) * 100, 2),
            }
        buckets.append({"label": _period_label(period), "sentiment": sentiment, "per_person": per_person})
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

    # LSM asymmetry: per-exchange accommodation (Danescu-Niculescu-Mizil 2012)
    # Who mirrors the other more per reply = accommodates more = may signal status/investment
    text_df_sorted = text_df.sort_values("timestamp").reset_index(drop=True)
    accomm_per_person = {s: [] for s in senders}
    for i in range(1, len(text_df_sorted)):
        curr = text_df_sorted.iloc[i]
        prev = text_df_sorted.iloc[i - 1]
        if curr["sender"] != prev["sender"]:
            sim = _cosine(_fw_vector(curr["text"]), _fw_vector(prev["text"]))
            accomm_per_person[curr["sender"]].append(sim)

    asymmetry = {
        s: round(float(np.mean(v)), 3) if v else 0.0
        for s, v in accomm_per_person.items()
    }
    if len(senders) == 2:
        s1, s2 = senders[0], senders[1]
        diff = asymmetry.get(s1, 0.0) - asymmetry.get(s2, 0.0)
        if abs(diff) > 0.02:
            higher = s1 if diff > 0 else s2
            lower = s2 if diff > 0 else s1
            asym_interp = f"{higher} mirrors {lower}'s language more — may signal higher investment or lower perceived status."
        else:
            asym_interp = "Both accommodate each other at similar rates."
    else:
        highest = max(asymmetry, key=asymmetry.get) if asymmetry else ""
        asym_interp = f"{highest} shows the highest language mirroring in the group." if highest else ""

    # Emoji register reciprocity: do partners use similar emoji valence profiles?
    def _emoji_vec(text):
        emojis = [e["emoji"] for e in emoji.emoji_list(text)]
        if not emojis:
            return np.zeros(3)
        total = len(emojis)
        normed = [e.replace('️', '') for e in emojis]
        return np.array([
            sum(1 for e in normed if e in EMOJI_AFFECTIVE) / total,
            sum(1 for e in normed if e in EMOJI_NEGATIVE) / total,
            sum(1 for e in normed if e in EMOJI_SOFTENING) / total,
        ], dtype=float)

    ev1 = _emoji_vec(texts[senders[0]])
    ev2 = _emoji_vec(texts[senders[1]])
    emoji_reciprocity = round(_cosine(ev1, ev2), 3) if (np.any(ev1) and np.any(ev2)) else None

    return {
        "overall_score": overall,
        "trend": trend,
        "interpretation": interp,
        "asymmetry": asymmetry,
        "asymmetry_interpretation": asym_interp,
        "emoji_reciprocity": emoji_reciprocity,
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

    # role heuristics (research-grounded: diversity + volume + initiation)
    roles = {}
    reply_diversity = {}
    total = len(df)
    for s in senders:
        sent = len(df[df["sender"] == s])
        received_replies = sum(reply_matrix.get(r, {}).get(s, 0) for r in senders if r != s)
        sent_replies = sum(reply_to.get(s, {}).values())
        unique_reply_targets = len(reply_to.get(s, {}))
        share = sent / total
        # diversity: fraction of other members this person replies to
        diversity = unique_reply_targets / max(len(senders) - 1, 1)
        reply_diversity[s] = {
            "unique_reply_targets": unique_reply_targets,
            "diversity_score": round(diversity, 3),
        }

        if share < 0.05:
            role = "lurker"
        elif diversity > 0.6 and sent_replies / max(sent, 1) > 0.4:
            role = "connector"  # replies to many unique people
        elif received_replies / max(sent, 1) > 0.5:
            role = "energizer"
        elif share > 0.3 and diversity < 0.4:
            role = "broadcaster"  # sends a lot but replies to few unique people
        else:
            role = "participant"

        roles[s] = role

    cohesion = _group_cohesion(df, senders, reply_matrix)
    topic_own = _topic_ownership(df, senders)
    subgroups = _subgroups(reply_matrix, senders)

    return {
        "reply_network": reply_matrix,
        "participant_roles": roles,
        "reply_diversity": reply_diversity,
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


def _conflict_patterns(df: pd.DataFrame, senders: list) -> list:
    """Detect conflict windows: bursts of 3+ distress signals within 45 minutes."""
    df = df.sort_values("timestamp").reset_index(drop=True)
    text_df = df[~df["is_media"] & (df["text"].str.len() > 0)].copy()

    ALL_CONFLICT = CATASTROPHIZE_WORDS | BLACK_WHITE_WORDS | CONFLICT_EXTRA

    def conflict_score(text: str) -> int:
        t = text.lower()
        words = [w.strip(".,!?") for w in t.split()]
        hits = sum(1 for w in words if w in ALL_CONFLICT)
        hits += sum(1 for ph in MIND_READ_PHRASES if ph in t)
        hits += 2 if sum(1 for c in text if c.isupper()) / max(len(text), 1) > 0.35 else 0
        return hits

    text_df["cscore"] = text_df["text"].apply(conflict_score)
    conflict_msgs = text_df[text_df["cscore"] > 0].copy()

    WINDOW = pd.Timedelta(minutes=45)
    MIN_SIGNALS = 3
    events = []
    processed_until = pd.Timestamp.min.tz_localize(None)

    for _, row in conflict_msgs.iterrows():
        ts = row["timestamp"]
        if hasattr(ts, "tzinfo") and ts.tzinfo is not None:
            ts = ts.tz_localize(None) if processed_until.tzinfo is None else ts
        if ts <= processed_until:
            continue

        window_end = ts + WINDOW
        in_window = conflict_msgs[
            (conflict_msgs["timestamp"] >= ts) &
            (conflict_msgs["timestamp"] <= window_end)
        ]

        if in_window["cscore"].sum() >= MIN_SIGNALS:
            ctx = text_df[
                (text_df["timestamp"] >= ts) &
                (text_df["timestamp"] <= window_end)
            ]
            initiator = in_window.iloc[0]["sender"]
            # De-escalator: last unique sender who replied after the peak
            unique_senders = ctx["sender"].tolist()
            de_escalator = unique_senders[-1] if unique_senders else None

            events.append({
                "start": ts.strftime("%Y-%m-%d %H:%M"),
                "initiator": initiator,
                "de_escalator": de_escalator if de_escalator != initiator else None,
                "intensity": round(in_window["cscore"].sum() / max(len(ctx), 1), 2),
                "message_count": len(ctx),
                "example": row["text"][:120],
            })
            processed_until = window_end

    return events


def _activity_patterns(df: pd.DataFrame, senders: list) -> dict:
    """Per-person message timing: peak hour, night owl %, weekend %, activity label."""
    result = {}
    for s in senders:
        msgs = df[df["sender"] == s]
        if len(msgs) == 0:
            continue
        hours = msgs["timestamp"].dt.hour
        total = len(msgs)
        hour_counts = hours.value_counts().sort_index()
        peak_hour = int(hour_counts.idxmax())
        night_pct = round(sum(1 for h in hours if h >= 23 or h < 4) / total * 100, 1)
        morning_pct = round(sum(1 for h in hours if 5 <= h < 9) / total * 100, 1)
        weekend_pct = round(sum(1 for d in msgs["timestamp"].dt.dayofweek if d >= 5) / total * 100, 1)

        if night_pct > 20:
            label = "night owl"
        elif morning_pct > 20:
            label = "early bird"
        else:
            label = "daytime"

        result[s] = {
            "peak_hour": peak_hour,
            "night_pct": night_pct,
            "morning_pct": morning_pct,
            "weekend_pct": weekend_pct,
            "activity_label": label,
            "hourly": {int(h): int(c) for h, c in hour_counts.items()},
        }
    return result


def _top_emojis(emojis: list, n: int = 5) -> list:
    from collections import Counter
    return [e for e, _ in Counter(emojis).most_common(n)]
