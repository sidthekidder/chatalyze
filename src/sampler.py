"""
Strategic message sampler.
Divides the chat into temporal buckets and samples proportionally from each,
so the LLM sees the full arc of the relationship rather than being skewed
by recent interactions.
"""
import pandas as pd


# Budget breakdown out of n=100:
# - anchors (first + last few): 10
# - longest messages per person: 10
# - signal messages (distortion keywords): 15
# - temporal buckets (bulk): 65

DISTRESS_WORDS = {
    "always", "never", "hate", "worst", "disaster", "impossible",
    "hopeless", "ruined", "everything", "nothing", "you don't", "you never",
}
N_BUCKETS = 5  # divide timeline into 5 equal periods


def sample(df: pd.DataFrame, features: dict, n: int = 100) -> list[dict]:
    text_df = df[~df["is_media"] & (df["text"].str.len() > 0)].copy()
    if len(text_df) == 0:
        return []

    selected_indices = set()

    # 1. Anchors: first 5 and last 5 only (not 15 — avoids recency bias)
    for idx in text_df.head(5).index:
        selected_indices.add(idx)
    for idx in text_df.tail(5).index:
        selected_indices.add(idx)

    # 2. Longest messages per person (up to 3 each — high signal, reveals thinking)
    for sender in features["senders"]:
        sender_msgs = text_df[text_df["sender"] == sender]
        for idx in sender_msgs.nlargest(3, "word_count").index:
            selected_indices.add(idx)

    # 3. Signal messages containing distortion/conflict keywords
    signal_budget = 15
    signal_count = 0
    for idx, row in text_df.iterrows():
        if idx in selected_indices:
            continue
        if any(w in row["text"].lower() for w in DISTRESS_WORDS):
            selected_indices.add(idx)
            signal_count += 1
            if signal_count >= signal_budget:
                break

    # 4. Temporal bucketing: split timeline into N_BUCKETS equal periods,
    #    sample proportionally from each so every era of the relationship is represented
    bucket_budget = n - len(selected_indices)
    if bucket_budget > 0:
        per_bucket = max(1, bucket_budget // N_BUCKETS)
        total_msgs = len(text_df)
        bucket_size = total_msgs // N_BUCKETS

        for i in range(N_BUCKETS):
            start = i * bucket_size
            end = start + bucket_size if i < N_BUCKETS - 1 else total_msgs
            bucket = text_df.iloc[start:end]
            eligible = bucket[~bucket.index.isin(selected_indices)]
            if len(eligible) == 0:
                continue
            k = min(per_bucket, len(eligible))
            # weight longer messages slightly higher within each bucket
            weights = eligible["word_count"].clip(lower=1)
            sampled_bucket = eligible.sample(k, weights=weights, random_state=42 + i)
            for idx in sampled_bucket.index:
                selected_indices.add(idx)

    sampled = text_df.loc[sorted(selected_indices)].head(n)

    return [
        {
            "sender": row["sender"],
            "time": str(row["timestamp"].strftime("%Y-%m-%d %H:%M")),
            "text": row["text"][:500],
        }
        for _, row in sampled.iterrows()
    ]
