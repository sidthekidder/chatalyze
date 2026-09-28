"""
Groq LLM integration. Single-pass analysis + coaching narrative.
Uses llama-3.3-70b-versatile — fast, free tier, 14,400 RPD.
"""
import json
import os
import time
from groq import Groq


def analyze(features: dict, sampled_messages: list[dict]) -> dict:
    """
    Single Groq call. Returns structured analysis + coaching narrative.
    Retries up to 3 times on rate limit errors.
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not set")

    client = Groq(api_key=api_key)
    prompt = _build_prompt(features, sampled_messages)

    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model="openai/gpt-oss-120b",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=3000,
            )
            return _parse_response(response.choices[0].message.content)
        except Exception as e:
            if "429" in str(e) and attempt < 2:
                time.sleep(30 * (attempt + 1))
            else:
                raise


def _build_prompt(features: dict, messages: list[dict]) -> str:
    is_group = features["is_group"]
    senders = features["senders"]
    chat_type = "group chat" if is_group else "1:1 conversation"
    group_roles_field = ',\n  "group_roles_observed": "<observations about who plays what role and how group energy flows>"' if is_group else ""
    group_report_hint = "For the group, also describe the group dynamic — who drives energy, who gets ignored, whether the group is functioning well." if is_group else ""

    messages_text = "\n".join(
        f"[{m['time']}] {m['sender']}: {m['text']}"
        for m in messages
    )

    stats_text = json.dumps({
        "date_range": features["date_range"],
        "message_share": features["dynamics"]["message_share_pct"],
        "conversation_initiations": features["dynamics"]["conversation_initiations"],
        "double_texts": features["dynamics"]["double_texts"],
        "left_on_read": features["dynamics"]["left_on_read"],
        "reply_times": features["dynamics"]["reply_time_stats"],
        "conflict_events_count": len(features.get("conflict_events", [])),
        "conflict_events": features.get("conflict_events", [])[:5],
        "per_person_stats": {
            s: {
                "avg_message_length_words": features["per_person"][s]["avg_message_length"],
                "questions_asked": features["per_person"][s]["questions_asked"],
                "pronoun_ratios": features["per_person"][s]["pronoun_ratios"],
                "distortion_signals": features["per_person"][s]["distortion_signals"],
            }
            for s in senders
        },
        **({"group_roles": features["group"]["participant_roles"],
            "group_cohesion": features["group"].get("cohesion", {}).get("interpretation"),
            "topic_ownership": features["group"].get("topic_ownership")} if is_group else {}),
        **({"language_accommodation": features["accommodation"]} if features.get("accommodation") else {}),
        **({"trajectory_summary": {
            "first_period": features["trajectory"][0] if features.get("trajectory") else None,
            "last_period": features["trajectory"][-1] if features.get("trajectory") else None,
        }} if features.get("trajectory") else {}),
    }, indent=2)

    return f"""You are analyzing a {chat_type} spanning {features['date_range']['days']} days.

## Computed Statistics
{stats_text}

## Representative Message Samples ({len(messages)} messages selected from the full conversation)
{messages_text}

---

Your task is to produce a deep, honest analysis. Do not be generic. Reference specific moments and actual phrases from the messages above.

Rules for the coaching report:
- Address each person by name with a heading (e.g. ## PersonName)
- Be direct and specific — name the actual pattern, quote the actual message
- "What to try" suggestions must be derived from THIS conversation, not generic advice
- NEVER suggest "scheduled check-ins", "regular catch-ups", or any generic calendar-based advice unless the data specifically shows communication frequency as a problem
- NEVER use therapy-speak like "create a safe space", "validate feelings", "set intentions"
- Write like a sharp, perceptive friend who has read everything

Respond in this exact format:

<analysis>
{{
  "patterns": [
    {{
      "person": "<sender name or 'group dynamic'>",
      "type": "<pattern type, e.g. catastrophizing, emotional labor, avoidance, etc.>",
      "evidence": "<exact quote or specific moment from the messages>",
      "significance": "<why this matters>"
    }}
  ],
  "dynamics": {{
    "power_balance": "<who leads, who follows, and how>",
    "emotional_labor": "<who carries more, specific evidence>",
    "trajectory": "<is this relationship/group warming, cooling, or stagnant based on the arc of messages>"
  }},
  "distortion_profile": {{
    "<sender name>": {{
      "catastrophizing": {{"severity": 0, "example": "<quote or null>"}},
      "mind_reading": {{"severity": 0, "example": "<quote or null>"}},
      "black_white_thinking": {{"severity": 0, "example": "<quote or null>"}},
      "emotional_reasoning": {{"severity": 0, "example": "<quote or null>"}}
    }}
  }}{group_roles_field}
}}
</analysis>

<report>
## [First person's name]
[Their patterns and suggestions]

## [Second person's name]
[Their patterns and suggestions]

{group_report_hint}

## What to try
2-3 specific, actionable things derived from what you actually saw in this conversation. Not generic. Not scheduled check-ins.
</report>"""


def _parse_response(text: str) -> dict:
    import re

    # Extract analysis block
    analysis = {}
    analysis_match = re.search(r"<analysis>(.*?)</analysis>", text, re.DOTALL)
    if analysis_match:
        raw = analysis_match.group(1).strip()
        # Strip code fences (```json ... ```) that some models add
        raw = re.sub(r"^```(?:json)?\s*\n?", "", raw, flags=re.MULTILINE)
        raw = re.sub(r"\n?```\s*$", "", raw, flags=re.MULTILINE)
        raw = raw.strip()
        try:
            analysis = json.loads(raw)
        except json.JSONDecodeError:
            # Try once more after stripping any trailing commas (common LLM mistake)
            raw_fixed = re.sub(r",\s*([}\]])", r"\1", raw)
            try:
                analysis = json.loads(raw_fixed)
            except json.JSONDecodeError:
                analysis = {}

    # Extract report block — prefer explicit <report> tags
    report = ""
    report_match = re.search(r"<report>(.*?)(?:</report>|$)", text, re.DOTALL)
    if report_match:
        report = report_match.group(1).strip()

    # Fallback: grab everything after </analysis> if no <report> tag
    if not report:
        after_match = re.search(r"</analysis>(.*?)$", text, re.DOTALL)
        if after_match:
            report = after_match.group(1).strip()

    # Last resort: full text (only if analysis also failed — avoids dumping JSON as report)
    if not report and not analysis:
        report = text.strip()

    # Strip generic title headers (but not person names)
    report = re.sub(
        r"^\*\*(Coaching Report|Personal Coaching Report|Analysis Report|Report)\*\*\s*\n?",
        "", report, flags=re.IGNORECASE
    ).strip()

    return {"analysis": analysis, "report": report}
