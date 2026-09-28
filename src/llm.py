"""
Gemini integration. Single-pass analysis + coaching narrative.
"""
import json
import os
import time
import google.generativeai as genai


def analyze(features: dict, sampled_messages: list[dict]) -> dict:
    """
    Single Gemini call. Returns structured analysis + coaching narrative.
    Retries up to 3 times on rate limit errors.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not set")

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-3.8-flash")
    prompt = _build_prompt(features, sampled_messages)

    for attempt in range(3):
        try:
            response = model.generate_content(prompt)
            return _parse_response(response.text)
        except Exception as e:
            if "429" in str(e) and attempt < 2:
                wait = 30 * (attempt + 1)
                time.sleep(wait)
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
        "reply_times": features["dynamics"]["reply_time_stats"],
        "per_person_stats": {
            s: {
                "avg_message_length_words": features["per_person"][s]["avg_message_length"],
                "questions_asked": features["per_person"][s]["questions_asked"],
                "pronoun_ratios": features["per_person"][s]["pronoun_ratios"],
                "distortion_signals": features["per_person"][s]["distortion_signals"],
                "top_emojis": features["per_person"][s]["top_emojis"],
            }
            for s in senders
        },
        **({"group_roles": features["group"]["participant_roles"]} if is_group else {}),
    }, indent=2)

    return f"""You are analyzing a {chat_type} spanning {features['date_range']['days']} days.

## Computed Statistics
{stats_text}

## Representative Message Samples ({len(messages)} messages selected from the full conversation)
{messages_text}

---

Your task is to produce a deep, honest analysis. Do not be generic. Reference specific moments and actual phrases from the messages above when making observations.

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
  }}{group_roles_field}
}}
</analysis>

<report>
Write the coaching report here. Address each person by name. Be direct — not harsh, but honest. Reference specific things they said. For each person, identify 1-2 patterns worth examining and give a concrete suggestion. Avoid therapy-speak. Write like a smart, perceptive friend who has read the whole conversation.

{group_report_hint}

End with a short section called 'What to try' with 2-3 specific, actionable things.
</report>"""


def _parse_response(text: str) -> dict:
    import re

    analysis_match = re.search(r"<analysis>(.*?)</analysis>", text, re.DOTALL)
    report_match = re.search(r"<report>(.*?)</report>", text, re.DOTALL)

    analysis = {}
    if analysis_match:
        try:
            analysis = json.loads(analysis_match.group(1).strip())
        except json.JSONDecodeError:
            analysis = {"raw": analysis_match.group(1).strip()}

    report = report_match.group(1).strip() if report_match else text

    return {"analysis": analysis, "report": report}
