import os
import json
import re
import pandas as pd
import streamlit as st
from datetime import date
from dotenv import load_dotenv

load_dotenv()

from src.parser import parse, to_dataframe, detect_format
from src.features import extract_all
from src.sampler import sample
from src.llm import analyze
from src.storage import save_report, load_report

APP_URL = os.getenv("APP_URL", "https://chatalyze.streamlit.app")


def _md_to_html(text: str) -> str:
    text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"\*(.+?)\*", r"<em>\1</em>", text)
    text = re.sub(r"^### (.+)$", r"<h4>\1</h4>", text, flags=re.MULTILINE)
    text = re.sub(r"^## (.+)$", r"<h3>\1</h3>", text, flags=re.MULTILINE)
    text = re.sub(r"^# (.+)$", r"<h2>\1</h2>", text, flags=re.MULTILINE)
    text = re.sub(r"^[-*] (.+)$", r"<li>\1</li>", text, flags=re.MULTILINE)
    text = re.sub(r"(<li>.*?</li>(\n|$))+", lambda m: f"<ul>{m.group(0)}</ul>", text, flags=re.DOTALL)
    paragraphs = [f"<p>{p.strip()}</p>" if not p.strip().startswith("<") else p.strip()
                  for p in text.split("\n\n") if p.strip()]
    return "\n".join(paragraphs)


def _build_html_report(features: dict, result: dict) -> str:
    patterns = result["analysis"].get("patterns", [])
    dynamics = result["analysis"].get("dynamics", {})
    report_html = _md_to_html(result["report"]) if result.get("report") else ""

    patterns_html = ""
    for p in patterns:
        evidence_html = f'<blockquote>{p["evidence"]}</blockquote>' if p.get("evidence") else ""
        patterns_html += f"""
        <div class="pattern">
            <div class="pattern-header">
                <span class="person">{p.get('person', '')}</span>
                <span class="type">{p.get('type', '')}</span>
            </div>
            {evidence_html}
            <p>{p.get("significance", "")}</p>
        </div>"""

    dynamics_html = ""
    for label, key in [("Balance", "power_balance"), ("Emotional labor", "emotional_labor"), ("Trajectory", "trajectory")]:
        if dynamics.get(key):
            dynamics_html += f'<div class="dynamic"><strong>{label}</strong><p>{dynamics[key]}</p></div>'

    senders = features.get("senders", [])
    stats_html = ""
    for s in senders:
        p = features["per_person"].get(s, {})
        rt = features["dynamics"]["reply_time_stats"].get(s, {})
        rt_html = f'<p>Median reply time: <strong>{rt["median_minutes"]} min</strong></p>' if rt else ""
        stats_html += f"""
        <div class="stat-card">
            <h4>{s}</h4>
            <p>{p.get('message_count', 0):,} messages &nbsp;·&nbsp; avg {p.get('avg_message_length', 0):.0f} words each</p>
            {rt_html}
            <p>Questions asked: {p.get('questions_asked', 0)}</p>
        </div>"""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Chatalyze Report</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #f9f9f9; color: #1a1a1a; line-height: 1.7; }}
  .container {{ max-width: 780px; margin: 0 auto; padding: 48px 24px; }}
  h1 {{ font-size: 2rem; font-weight: 700; margin-bottom: 4px; }}
  h2 {{ font-size: 1.25rem; font-weight: 600; margin: 40px 0 16px; border-bottom: 2px solid #eee; padding-bottom: 8px; }}
  h3, h4 {{ font-weight: 600; margin-bottom: 8px; }}
  .meta {{ color: #666; font-size: 0.9rem; margin-bottom: 40px; }}
  .stat-cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 16px; margin-bottom: 8px; }}
  .stat-card {{ background: #fff; border: 1px solid #e5e5e5; border-radius: 10px; padding: 18px; }}
  .stat-card h4 {{ font-size: 1rem; margin-bottom: 6px; }}
  .stat-card p {{ font-size: 0.88rem; color: #444; margin-top: 4px; }}
  .pattern {{ background: #fff; border: 1px solid #e5e5e5; border-radius: 10px; padding: 20px; margin-bottom: 12px; }}
  .pattern-header {{ display: flex; gap: 12px; align-items: center; margin-bottom: 10px; }}
  .person {{ font-weight: 600; }}
  .type {{ background: #6C63FF22; color: #6C63FF; padding: 2px 10px; border-radius: 20px; font-size: 0.82rem; }}
  blockquote {{ border-left: 3px solid #6C63FF; padding: 8px 16px; margin: 10px 0; color: #555; font-style: italic; background: #f5f4ff; border-radius: 0 6px 6px 0; }}
  .dynamics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 16px; }}
  .dynamic {{ background: #fff; border: 1px solid #e5e5e5; border-radius: 10px; padding: 18px; }}
  .dynamic p {{ font-size: 0.9rem; color: #444; margin-top: 6px; }}
  .report {{ background: #fff; border: 1px solid #e5e5e5; border-radius: 10px; padding: 28px; }}
  .report h2, .report h3 {{ border: none; margin-top: 20px; }}
  .report p {{ margin: 12px 0; }}
  .report ul {{ padding-left: 20px; }}
  .footer {{ margin-top: 48px; font-size: 0.8rem; color: #999; text-align: center; }}
</style>
</head>
<body>
<div class="container">
  <h1>Chatalyze Report</h1>
  <p class="meta">Generated {date.today().strftime('%B %d, %Y')} &nbsp;·&nbsp; {features['date_range']['days']} days of conversation &nbsp;·&nbsp; {features['total_messages']:,} messages</p>

  <h2>At a glance</h2>
  <div class="stat-cards">{stats_html}</div>

  {"<h2>Patterns identified</h2>" + patterns_html if patterns_html else ""}

  {"<h2>Relationship dynamics</h2><div class='dynamics'>" + dynamics_html + "</div>" if dynamics_html else ""}

  <h2>Coaching report</h2>
  <div class="report">{report_html}</div>

  <div class="footer">Made with Chatalyze &nbsp;·&nbsp; chatalyze.app</div>
</div>
</body>
</html>"""


def _render_analysis(result: dict, features: dict, is_shared: bool = False) -> None:
    import plotly.express as px

    if not is_shared:
        st.success("Done")
    st.divider()

    patterns = result["analysis"].get("patterns", [])
    if patterns:
        st.subheader("Patterns identified")
        for p in patterns:
            with st.expander(f"**{p.get('person', '?')}** — {p.get('type', '')}"):
                if p.get("evidence"):
                    st.markdown(f"> *\"{p['evidence']}\"*")
                if p.get("significance"):
                    st.markdown(p["significance"])

    dynamics = result["analysis"].get("dynamics", {})
    if dynamics:
        st.subheader("Relationship dynamics")
        cols = st.columns(3)
        if dynamics.get("power_balance"):
            cols[0].markdown(f"**Balance**\n\n{dynamics['power_balance']}")
        if dynamics.get("emotional_labor"):
            cols[1].markdown(f"**Emotional labor**\n\n{dynamics['emotional_labor']}")
        if dynamics.get("trajectory"):
            cols[2].markdown(f"**Trajectory**\n\n{dynamics['trajectory']}")

    dyn = features.get("dynamics", {})
    lor = dyn.get("left_on_read", {})
    dbl = dyn.get("double_texts", {})
    ini = dyn.get("conversation_initiations", {})
    if lor or dbl or ini:
        st.subheader("Behavioural patterns")
        stat_senders = features["senders"][:6]
        bcols = st.columns(len(stat_senders))
        for i, s in enumerate(stat_senders):
            bcols[i].markdown(f"**{s}**")
            bcols[i].markdown(
                f"Left on read: **{lor.get(s, 0)}×**  \n"
                f"Double-texts sent: **{dbl.get(s, 0)}×**  \n"
                f"Started conversations: **{ini.get(s, 0)}×**"
            )

    trajectory = features.get("trajectory", [])
    if len(trajectory) >= 3:
        st.divider()
        st.subheader("Conversation over time")
        traj_senders = features["senders"]
        share_data = {"Period": [b["label"] for b in trajectory]}
        distortion_data = {"Period": [b["label"] for b in trajectory]}
        for s in traj_senders:
            share_data[s] = [b["per_person"].get(s, {}).get("message_share_pct", 0) for b in trajectory]
            distortion_data[s] = [b["per_person"].get(s, {}).get("distortion_signals", 0) for b in trajectory]
        share_df = pd.DataFrame(share_data).melt("Period", var_name="Person", value_name="Share %")
        dist_df = pd.DataFrame(distortion_data).melt("Period", var_name="Person", value_name="Distortion signals")
        col1, col2 = st.columns(2)
        with col1:
            fig = px.line(share_df, x="Period", y="Share %", color="Person",
                          title="Message share over time", markers=True, height=280)
            fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", legend_title="")
            st.plotly_chart(fig, width="stretch")
        with col2:
            fig = px.bar(dist_df, x="Period", y="Distortion signals", color="Person",
                         title="Distortion signals over time", barmode="group", height=280)
            fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", legend_title="")
            st.plotly_chart(fig, width="stretch")

    accommodation = features.get("accommodation")
    if accommodation:
        st.divider()
        st.subheader("Language accommodation")
        score = accommodation["overall_score"]
        trend = accommodation.get("trend", [])
        c1, c2 = st.columns([1, 3])
        c1.metric("Alignment score", f"{score:.0%}",
                  help="How similar your function-word usage is. Higher = more linguistic mirroring.")
        c1.caption(accommodation["interpretation"])
        if len(trend) >= 3:
            trend_df = pd.DataFrame(trend)
            fig = px.line(trend_df, x="label", y="score", markers=True,
                          title="Linguistic alignment over time", height=250,
                          labels={"label": "", "score": "Alignment"})
            fig.update_yaxes(range=[0, 1])
            fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
            c2.plotly_chart(fig, width="stretch")

    group_data = features.get("group", {})
    if group_data:
        cohesion = group_data.get("cohesion", {})
        topic_own = group_data.get("topic_ownership", {})
        subgroups = group_data.get("subgroups", [])
        if cohesion:
            st.divider()
            st.subheader("Group health")
            gc1, gc2, gc3 = st.columns(3)
            gc1.metric("Cohesion score", f"{cohesion['overall_cohesion']:.0%}")
            gc2.metric("Participation balance", f"{cohesion['participation_balance']:.0%}",
                       help="How evenly messages are distributed across members")
            gc3.metric("Reply ratio", f"{cohesion['reply_ratio']:.0%}",
                       help="Fraction of messages that get a quick reply")
            st.caption(cohesion["interpretation"])
        if topic_own:
            st.subheader("Topic ownership")
            st.caption("Who sends messages that get a quick reply vs go unanswered")
            to_df = pd.DataFrame([{"Person": s, **v} for s, v in topic_own.items()])
            fig = px.bar(to_df, x="Person", y="traction_rate_pct",
                         title="% of messages replied to within 5 min",
                         labels={"traction_rate_pct": "Traction rate %"}, height=260)
            fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, width="stretch")
        if subgroups and len(subgroups) > 1:
            st.subheader("Subgroups detected")
            for sg in subgroups:
                label = "Tight cluster" if len(sg) > 1 else "Peripheral"
                st.markdown(f"**{label}**: {', '.join(sg)}")

    st.divider()
    st.subheader("Coaching report")
    st.markdown(result["report"])

    if not is_shared:
        st.divider()
        if st.button("🔗 Share report", type="primary"):
            with st.spinner("Saving…"):
                rid = save_report(result, features)
            if rid:
                share_url = f"{APP_URL}?r={rid}"
                st.success("Report saved — share this link:")
                st.code(share_url, language=None)
            else:
                st.warning("Sharing requires UPSTASH_REDIS_REST_URL and UPSTASH_REDIS_REST_TOKEN to be set.")

    with st.expander("Raw statistics"):
        st.json(features)


# --- App ---

st.set_page_config(
    page_title="Chatalyze",
    page_icon="🪞",
    layout="wide",
)

st.title("Chatalyze")
st.caption("Your words are a mirror. Most people never look.")

# --- Shared report (standalone view, no upload form) ---
_shared_id = st.query_params.get("r")
if _shared_id:
    st.divider()
    with st.spinner("Loading shared report…"):
        _loaded = load_report(_shared_id)
    if _loaded:
        _result, _features = _loaded
        st.caption(
            f"Shared report · {_features['date_range']['days']} days · "
            f"{_features['total_messages']:,} messages · "
            f"{', '.join(_features['senders'])}"
        )
        _render_analysis(_result, _features, is_shared=True)
    else:
        st.error("Report not found or the link has expired.")
    st.stop()

st.divider()

# --- Upload ---
st.subheader("Upload your chat export")
with st.expander("How to export from WhatsApp"):
    st.markdown("""
**Android**: Open chat → ⋮ menu → More → Export chat → Without media
**iPhone**: Open chat → Contact/Group name → Export Chat → Without media

You'll get a `.txt` file. Upload that here.
    """)

uploaded = st.file_uploader("Choose a .txt export file", type=["txt"])

if uploaded:
    raw = uploaded.read().decode("utf-8", errors="replace")
    fmt = detect_format(raw)
    messages, is_group = parse(raw)

    if not messages:
        st.error(f"Couldn't parse this file (detected format: {fmt}). Make sure it's a WhatsApp .txt export without media.")
        st.caption("Try: Open chat → ⋮ → More → Export chat → Without media. If on Android with 24h clock or non-English locale, this should now work.")
        st.stop()

    df = to_dataframe(messages)
    senders = df["sender"].unique().tolist()

    st.success(f"Parsed **{len(messages):,} messages** from **{len(senders)} {'participants' if is_group else 'people'}** over **{(df['timestamp'].max() - df['timestamp'].min()).days} days**")

    if is_group:
        st.info(f"Group chat detected: {', '.join(senders)}")

    st.divider()

    # --- Stats preview ---
    st.subheader("Overview")
    cols = st.columns(min(len(senders), 6))
    for i, sender in enumerate(senders[:6]):
        count = len(df[df["sender"] == sender])
        pct = round(count / len(df) * 100, 1)
        cols[i % len(cols)].metric(sender, f"{count:,} messages", f"{pct}% of conversation")

    import plotly.express as px
    daily = df.groupby([df["timestamp"].dt.date, "sender"]).size().reset_index(name="count")
    daily.columns = ["date", "sender", "count"]
    fig = px.bar(daily, x="date", y="count", color="sender",
                 title="Messages over time", labels={"date": "", "count": "Messages"}, height=300)
    fig.update_layout(legend_title="", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, width="stretch")

    st.divider()

    # --- Run analysis ---
    st.subheader("Analysis")

    groq_key = os.getenv("GROQ_API_KEY") or st.text_input(
        "Groq API key", type="password", help="Get a free key at console.groq.com"
    )

    if "analyzing" not in st.session_state:
        st.session_state.analyzing = False

    if st.button("Run analysis", type="primary",
                 disabled=not groq_key or st.session_state.analyzing):
        st.session_state.analyzing = True
        if groq_key:
            os.environ["GROQ_API_KEY"] = groq_key

        with st.spinner("Extracting features..."):
            features = extract_all(df, is_group)
            sampled = sample(df, features)

        with st.spinner(f"Analyzing {len(sampled)} representative messages with Groq..."):
            try:
                result = analyze(features, sampled)
            except Exception as e:
                st.session_state.analyzing = False
                st.error(f"Analysis failed: {e}")
                st.stop()

        st.session_state.analyzing = False
        st.session_state.result = result
        st.session_state.features = features

    if "result" in st.session_state and "features" in st.session_state:
        _render_analysis(st.session_state.result, st.session_state.features)
