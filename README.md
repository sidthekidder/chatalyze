# Chatalyze

**Your words are a mirror. Most people never look.**

Chatalyze analyzes a WhatsApp chat export and gives you a psychologically-grounded picture of how you communicate — cognitive patterns, relationship dynamics, emotional labor, reply asymmetry — with a coaching report written by an LLM that has actually read your conversation.

**[→ Try it live](https://chatalyzer.streamlit.app)**

Not a word cloud. Not a stats dashboard. A report that tells you something specific and true about yourself.

---

## What it analyzes

**For 1:1 chats**
- Cognitive distortion profile — catastrophizing, black-and-white thinking, personalization — with real quotes pulled from your messages
- Language accommodation score — are you growing more linguistically similar over time, or diverging? (predicts relationship trajectory better than sentiment)
- Emotional labor balance — who comforts, who vents, who initiates repair after conflict
- Conversation dynamics — who initiates, double-texts, leaves on read, and sets the pace
- Reply time distributions and directional asymmetry — who responds faster to whom
- Reply time trend — are response times rising (disengagement signal) or falling across the arc of the conversation?
- Pronoun fingerprint — I vs we vs you balance, and what it signals about power and investment
- Emoji register — affective warmth (❤️😊) vs distress (😭😡) vs hedging/softening (😅😂🙏)
- Activity patterns — when each person is most communicative, sleep window estimates
- Emotional arc — does distress language in the conversation actually decrease over time?

**For group chats**
- Participant role map — connectors (high reply-diversity), broadcasters (sends to many, replies to few), lurkers, energizers
- Who routes attention and information through themselves
- Topic ownership — whose topics gain traction vs whose get ignored
- Subgroup and clique detection from reply patterns
- Group cohesion score and participation balance
- Temporal role shift — who was active early but went quiet? Who emerged over time?

**The coaching report**
The LLM receives a statistical feature bundle plus strategically sampled message chunks (not all messages — a representative arc). It addresses each person by name, references real moments, and ends with specific things to try. No generic advice, no therapy-speak.

---

## How to use it

**Step 1: Export your WhatsApp chat**

- **Android**: Open chat → ⋮ → More → Export chat → Without media
- **iPhone**: Open chat → contact/group name → Export Chat → Without media

You'll get a `.txt` file.

**Step 2: Visit the app**

Go to [chatalyzer.streamlit.app](https://chatalyzer.streamlit.app), upload the `.txt` file, and click **Run analysis**.

Or click **"See a sample report"** first to see what the output looks like before uploading your own chat.

**Step 3: Read your report**

The report has four sections: observed patterns (with evidence), relationship dynamics, distortion profile per person, and a coaching section with specific things to try.

---

## Run locally

```bash
git clone https://github.com/sidthekidder/chatalyze
cd chatalyze
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # add your GROQ_API_KEY
streamlit run app.py
```

Get a free Groq API key at [console.groq.com](https://console.groq.com) — no credit card needed (14,400 requests/day free tier).

Optional: set `UPSTASH_REDIS_REST_URL` and `UPSTASH_REDIS_REST_TOKEN` to enable shareable report links.

---

## How it works

```
WhatsApp .txt export
        ↓
Parser  —  handles Android/iOS formats, 24h/12h time, RTL unicode markers,
           system messages, date separators, multilingual exports
        ↓
Feature extraction  —  all computation in Python, no LLM:
  behavioral:    reply times, double-texts, initiation, left-on-read,
                 directional RT asymmetry, daytime-normalized RT trend
  linguistic:    pronoun ratios, distortion signal words (density per 100 words),
                 function-word vectors for Language Style Matching (LSM),
                 LSM asymmetry (who accommodates whom), emoji bucket classification
  temporal:      activity patterns per person, emotional arc (neg-word density
                 across conversation thirds), seeker distress trajectory
  group:         reply network, reply diversity, role classification,
                 topic ownership, cohesion score, temporal role shift
        ↓
Sampler  —  ~100 messages via temporal bucketing + longest messages +
            signal keywords + anchors. LLM sees the full arc, not just recent.
        ↓
Single Groq call  —  stats bundle + sampled messages → structured analysis
                     (patterns, dynamics, distortion profile) + coaching narrative
        ↓
Report  —  rendered in Streamlit with charts, shareable link, share card
```

---

## Research basis

The feature set is grounded in three decades of psycholinguistics and relationship research:

- **LIWC / Pennebaker (1990s–2020s)** — function word use (pronouns, articles, prepositions) predicts psychological state and relationship health. High I/me/my correlates with depression and lower relationship investment; we/us signals shared identity. Function word matching between two people predicts relationship stability better than sentiment does.
- **Language Style Matching (Ireland & Pennebaker 2010; Groom et al.)** — unconscious mirroring of each other's "the", "but", "is" reveals who adapts to whom (power) and who's invested.
- **Cognitive distortion detection (Shickel et al., C2D2 dataset; Rashkin et al.)** — catastrophizing, black-and-white thinking, and personalization are detectable from text across informal registers with reasonable reliability.
- **Gottman Institute** — contemptuous language is the single strongest predictor of relationship dissolution; confrontational pronoun use (you/your framing) precedes conflict escalation.
- **ESConv (Liu et al. 2021)** — 8-strategy emotional support taxonomy; supporter future-focus and explorative questioning are validated signals of effective comforting behavior.
- **Emoji pragmatics (Herring et al.)** — 😂 functions primarily as a hedging/softening marker rather than a happiness signal in digital conversation; emoji register differences reveal personality and relational stance.
- **Reply latency (Martin 2025)** — ~70% of WhatsApp messages get a response within 5 minutes; rising RT over conversation thirds signals disengagement.

---

## Privacy

Your chat file is processed in memory and never stored. The Groq API receives only a statistical feature bundle and ~100 sampled messages — not your full conversation. If you want zero external calls, the architecture is designed to accept a local Ollama model as a drop-in replacement.

When you share a report link, the computed features and analysis JSON are stored in Upstash Redis with a 90-day TTL. Raw message content is never stored.

---

## Contributing

Issues and PRs welcome. If you have a WhatsApp export format that fails to parse (different locale, time format, or OS version), open an issue and paste the first 3–4 lines with names/content redacted.

---

*Built by [@sidthekidder](https://github.com/sidthekidder)*
