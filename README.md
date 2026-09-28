# Chatalyze

**Your words are a mirror. Most people never look.**

Chatalyze analyzes WhatsApp chat exports and gives you a psychologically-grounded picture of how you communicate — cognitive patterns, relationship dynamics, emotional labor, and more — with a personalized coaching report written by an LLM that has read your actual conversation.

Not a stats dashboard. Not a word cloud. Something that tells you something true about yourself.

![Chatalyze screenshot](https://raw.githubusercontent.com/sidthekidder/chatalyze/main/docs/screenshot.png)

---

## What it analyzes

**For 1:1 chats**
- Cognitive distortion profile — catastrophizing, mind-reading, black-and-white thinking, with real quotes from your messages
- Reply time distributions, double-text patterns, who initiates conversations
- Emotional labor balance — who comforts, who vents, who apologizes
- Pronoun fingerprint — I vs we vs you, and what the balance signals
- Sentiment trajectory — is this relationship warming or cooling over time

**For group chats**
- Participant role map — connectors, broadcasters, lurkers, energizers
- Who drives topics, whose messages get ignored
- Subgroup and clique detection from reply patterns
- Group cohesion and energy flow

**Coaching report**
The LLM receives extracted features alongside strategically sampled message chunks — not just averages, but real moments. The report addresses each person by name, references specific things they said, and ends with concrete things to try.

---

## How to use it

**1. Export your WhatsApp chat**

- **Android**: Open chat → ⋮ → More → Export chat → Without media
- **iPhone**: Open chat → contact name → Export Chat → Without media

You'll get a `.txt` file.

**2. Get a free Groq API key**

Go to [console.groq.com](https://console.groq.com) → API Keys → Create API Key. No credit card needed for the free tier (14,400 requests/day).

**3. Run locally**

```bash
git clone https://github.com/sidthekidder/chatalyze
cd chatalyze
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
echo "GROQ_API_KEY=your_key_here" > .env
streamlit run app.py
```

Upload your `.txt` file, click **Run analysis**, and get your report. You can also download it as a self-contained HTML file to save or share.

---

## How it works

```
WhatsApp .txt export
        ↓
Parser  —  handles Android/iOS formats, 24h/12h time, date separators,
           RTL/LTR unicode markers, multilingual exports
        ↓
Feature extraction  —  all computation in Python, no LLM:
  behavioral: reply times, double-texts, initiation, left-on-read
  linguistic:  pronoun ratios, distortion signal words, emoji analysis
  group:       reply network, participant roles, topic ownership
        ↓
Sampler  —  picks ~100 messages via temporal bucketing (5 equal time
            windows) + longest messages + signal keywords + anchors.
            Ensures the LLM sees the full arc, not just recent events.
        ↓
Single Groq call  —  stats bundle + sampled messages → structured
                     analysis (JSON) + coaching narrative in one pass
        ↓
Report  —  patterns, dynamics, coaching report, HTML download
```

The LLM never receives all messages — only a strategic sample alongside computed statistics. This keeps cost low and quality high: the model reasons about patterns, not raw volume.

---

## Research basis

The feature set is grounded in psycholinguistics and relationship research:

- **LIWC (Pennebaker)** — pronoun use patterns predict psychological state and relationship quality. Function word matching between two people predicts relationship stability better than sentiment.
- **Cognitive distortion detection** — NLP research (C2D2 dataset) shows catastrophizing, mind-reading, and black-and-white thinking are detectable from text across informal registers.
- **Gottman Institute** — contemptuous language is the single strongest predictor of relationship dissolution.
- **Communication Accommodation Theory** — linguistic convergence/divergence over time signals relationship trajectory.

---

---

## Roadmap

- [ ] iMessage export support
- [ ] Longitudinal tracking — compare analyses over time
- [ ] Group chat UX improvements

---

## Contributing

Issues and PRs welcome. If you have a WhatsApp export format that fails to parse, open an issue and paste the first 3-4 lines (redact names/content).

---

## Privacy

Your chat file is processed in memory on your machine and never sent anywhere. The Groq API receives only a statistical feature bundle and ~100 sampled messages — not your full conversation history. If you want zero external calls, the architecture is designed to support local Ollama models as a drop-in replacement.

---

*Built by [@sidthekidder](https://github.com/sidthekidder)*
