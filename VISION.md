# Chatalyze

> *Your words are a mirror. Most people never look.*

---

## The Problem

Every day billions of people communicate through WhatsApp, iMessage, Telegram. These conversations contain the most unfiltered version of who someone is — how they argue, how they comfort, how they deflect, how they love. And yet nobody reflects on them.

A therapist charges $200/hr to tell you that you catastrophize under stress, that you carry emotional labor others don't reciprocate, that you use "you never" and "you always" when you're hurt. Most people never get that feedback — not because it isn't valuable, but because it's inaccessible.

The evidence has been sitting in your phone the whole time.

---

## What Chatalyze Does

You export a chat — 1:1 or a group. Chatalyze runs it through a pipeline that extracts psychological and relational signals from your language patterns, then uses an LLM to generate a coaching report grounded in what you actually said.

Not a sentiment bar chart. Not a word cloud. A report that tells you something true and specific about how you communicate — and what to do about it.

The LLM sees chunks of your actual conversation alongside the extracted features. This matters: a coaching note that references a real moment ("when the conversation shifted here, you moved into...") is an order of magnitude more useful than generic advice derived from averages.

---

## The Research Foundation

This isn't vibes-based. Thirty years of psycholinguistics (Pennebaker, LIWC) established that *how* people use language — not what they say, but the function words, pronouns, and structures they default to — is deeply predictive of psychological state and relationship health:

- High I/me/my usage correlates with depression, self-focus, lower relationship status
- We/us usage correlates with shared identity and relationship investment
- **Function word matching** between two people — unconsciously mirroring each other's "the", "but", "is" — predicts relationship quality better than sentiment does
- Contemptuous language (Gottman Institute) is the single strongest predictor of relationship dissolution

NLP research has produced validated approaches for detecting **cognitive distortions** — the irrational thought patterns CBT therapy targets. These are detectable from text:
- Catastrophizing ("this always ruins everything")
- Mind reading ("you obviously don't care")
- Black-and-white thinking ("you're either with me or you're not")
- Personalization (everything is your fault or theirs)
- Emotional reasoning ("I feel like you hate me therefore you do")

For group chats, network analysis research identifies distinct participant roles — connectors, broadcasters, lurkers, de-escalators — from reply patterns and message structure. Who routes information through themselves. Who brings energy. Who drains it.

---

## Language Is Messy and Global — Good

Most NLP tools assume English. Most assume formal or semi-formal writing. Most fail the moment someone writes "haha ngl tho 😭" or switches between Hindi and English mid-sentence.

Chatalyze should not.

The people with the most to gain from this tool are not 40-year-old Americans in therapy already. They're 22-year-old users in Mumbai, São Paulo, Lagos, Jakarta — code-switching between languages, using emoji as punctuation, communicating in a register that existing tools misread or ignore entirely.

The research on code-switching in WhatsApp specifically (there's a growing body of it) shows that hybrid language practices are not noise — they're expressive and meaningful. Gen Z informal language has its own grammar and emotional logic. Emoji use varies by culture but carries genuine signal: who uses softening emoji ("😅", "😭") vs assertive ones vs none at all reveals communication style as clearly as word choice does.

The technical approach needs to accommodate this: multilingual embeddings (XLM-RoBERTa), emoji-aware tokenization, LLM prompting that doesn't penalize informal language, distortion detection that works across registers. Getting this right is hard and most competitors won't bother. That's the opportunity.

---

## What It Analyzes

**For 1:1 chats:**
- Cognitive distortion profile per person — frequency, type, real examples pulled from the chat
- Language accommodation score: are you growing more similar to each other over time, or diverging?
- Emotional labor balance: who comforts, who vents, who apologizes, who initiates repair after conflict
- Conversation dynamics: who initiates, who ends, who double-texts, reply time distributions for both sides
- Left-on-read patterns: how often, in what contexts, after what types of messages
- Power signals: who steers topics, who chases, who sets the pace of the relationship
- Sentiment trajectory: is this relationship warming or cooling?
- Conflict patterns: how arguments start, escalate, and close (or stay open)
- Pronoun fingerprint: I vs we vs you — what the balance signals
- Sleep and activity patterns from message timing — when each person is most communicative

**For group chats:**
- Participant role map: connectors (high reply-to ratio), broadcasters (send to many, receive from few), lurkers, energizers, de-escalators
- Who routes information and attention through themselves
- Subgroup detection: do cliques exist within the group?
- Topic ownership: who introduces topics that gain traction vs whose messages get ignored
- Sentiment by member over time: who shifts the group's emotional tone
- Group cohesion score: is this group functioning or fragmenting?

**The coaching layer:**
The LLM receives the extracted features plus representative chunks of actual conversation. It writes a narrative that is specific, direct, and actionable — not a horoscope. If someone catastrophizes 40 times in a conversation, the report says so, shows them an example from their own words, explains the pattern, and gives them something concrete to try.

---

## Where the Value Is

The deepest value isn't a one-time analysis. It's longitudinal.

When someone can track whether their catastrophizing frequency is going down month over month — or whether a relationship's sentiment trajectory has been declining for six months before they consciously felt it — Chatalyze becomes something they return to. The data compounds. The insights sharpen.

That's the product worth building.

---

## Business Model

Free tier does a single analysis with core insights. Enough to be genuinely useful and shareable.

Pro is a subscription — unlimited analyses, longitudinal tracking, cross-relationship comparison, deeper distortion profiling. The price point should feel like a no-brainer relative to what a single therapy session costs.

No enterprise tier. Not yet. Build something people love first.

---

## Market Landscape

### Phase 1 — Personal Chat

The existing tools fall into two buckets: fun stats, and shallow AI wrappers.

**Unwrapped** — "Spotify Wrapped for WhatsApp." Shows biggest talkers, favorite phrases, memorable moments. Shareable, social, zero psychological depth. This is entertainment, not insight.

**ChatBump AI** — markets itself as "AI chat analysis and conversation insights." Active on TikTok/Instagram, targeting younger users. From what's visible publicly the analysis is surface-level; no indication of any research-backed feature set.

**Texts From My Ex** — relationship insights from texts. One-trick product, no depth.

**TunedTogether** — relationship compatibility from linguistics. Narrow, couples-specific.

**BrutalVerdict** — the closest thing to a direct competitor. Analyzes WhatsApp and Instagram DMs, supports Hinglish, does reply times, double-text ratios, read receipt patterns, mood mapping. Local in-browser processing. Free core, premium for chat comparison. Positioned around relationship effort — "is this person putting in as much as you?" — which limits the appeal to a specific moment of doubt rather than ongoing self-awareness. No coaching, no cognitive framing, no longitudinal tracking, no group chats.

**Crystal Knows** (original product, now pivoted) — Crystal's original thesis was predicting someone else's personality from their LinkedIn profile so you could tailor your communication to them. Compelling idea, primarily used by B2B sales. They've since pivoted to a personal coaching platform at $49/mo using DISC/Big Five/Enneagram frameworks. The pivot away from the original use case is interesting signal — the business model around "analyze someone else" is harder than it looks.

**The gap**: nobody is doing research-backed, longitudinal, psychologically serious analysis of your own communication patterns. The entire category is either a stats dashboard or a novelty. There is no product that tells you something true and uncomfortable about yourself, grounded in 30 years of psycholinguistics, and then helps you change.

### Phase 2 — Professional Communication

**Crystal Knows** — as above, used heavily in sales for profiling buyers before outreach. Analyzes *other people*, not yourself. Not a self-reflection tool.

**Humantic AI** — enterprise sales intelligence. DISC personality profiles of prospects from public data (LinkedIn, emails). Fully B2B, fully about the other person, no self-awareness angle.

**Textio** — AI for HR writing: job descriptions, performance reviews, feedback. Bias detection in formal documents. Not personal communication analysis.

**Grammarly** — real-time tone suggestions in writing. No historical pattern analysis, no psychological framing, no insight into recurring behaviors. It tells you a sentence sounds "assertive" or "direct" in the moment; it doesn't tell you that you've been passive-aggressive in 40% of your emails to your manager for six months.

**The gap**: every professional tool either analyzes *other people* (Crystal, Humantic) or fixes *individual documents in real time* (Grammarly, Textio). Nobody is helping individuals understand their own professional communication patterns over time — the hedging, the over-explanation, the credit deflection, the silence in group channels — and connecting those patterns to career outcomes. That product doesn't exist.

---

## Phase Two: Professional Communication

Personal chat is phase one. The natural extension is professional communication — Slack, email, Teams — and it's a meaningfully different product with a potentially larger ceiling.

The signals that matter in professional contexts are different:

**Career self-awareness**
- Hedging language: "sorry to bother you", "I might be wrong but", "does this make sense?" — passive markers that actively hurt career progression and are almost invisible to the person using them
- Over-explanation: people who feel insecure overwrite. Message length relative to context is measurable.
- Credit patterns: "I built X" vs "we built X" vs "the team built X" — who takes ownership of their work in writing
- Assertiveness asymmetry: how you write to your manager vs your reports vs peers reveals your internal status model

**Visibility and influence**
- Slack channel participation vs lurking: who contributes to public channels vs only operates in DMs
- Idea traction: whose suggestions get picked up and built on, and whose get ignored
- Response time patterns by relationship: responding to a manager in 2 minutes and a report in 2 days is a pattern worth seeing

**The value proposition is sharper here**: personal chat is "understand yourself and your relationships." Professional chat is "communicate like someone who gets promoted." That's a more concrete, outcome-oriented hook — and one people will pay for without needing much convincing.

The privacy model stays B2C: individuals export their own Slack message history (Slack supports this) or email archive. No workspace admin access, no company data handled. The person analyzing is always the person who owns the data.

This is phase two. Build the personal product first, prove the analysis is genuinely useful, then extend the engine to professional contexts with a new feature set and positioning on top.

---

## What This Is Not

Not a replacement for therapy. Not a tool for surveilling someone else's messages without consent. Not a judgment about whether a relationship is good or bad.

A mirror. What you do with what you see is yours.

---

## Open Questions

- Which cognitive distortion model works best across informal, multilingual text — fine-tuned classifier vs few-shot LLM?
- How do we handle consent in group chats where other people's messages are analyzed?
- What's the right directness level in coaching? Too soft is useless. Too harsh and people close the tab.
- How do we handle languages we haven't explicitly tested? Fail gracefully or attempt anyway?
- Should longitudinal tracking require account creation, or can it work locally?
- What does a genuinely useful group chat report look like vs a 1:1 report — different enough to need a separate UX?

---

*Started: September 2026*
*Status: Vision / Pre-build*
