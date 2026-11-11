---
name: what-happened
description: Tight recap of where this session stands, to scan and resume from.
disable-model-invocation: true
---

# What happened

Drop the user back into the session in under 30 seconds of reading. They forgot — after a
break, a tangent, or a long subagent run — and their job is to scan and resume, not to
re-read a transcript.

Produce one short recap. It opens on the task and ends on the next action; nothing else
belongs in it. Write it in the voice of the `wait-what` skill: read
`.claude/skills/wait-what/SKILL.md` and follow it — ASD-STE100 Simplified Technical English,
using the ubiquitous language from `docs/` (the map in `docs/index.md`).

## Output

Use these sections, in this order. Skip one only when there is genuinely nothing to put in
it. Under 150 words, hard cap 250.

```
**Now**: <1 line — the active task in concrete terms>
**Why**: <1 line — the trigger or motivation the user gave>
**Goal**: <1 line — the success condition>
**Done**: <≤4 bullets, most recent first>
**Next**: <1–2 lines — the very next action or open decision>
**Open** (only when blocking): <bullets>
```

## Anchors

The recap works on anchors — the concrete handles a brain latches onto faster than prose.
Every line carries one, or carries a state:

- **Names**: file paths, function names, the worktree/branch, the plan doc in `docs/plans/`,
  a migration name. Keep identifiers in their original form; never translate them.
- **States**: gate green or not (`typecheck` / `lint` / `qa`), committed / merged / pushed,
  what a background agent is still doing.
- **Decisions the user confirmed**, especially scope changes ("dropped quota, kept only the
  AI judge"). State them as settled — re-justifying a closed decision reopens it.
- **The most recent meaningful action**: what the last tool call or agent run actually
  produced, not which agent ran or how many files it read.

Point at a file path where code or SQL would go, and skip anything the user said this turn —
echoing their own message back costs them the 30 seconds you're saving.

When the session is long enough that the early turns are gone, `.remember/now.md` and the
recent `today-*.md` hold what the transcript no longer does.

## Edge cases

- **Barely started (under ~3 turns)** — say so directly: "we just started — you pointed at X,
  nothing built yet."
- **Topic changed mid-session** — recap the *current* topic only, and give the earlier one a
  single line if it is still open: "(paused the Y work earlier.)"
- **Just finished a big chunk** — lead **Done** with what shipped, and make **Next** the
  verification or the follow-up commit.
- **Blocked on a pending question** — put the blocker under **Open** and make **Next** be
  answering it.

## Before sending

- Could the user paste this into a fresh conversation and pick the thread back up? If not,
  the missing anchor is what to add.
- Over 250 words? Cut **Done** bullets first, then trim **Why**.

When what they need is a decision rather than a state, that is `/what-next` — this skill
reports where things stand and stops there.
