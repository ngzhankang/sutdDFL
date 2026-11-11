---
name: grill-me
description: Interview me about a loose idea, in rounds, until it is clear enough to commit to.
disable-model-invocation: true
---

# Grill me

The user has an idea that is not yet a plan: a feature, an experiment, a refactor, a research direction. Your job is to interview them until the idea is sharp enough to commit to. The output is a clearer idea in the user's head and in this conversation. Write no files, make no plan, and change no code during the grill.

## How a round works

1. Map the open branches of the idea: goal, scope, constraints, success criteria, risks, and the decisions they depend on.
2. Ask only the questions whose prerequisites are already answered. A question that depends on an unsettled decision waits for a later round. Group a round's questions in a short numbered list so the user can answer by number.
3. For each question, say briefly why it matters, and offer your recommended answer or the options with their trade-offs when you have a view. The user decides; you sharpen.
4. After the answers, restate what is now settled in one or two lines, then open the next round.

## Working with the user

- The user owns the scope. When an answer drifts from the original idea, name the drift and ask whether to follow it or park it.
- "I don't know" is a valid answer. Record it as an open question and route around it, or suggest the smallest experiment that would answer it.
- Push back when an answer conflicts with an earlier one or with what the code in this repo shows. Read the code to check a factual claim instead of asking the user something the repo can answer.
- Some questions cannot be settled by talking, such as how something should feel or whether a model will converge. Call these out and suggest a quick prototype or run, then resume the grill with the result.

## When it ends

Count rounds, not questions. The grill ends when no unresolved branch remains, or when the user stops it. Close with a short summary: the idea as now decided, the decisions made, and any open questions left. If the grill keeps growing past several rounds, suggest splitting the idea into smaller pieces and grilling one at a time.
