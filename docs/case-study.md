# Case Study: Agentic RAG over SEC Filings

*Answers natural-language questions about public-company annual reports and cites every
claim to the passage it came from. It plans its own retrieval, bounds the work any single
query can trigger, and traces every step.*

🔗 **[Live demo](https://huggingface.co/spaces/gaurav-goyal/tenk-insight)** ·
[Technical deep-dive](./design-decisions.md) · [Source & README](../README.md)

---

**At a glance**

- 13 annual reports from 8 companies, roughly 3,000 pages
- Breaks a question into separate searches, then searches again when the retrieved passages
  don't cover the question or the drafted answer isn't backed by them
- Every claim shows the passage it came from, with a link to the filing on SEC EDGAR
- Questions it can't answer from the loaded filings are declined rather than guessed at
- The number of model calls per query is bounded in configuration, not left to the
  model's discretion, so cost has a known ceiling
- Roughly $2 to $3 per thousand questions, measured from traced token usage
- Answer quality is scored against a fixed question set, so the effect of a change can be
  measured instead of assumed

---

## The problem

A 10-K annual report runs to several hundred pages of dense legal and financial writing.
The answer to a specific question is usually in there somewhere, but finding it means
knowing which section to open, and reading carefully enough to catch where a number gets
qualified further down the page.

General-purpose chatbots answer questions like this readily. The trouble is that they
invent figures, and offer no way to check one against the source. A fabricated number reads
exactly like a real one.

None of this is specific to SEC filings. The same thing happens with contracts, internal
policy, regulatory documents, support manuals, anywhere a wrong answer costs something.

Wiring up document search over a set of files is no longer the hard part. The hard part
starts after the demo works. Answer quality has to be measurable, so a regression shows up
as a number rather than a complaint. A wrong answer has to be traceable to the step that
produced it. Cost and latency per query have to be known before the system goes in front
of real users.

---

## The solution

The system answers only from filings it has already indexed, and that constraint is
enforced by the structure of the pipeline rather than by instruction alone.

![How it works: filings are indexed into a searchable library. Each question is planned, searched, and checked, looping back when the passages found aren't enough, then answered with citations. Every run is traced, and quality is scored against a fixed reference set after each change](./assets/how-it-works.svg)

**It plans the search before running it.** A question never goes straight to the index.
The system first works out which company and fiscal year it refers to, then splits a
multi-part question into one search per concept. A question outside what the filings cover is
caught here, before any retrieval happens, and comes back as a refusal rather than an
answer assembled from the model's memory.

**The model plans, the pipeline controls.** The model makes the judgement calls. It decides
what to look for, whether the passages it got back are enough to answer, and whether the
answer it drafted holds up against them. What it does not decide is how many times it may
ask. Every loop has a limit set in configuration, so a hard question costs more than an
easy one but never more than a fixed ceiling. Without that limit, a model that keeps
deciding it needs more context has no reason to stop, and the cost climbs with every pass.

**Every figure traces back to a passage you can open.** Answers are written only from
retrieved text, and as it writes, the model has to match each figure to the numbered passage
that states it. A separate pass then re-reads the finished answer, checking that it
addresses the question and that every figure it reports is genuinely stated by the passage
cited for it. When something fails that check, the system searches again and rewrites the
answer rather than showing the first draft. Every citation expands in place to show its
passage, with the filing and section it came from and a link to the document on SEC EDGAR,
so checking a number takes one click.

**It reports figures, it doesn't derive them.** Language models are unreliable at
arithmetic, and a miscalculated margin looks exactly like a correct one. So the system
never computes. If a filing states revenue and cost but never prints gross margin, the
answer returns both components and says the margin itself isn't stated. If derived metrics
became a requirement here, the answer would be a calculator tool that computes
deterministically from figures the model has quoted, so the arithmetic still never depends
on the model.

**Comparisons across companies and years.** A question spanning two companies becomes two
parallel searches, and the answer sets the figures side by side with a citation for each.

---

## How I know it works

The question worth asking of any RAG system is how you'd know if it got worse. Without a
fixed set of questions and scores attached to them, quality is whatever the last demo
happened to feel like. Four things get measured here on every run.

**Groundedness.** Is every claim in the answer supported by the passages it cites? This is
the one that catches invention.

**Retrieval coverage.** Was the evidence found at all? Keeping this separate from
groundedness matters, because an answer can fail either because the model invented
something or because the right passage was never retrieved, and those have nothing to do
with each other. Measuring them together hides which one you have.

**Answer correctness.** Does the answer match the reference answer for that question?

**Refusal accuracy.** Does it decline what it can't answer, without declining things it
can? Both directions count. A system that refuses everything scores perfectly on the first
half.

Retrieval precision is tracked alongside them, measuring whether the relevant passages came
back at the top of the list rather than buried under less useful ones. That's what explains
a drop when coverage looks fine but answers get worse.

Across the question set, the evidence needed was retrieved about 95% of the time, roughly
nine in ten claims in an answer were supported by the passages cited, and all but one of
the questions outside what the filings cover were correctly declined. The per-metric table
is in the [README](../README.md#results).

The measurements are also what found the largest problem. Comparisons scored well below
single-company questions on retrieval, and the traces showed why. The planner was pinning
each search to a specific filing section, but companies don't agree on which section holds
a given figure, so a confident wrong guess excluded the answer entirely. The fix was to
widen the search whenever a section-filtered pass comes back empty or weak. Coverage on
comparisons went from roughly 72% to 94%.

The scores come with a caveat. The model grading each run is the same one that produced the
answer, so the two share blind spots. Rather than take a low score at face value, I read
the trace behind it. One was a correct and properly sourced answer the grader had marked
down because it read $4.9 billion and $4,899 million as different figures. An independent
grader leads the [what's next list](./design-decisions.md#whats-next), alongside
deterministic metrics for the parts of scoring that don't need a model at all.

---

## Built to operate

A demo has to work once. Something you run has to have a cost you can predict, a latency
you can name, and a failure you can trace.

**Cost is bounded before the query runs.** A typical answer costs roughly $2 to $3 per
thousand questions at current pricing, measured from token counts captured on every call
rather than estimated. It also doesn't grow with the corpus,
because the model only ever sees the handful of passages retrieved for that question.
Thirteen filings or thirteen thousand, the input to the answer step is the same size.
Building the index makes no model API calls at all, so adding documents costs machine time
rather than tokens.

**The latency is a deliberate tradeoff.** Answers take 15 to 40 seconds. Most of that time
is three model calls running in sequence, and what makes them slow is the model writing out
its reasoning before it answers, which is what keeps a small, cheap model accurate on
financial figures. Latency can come down by trimming how much the model writes, caching
repeated questions, swapping in a faster model, or cutting a retrieval pass. Which of those
are worth pulling is a question for the eval rather than a guess. Questions outside what the
filings cover come back in about 3 seconds, because they stop at the first step.

**Every run is traced end to end.** Each step emits a record of the query it ran, the
passages it retrieved, the tokens it used and the time it took. When an answer is wrong,
the trace shows which step produced it, which is how both problems in the section above
were found. The app shows token usage and cost alongside each answer, and any answer can be
rated, so real use can surface problems a fixed question set never covered.

**Behaviour lives in configuration.** The embedding model, chunk sizes, how many passages
come back, which agentic steps run, and which provider and model to call are all set in one
file. Moving from a cloud model to one running locally, or from the small model to a larger
one, is a configuration change rather than a code change.

---

## What carries over

The filings are the part that's specific. A 10-K has named sections and financial tables
laid out in XBRL, and reading those correctly is its own work. What generalises isn't the
pipeline itself but the properties it has to have: every claim tied to a passage someone
can open, anything outside the corpus declined rather than guessed at, loops bounded so
cost has a ceiling, every run traced, and quality scored against a fixed question set.

Contracts, internal policy, regulatory documents and support material all pose the problem
this system was built for. The documents are long, the answer is somewhere inside them, and
a confident wrong answer costs more than no answer at all.

What sits under those properties changes with the documents. How they're split, which
retrieval signal carries the weight, what metadata is worth filtering on, which model earns
its cost, and what the answer prompt has to guard against. Priorities shift as well. A
support corpus gets asked the same things over and over, which makes caching the first
thing to build rather than an optimisation. Contract review tends to want higher recall and
can afford to wait for it. A compliance system usually wants refusal tuned harder than
coverage, because a missing answer is recoverable and a wrong one often isn't.

The piece I'd build first in any of them is the question set. Without it, every decision
after it is a guess about whether it helped.

---

**Gaurav Goyal**, independent AI consultant. I build production-grade RAG systems that stay
grounded, traced, and measurable.

Available to build a new system, take a prototype to production, or improve one already
running.

▶️ **[Try the demo](https://huggingface.co/spaces/gaurav-goyal/tenk-insight)** ·
[ggoyal2211@gmail.com](mailto:ggoyal2211@gmail.com)
