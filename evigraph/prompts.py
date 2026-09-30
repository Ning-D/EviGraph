"""Fixed prompts of the graph-building VLM, verbatim as used to build the paper's graphs.

Action names map to the paper as EXPLORE_SLOT = Explore-slot, FOLLOW_EDGE =
Temporal-expand and ANSWER = Stop; BACKTRACK marks a node's vicinity as unproductive
(rarely chosen: 4 times in the 4,880 paper graphs). Sub-needs are called
"sub-questions" in the prompts.
"""

# Query-conditioned clip scoring (one mid-frame per 20 s clip, 10 clips per call).
SCORE_CLIPS = """\
You are scoring short video clips for how useful they are for answering a \
multiple-choice question about a long video.
For each clip, output a relevance score 1-5:
  1 = completely unrelated to the question
  2 = background / weak contextual link
  3 = somewhat relevant, minor evidence
  4 = relevant, clear evidence or context
  5 = highly relevant, key evidence for answering the question
Also give a brief observation (<=80 chars) of what the clip shows.
Output ONLY valid JSON:
{"clips": [{"clip_idx": 0, "score": 1, "observation": "..."}, ...]}"""

# Sub-need decomposition (2-4 sub-questions) of the question text, which includes the options.
DECOMPOSE = """\
Decompose this multiple-choice video question into 2-4 atomic sub-questions \
(information needs) that must be verified IN THE VIDEO to answer it. Focus on \
WHAT to check -- key events, entities, and temporal relations -- do NOT guess \
which option is correct and do not just restate the options. \
Return ONLY JSON: {"sub_questions":["...","..."]}"""

# Gap-driven policy: each step targets the least-covered sub-question or the least-
# discriminated option.
POLICY = """\
You collect video evidence for a multiple-choice question about a long video, with \
TWO goals at once: (a) COVER every sub-question with evidence, and (b) DISCRIMINATE \
the options -- gather evidence that backs exactly ONE option and rules others out. \
Each step, target the BIGGEST current gap: a sub-question with the fewest nodes, OR \
an option that is still undecided (little for/against evidence). Use FOLLOW_EDGE to \
chase what happens just before/after a key moment when that is what tells options apart.
Actions: EXPLORE_SLOT (go to a timestamp), FOLLOW_EDGE (near a node along a relation),
BACKTRACK (mark vicinity dead), ANSWER (all sub-questions covered AND options discriminated).
Return ONLY JSON:
{"reasoning":"which gap is biggest (a sub-question or an option) and why (1-2 sentences)",
 "target":"subq|option","sub_q":<int if target=subq>,"option":<int if target=option>,
 "action":"FOLLOW_EDGE|EXPLORE_SLOT|BACKTRACK|ANSWER",
 "node_id":"<id for FOLLOW_EDGE/BACKTRACK>","direction":"before|after",
 "timestamp":<float seconds for EXPLORE_SLOT>,"missing_info":"what evidence is needed"}
Rules: if budget_left==1 always ANSWER; only ANSWER if every sub-question is covered AND \
the options are discriminated; prefer FOLLOW_EDGE to resolve a contested option."""

# Fallback policy, used only if the decomposition returned no sub-questions.
POLICY_NO_SUBNEEDS = """\
You are a video-evidence collection policy agent answering a multiple-choice \
question about a long video. Given the current evidence graph, choose the single \
best next action to fill gaps in the evidence chain.
Actions:
  FOLLOW_EDGE  — explore near a node, following one of its typed relations.
  EXPLORE_SLOT — go to a timestamp to find a specific missing piece of evidence.
  BACKTRACK    — mark a node's vicinity as unproductive.
  ANSWER       — the chain is sufficient, stop.
Return ONLY JSON:
{"reasoning":"what is missing and why (1-2 sentences)",
 "action":"FOLLOW_EDGE|EXPLORE_SLOT|BACKTRACK|ANSWER",
 "node_id":"<id for FOLLOW_EDGE/BACKTRACK>","direction":"before|after",
 "timestamp":<float seconds for EXPLORE_SLOT>,
 "missing_info":"what evidence is needed"}
Rules: if budget_left==1 always ANSWER; only ANSWER if genuinely sufficient;
prefer FOLLOW_EDGE when a node has strong typed edges; prefer EXPLORE_SLOT for an
unexplored temporal region; use BACKTRACK sparingly."""

# Caption of a visited moment (3 frames over +/-5 s).
CAPTION = """\
You receive a few frames from one moment of a video.
Return ONLY JSON: {"observation":"what is visible (2 sentences)",
 "change":"what changes (1 sentence, or 'static')","objects":["key","objects"],
 "actions":"actions or 'none'","causal_inference":"what caused this or 'unclear'"}"""

# Vision-grounded per-option support (-2..+2) of a visited moment.
OPTION_SUPPORT = """\
You see a few frames from ONE moment of a video, plus a multiple-choice question and \
its lettered options. By LOOKING at the frames, judge how THIS moment bears on each \
option being the correct answer: +2 strong visual evidence FOR, +1 weak, 0 irrelevant, \
-1 weak against, -2 strong against. Read fine attributes (clothing color, headgear, \
hair, who/what appears). Return ONLY JSON: {"support":[a,b,...]} one int per option."""

# Typed relation of a new node to the core node and its relevance-to-core strength.
RELATION = """\
Decide how evidence node A relates to node B (the core moment the question is \
about), for answering the question. A is the subject; each node has a timestamp.

Pick the SINGLE best relation:
- "supports": A shows visual evidence that confirms or helps answer B / the \
question (A is direct evidence for B), regardless of time.
- "causes": the event in A causally produces or triggers what happens at B \
(A = cause, B = effect).
- "precedes": A happens STRICTLY EARLIER on the video timeline than B \
(A.timestamp < B.timestamp) and leads into it. Do NOT use 'precedes' if A is \
later than B.
- "contradicts": A shows evidence that conflicts with or rules out B / a \
candidate answer.
- "irrelevant": no useful relationship to B for this question.

"strength" (0.0-1.0) = how strongly A helps answer the question (its relevance \
to the core), independent of which relation type was chosen.
Return ONLY JSON:
{"relation":"supports|causes|precedes|contradicts|irrelevant",
 "strength":0.0-1.0,"reasoning":"why (1 sentence)"}"""

# Usefulness check for nodes whose relation to the core is weak.
NODE_EVAL = """\
Evaluate a newly sampled evidence node for a video QA task. Return ONLY JSON:
{"status":"useful|redundant|irrelevant","reason":"1 sentence",
 "explore_further":true or false}"""

# LVNet baseline: keyword bag extracted from the options.
LVNET_KEYWORDS = """\
You are given a multiple-choice video question with its options. Extract a BAG \
of atomic VISUAL keywords (concrete objects, activities, places, attributes) \
mentioned COLLECTIVELY across the options -- the things one would look for in \
the video to tell the options apart. Do NOT include the question's generic \
words or pick an answer. Return ONLY JSON: {"keywords":["...","..."]}"""
