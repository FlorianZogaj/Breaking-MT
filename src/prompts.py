# == Gen prompt
MODIFICATION_PROMPT = """\
Please rewrite the following sentence so that it becomes \
slightly more complex and harder to understand, while still \
remaining logical and roughly the same length. Do not change the meaning. \
Only output the rewritten sentence, no comments or additional text. \
Only increase the linguistic complexity. Sentence you have to change: {sentence}"""

NEW_PROMPT = """You are a rewriting assistant.
Rewrite the given English sentence(s) so that it becomes harder to translate into other languages,
while preserving the original meaning as much as possible.

Guidelines:
- Keep the rewrite fluent, grammatical English (no gibberish).
- You may increase translation difficulty by using idioms, phrasal verbs, wordplay, ambiguity,
  nested clauses, unusual but natural collocations, or culturally-bound expressions.
- Do NOT add explanations, commentary, or multiple options.
- Keep length roughly similar: between half and twice the original length.

Only output the rewritten sentence(s), no comments or additional text.
Sentence(s) you have to change: {sentence}"""


USER_PROMPT = """You are a rewriting assistant.
Rewrite the given English sentence so that it becomes harder to translate into other languages,
while preserving the original meaning as much as possible.

Guidelines:
- Keep the rewrite fluent, grammatical English (no gibberish).
- You may increase translation difficulty by using idioms, phrasal verbs, wordplay, ambiguity,
  nested clauses, unusual but natural collocations, or culturally-bound expressions.
- Do NOT add explanations, commentary, or multiple options.
- Keep length roughly similar: between half and twice the original length.
- Keep names, numbers, dates, and proper nouns unchanged unless absolutely necessary.

The original sentence is:

{sentence}

Output EXACTLY in this format, and only this format:
<rewrite>
...
</rewrite>

Important:
- The output must start with <rewrite> and end with </rewrite>.

Output:
<rewrite>
"""

# == Evaluation judge prompt
JUDGE_PROMPT = (
    "Analyze the following text and return the answer in JSON. "
    "We want to determine the following attributes:\n"
    "- naturalness: on a scale from 0 (wholly unnatural) to 100 (fully human-like).\n"
    "- word rarity: on average, how rare are the words from 0 (used daily) to 100 (unknown).\n"
    "- syntax complexity: on a scale from 0 (simplest) to 100 (most complex).\n"
    "- topics: list of 1 to 5 topics that the text is about.\n"
    "Provide only the output in JSON and nothing else.\n"
    'The output should look like this: {{"naturalness": 80, "word rarity": 50, '
    '"syntax complexity": 70, "topics": ["science", "technology"]}}\n'
    "The sentence to analyze is: > {source_text}"
)
