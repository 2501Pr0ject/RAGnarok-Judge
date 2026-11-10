# ruff: noqa: E501  (patterns are kept on one line each where that reads better)
"""Meta sentences: sentences of an answer with no checkable content. Abstentions, apart.

Two meta categories only:

    announcement   announces a summary, an answer or steps, and says nothing else
    courtesy       offers help, hopes to have helped, closes the answer

A pattern has to cover the whole sentence. A sentence that carries a claim is therefore not
meta, whatever its opening: "Based on the given passages, <claim>" does not match, and neither
does "The passages do not mention <something>", which is a claim about the context.

Abstentions, sentences saying that no answer can be given, are not meta: an abstention
asserts that the context does not answer the question, which can be true or false (in the
audit of 99 training sentences, one such sentence was judged contradicted). Whether it is
true depends on the question, which no judge of the protocol receives: no judge can check an
abstention, and the metrics on this category describe how the judges behave on these
sentences, not a verification. They form a category of their own, matched by a second list
and reported separately in the secondary metrics. Decided on 2026-11-10, after the
validation of 30 matched sentences and 15 sentences left out by the meta list; the labels of
the training set do not change.

Left out on purpose, since precision comes first (the lists only serve secondary metrics):

    announcements that name a subject    an announcement of steps that says what the steps
                                         are for, for instance repotting a plant
    fragments without a proposition      a bare number, a passage number used as a citation
                                         marker

Matching. The sentence is normalised (straight quotes, leading list markers and quotes
removed, trailing closing quotes, brackets and asterisks removed, lower case, single spaces),
then compared with every pattern by full match.

The lists were written on sentences of the training split only.

    python analysis/meta_match.py export analysis/meta_patterns.json analysis/abstention_patterns.json
    python analysis/meta_match.py count ragtruth-out/train-rule5-rule1.jsonl
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

VERSION = "2026-11-10"

LEAD = r"(?:(?:sure|okay|ok|certainly|of course|alright|great)[,!.]? )?"
CONNECTOR = r"(?:(?:therefore|thus|so|hence|consequently|in conclusion|as a result|unfortunately|note)[,:]? )?"
COUNT = r"(?:two|three|four|five|six|\d+)"
BASIS = (
    r"based (?:solely |only |purely )?on (?:the |these |this )?(?:" + COUNT + r" )?"
    r"(?:given |provided |above |following )?(?:passages?|information|context|text)(?: (?:given|provided|above))?"
)
END = r"[:.!]*"
THING = r"(?:news articles?|news story|news|articles?|story|text|passages?|content)"
LENGTH = (
    r"(?:,? (?:in|within|of|under|using)"
    r"(?: about| approximately| around| at most| no more than| less than| fewer than| up to)?"
    r" \d+ (?:words?|characters?)(?: or (?:less|fewer))?)?"
)
QUESTION = r"(?: to (?:the|your|this) questions?(?: \"[^\"]*\")?)?"
NEED = (
    r"(?:have|need) (?:any )?(?:further |other |more |additional )?"
    r"(?:questions?|assistance|help|information|clarification|details)"
    r"(?: or (?:concerns|need (?:any )?(?:further |more |more detailed )"
    r"(?:clarification|assistance|information|help|instructions)))?"
)
OFFER = (
    r"(?:please )?(?:feel free to ask(?: me)?|let me know|i would be happy to help|i'd be happy to help"
    r"|don't hesitate to ask|do not hesitate to ask)"
)

META_PATTERNS: dict[str, list[str]] = {
    "announcement": [
        # announces a summary, with an optional source and an optional length
        LEAD
        + r"(?:here is|here's|here are|below is|the following is|this is) (?:a |an |the |my |your )?"
        + r"(?:(?:brief|short|concise|quick) )?(?:\d+[- ]words? )?(?:article |news )?(?:summary|summaries|summarization|recap)"
        + r"(?: (?:that )?you (?:requested|asked for))?"
        + r"(?: of (?:the |this |your )?(?:provided |given |above |following )?"
        + THING
        + r"(?: you provided)?)?"
        + LENGTH
        + r"(?:,? as requested| you requested)?"
        + END,
        # announces the main points, with an optional source
        LEAD
        + r"(?:here are|here is|below are|the following are) (?:the |some )?"
        + r"(?:main|key|most important|important) (?:points|takeaways|highlights|details|facts)"
        + r"(?: (?:from|of|in) (?:the |this )?"
        + THING
        + r")?"
        + LENGTH
        + END,
        # announces an answer, with an optional basis after it
        LEAD
        + r"(?:here is|here's|here are) (?:my |the |a |an )?(?:brief |short |concise )?"
        + r"(?:answers?|response|attempt to (?:briefly )?answer (?:the|your|this) question)"
        + QUESTION
        + r"(?:,? "
        + BASIS
        + r")?"
        + END,
        # the basis first, then the announcement of the answer
        r"(?:"
        + BASIS
        + r"|according to (?:the )?(?:given |provided )?passages?),? (?:here is|here's|here are) "
        + r"(?:my |the |a |an )?(?:brief |short |concise )?"
        + r"(?:answers?|response|what i (?:can|could) (?:determine|find|gather))"
        + QUESTION
        + END,
        # announces the answer, with an optional connector and an optional basis before it
        CONNECTOR
        + r"(?:"
        + BASIS
        + r",? )?the (?:short |brief )?answer(?: to (?:the|your|this) question)?(?: \"[^\"]*\")? is(?: as follows)?:",
        # accepts to answer the question
        r"(?:sure|certainly)[,!.]? i can answer (?:that|this|the|your) question(?: " + BASIS + r")?" + END,
        # announces steps without naming what they are for
        LEAD
        + r"(?:here are the steps(?: to follow| involved)?|here's how(?: to do it)?"
        + r"|(?:to do (?:this|that|so),? )?(?:please )?follow these steps"
        + r"|alternatively,? you can also follow these steps|the steps are(?: as follows)?):",
    ],
    "courtesy": [
        # hopes to have helped
        r"(?:i )?hope (?:this|that|it|these (?:tips|ideas|suggestions|steps))(?: information| answer| summary)?"
        + r" (?:helps?(?: you)?(?: out)?|answers (?:your|the) question)"
        + END,
        # offers further help, with an optional condition after it
        r"(?:please )?(?:let me know|feel free to ask(?: me)?|do not hesitate to ask|don't hesitate to ask)"
        + r"(?: if you "
        + NEED
        + r")?"
        + END,
        # the condition first, then the offer of help
        r"if you " + NEED + r",? " + OFFER + END,
        r"if there(?:'s| is) anything else(?: (?:you need help with|i can help(?: you)? with))?,? " + OFFER + END,
        r"(?:please )?let me know if there(?:'s| is| are) (?:anything else|any (?:other|further|more) questions)"
        + r"(?: i can (?:help|assist)(?: you)?(?: with)?)?"
        + END,
        # accepts to help
        r"(?:sure|certainly|of course|absolutely)[,!.]? (?:i can|i'd be happy to|i would be happy to|i'll|i will|happy to)"
        + r" help(?: you)?(?: with (?:that|this))?"
        + END,
        # closings of one or two words
        r"(?:and )?that's it" + END,
        r"(?:sure|certainly|of course|okay|ok|great)[.!]+",
        r"thank you(?: for (?:asking|your question|reading))?" + END,
        r"good luck" + END,
    ],
}

ABSTENTION_PATTERNS: dict[str, list[str]] = {
    "abstention": [
        # cannot answer, with an optional basis
        CONNECTOR
        + r"(?:i am |i'm |we are |i was )?unable to (?:provide an answer|answer)(?: (?:the|this|your|that) question)?"
        + r"(?: (?:"
        + BASIS
        + r"|based (?:solely |only )?on (?:this|the) passage"
        + r"|(?:with|from) (?:the )?(?:given|provided) (?:passages|information)))?"
        + END,
        # first person, cannot answer, with an optional connector and an optional basis
        CONNECTOR
        + r"i (?:cannot|can't|can not|am not able to|do not have enough information to|don't have enough information to) "
        + r"(?:provide (?:an|a (?:definitive|specific|clear|direct|accurate|precise|complete)) answer|answer)"
        + r"(?: (?:to )?(?:the|this|your|that) question)?"
        + r"(?: (?:"
        + BASIS
        + r"|without (?:additional|more|further) (?:information|context)))?"
        + END,
        # the passages or the context lack the information needed
        CONNECTOR
        + r"(?:the |these )?(?:given |provided )?(?:passages?(?: \d+)?|context|information(?: provided)?) (?:do|does) not "
        + r"(?:provide|contain|offer|give|include|have) (?:enough|sufficient|adequate|any|the necessary|relevant) "
        + r"(?:relevant )?information to (?:answer|address) (?:the|this|your|that) (?:specific )?question"
        + END,
        CONNECTOR
        + r"there is (?:not enough|no|insufficient) information(?: (?:in|from) (?:the )?(?:given |provided )?passages?)?"
        + r" to answer (?:the|this|your) question"
        + END,
        CONNECTOR + r"(?:the|this|your) question cannot be answered(?: " + BASIS + r")?" + END,
    ],
}

PATTERNS = {**META_PATTERNS, **ABSTENTION_PATTERNS}
_COMPILED = {category: [re.compile(pattern) for pattern in patterns] for category, patterns in PATTERNS.items()}
_LEADING = re.compile(r"^[\s\-\*•\"'(\[]*(?:\d+[.)]\s*)?")
_TRAILING = re.compile(r"[\s\"')\]*]+$")


def normalise(sentence: str) -> str:
    text = sentence.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    text = text.replace(" ", " ")
    text = _LEADING.sub("", text)
    text = _TRAILING.sub("", text)
    return re.sub(r"\s+", " ", text).strip().lower()


def _category(sentence: str, lists: dict[str, list[str]]) -> tuple[str, int] | None:
    text = normalise(sentence)
    for category in lists:
        for index, pattern in enumerate(_COMPILED[category]):
            if pattern.fullmatch(text):
                return category, index
    return None


def meta_category(sentence: str) -> tuple[str, int] | None:
    """(category, index of the pattern in its category), or None when the sentence is not meta."""
    return _category(sentence, META_PATTERNS)


def is_abstention(sentence: str) -> bool:
    """True when the whole sentence says that no answer can be given."""
    return _category(sentence, ABSTENTION_PATTERNS) is not None


NORMALISATION = [
    "curly quotes to straight quotes, non-breaking space to space",
    "leading spaces, list markers, opening quotes and brackets, and numbering removed",
    "trailing spaces, closing quotes, brackets and asterisks removed",
    "single spaces, lower case",
]


def export(meta_path: Path, abstention_path: Path) -> None:
    meta = {
        "version": VERSION,
        "definition": "sentences with no checkable content: announcement, courtesy",
        "not_meta": "abstentions: a sentence saying that no answer can be given asserts something that depends "
        "on the question, which no judge receives; they are matched by the abstention list and reported as "
        "behaviour, not as a verification",
        "matching": "full match of a pattern on the normalised sentence",
        "left_out_on_purpose": [
            "announcements that name a subject",
            "fragments without a proposition, such as a bare number or a citation marker",
        ],
        "normalisation": NORMALISATION,
        "patterns": META_PATTERNS,
    }
    abstention = {
        "version": VERSION,
        "definition": "sentences saying that no answer can be given, without naming what is missing; "
        "reported as a category of their own, apart from meta sentences",
        "matching": "full match of a pattern on the normalised sentence",
        "normalisation": NORMALISATION,
        "patterns": ABSTENTION_PATTERNS,
    }
    meta_path.write_text(json.dumps(meta, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    abstention_path.write_text(json.dumps(abstention, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


def count(path: Path) -> None:
    by_category: Counter[str] = Counter()
    by_label: Counter[tuple[str, str]] = Counter()
    total = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            total += 1
            hit = meta_category(record["sentence"])
            abstains = is_abstention(record["sentence"])
            if hit and abstains:
                raise SystemExit(f"{record['id']}: matched by the meta list and by the abstention list")
            if hit:
                by_category[hit[0]] += 1
                by_label[(hit[0], record["label"])] += 1
            if abstains:
                by_category["abstention"] += 1
                by_label[("abstention", record["label"])] += 1
    meta_total = sum(by_category[category] for category in META_PATTERNS)
    print(f"{path}: {total} sentences, {meta_total} meta, {by_category['abstention']} abstentions")
    for category in PATTERNS:
        labels = {label: n for (cat, label), n in sorted(by_label.items()) if cat == category}
        print(f"  {category}: {by_category[category]} {labels}")


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "export":
        export(Path(sys.argv[2]), Path(sys.argv[3]))
    elif len(sys.argv) == 3 and sys.argv[1] == "count":
        count(Path(sys.argv[2]))
    else:
        raise SystemExit(__doc__)
