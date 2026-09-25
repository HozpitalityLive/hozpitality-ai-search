"""Deterministic interpretation of a chat message relative to conversation state.

This layer is authoritative. It decides *what the user asked for* (search,
refine, show more, compare, open a result, reset, ...) and *which state fields
the message explicitly changes*. The LLM never makes these decisions.

Guiding rules
-------------
* Only fields explicitly changed by the current message are modified.
* The entity never changes implicitly. A role word ("chef") cannot turn a job
  search into a professional search; only an explicit module noun can
  ("show me professionals instead").
* Removals ("remove the accommodation requirement", "anywhere, not just
  Dubai") are detected first and their text is consumed, so the removed value
  can never be re-added by the same sentence.
* Result references ("the second one", "the first three") are resolved to
  positions here and to record IDs by the chat service, never by the LLM.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from .normalization import normalize, tokens
from .intent import RECORDS_QUESTION_RE, classify
from .schema_map import filter_applies
from .query_understanding import (
    CITIES,
    CITY_COUNTRY,
    COUNTRIES,
    ENTITY_WORDS,
    PROFESSIONAL_ROLE_TERMS,
    SearchPlan,
    _entity_match,
    understand,
)

ENTITY_LABELS = {
    "job": ("job", "jobs"),
    "professional": ("professional", "professionals"),
    "company": ("company", "companies"),
    "product": ("product", "products"),
    "article": ("article", "articles"),
    "event": ("event", "events"),
    "award": ("award", "awards"),
    "faq": ("FAQ", "FAQs"),
}

# Generic nouns that name a module only weakly. They set the entity when the
# conversation has none, but never switch an existing entity: "only management
# positions" inside a professional search must not become a job search.
WEAK_ENTITY_TERMS = {
    "position",
    "positions",
    "role",
    "roles",
    "opening",
    "openings",
    "career",
    "careers",
    "employment",
    "people",
    "person",
    "persons",
    "expert",
    "experts",
    "talent",
    "question",
    "questions",
    "business",
    "businesses",
    "story",
    "stories",
    "post",
    "posts",
    "honour",
    "honours",
    "honor",
    "honors",
    "recognition",
}


FILTER_LABELS = {
    "level": "level",
    "accommodation": "accommodation",
    "experience": "experience",
    "salary_min": "salary",
    "salary_currency": "salary currency",
    "employment_type": "employment type",
    "department": "department",
    "industry": "industry",
    "category": "category",
    "verified": "verified",
    "featured": "featured",
    "currently_working": "currently working",
}

# Conversational filler that must never become a search keyword.
FILLER_WORDS = {
    "actually",
    "instead",
    "rather",
    "just",
    "also",
    "then",
    "now",
    "okay",
    "ok",
    "please",
    "pls",
    "plz",
    "maybe",
    "what",
    "about",
    "how",
    "change",
    "changed",
    "switch",
    "that",
    "it",
    "to",
    "them",
    "these",
    "those",
    "this",
    "result",
    "results",
    "ones",
    "one",
    "some",
    "any",
    "show",
    "give",
    "filter",
    "filters",
    "restrict",
    "restricted",
    "limit",
    "limited",
    "only",
    "prefer",
    "preferably",
    "but",
    "yes",
    "no",
    "sure",
    "again",
    "same",
    "search",
    "let's",
    "lets",
    "let",
    "us",
    "can",
    "you",
    "u",
    "could",
    "would",
    "should",
    "will",
    "thanks",
    "thank",
    "hmm",
    "hi",
    "hello",
    "hey",
    "well",
    "so",
    "too",
    "else",
    "there",
    "here",
    "is",
    "are",
    "was",
    "be",
    "do",
    "does",
    "did",
    "have",
    "has",
    "get",
    "got",
    "see",
    "want",
    "wanted",
    "need",
    "needs",
    "looking",
    "look",
    "find",
    "list",
    "try",
    "go",
    "back",
    "with",
    "without",
    "not",
    "don't",
    "dont",
    "doesn't",
    "either",
    "or",
    "and",
    "the",
    "a",
    "an",
    "of",
    "in",
    "at",
    "on",
    "for",
    "from",
    "by",
    "into",
    "near",
    "around",
    "within",
    "based",
    "located",
    "location",
    "locations",
    "place",
    "places",
    "area",
    "city",
    "country",
    "anywhere",
    "everywhere",
    "worldwide",
    "globally",
    "global",
    "level",
    "levels",
    "requirement",
    "requirements",
    "constraint",
    "condition",
    "option",
    "options",
    "more",
    "less",
    "other",
    "others",
    "additional",
    "another",
    "next",
    "page",
    "which",
    "who",
    "where",
    "when",
    "why",
    "me",
    "my",
    "i",
    "we",
    "our",
    "type",
    "kind",
    "kinds",
    "sort",
    "all",
    "every",
    "each",
    "new",
    "start",
    "over",
    "reset",
    "clear",
    "remove",
    "drop",
    "delete",
    "ignore",
    "forget",
    "yeah",
    "yep",
    "nope",
    "positions",
    "position",
    "roles",
    "role",
    "openings",
    "opening",
    "jobs",
    "job",
    "vacancy",
    "vacancies",
    "tell",
    "detail",
    "details",
    "info",
    "information",
    "describe",
    "explain",
}

REFINEMENT_LEAD = re.compile(
    r"^(?:only|just|with|without|and|also|plus|but|that|which|who|preferably|must|should|"
    r"excluding|except|in|at|near|for|having|offering|providing|who\s+are|that\s+are|"
    r"ones?|those|the\s+ones)\b"
)
RESTRICTIVE = re.compile(
    r"\b(?:only|just|strictly|exclusively|must\s+be|limit(?:ed)?\s+to|restrict(?:ed)?\s+to)\b"
)
SEARCH_VERB = re.compile(
    r"^(?:please\s+)?(?:find|search(?:\s+for)?|show(?:\s+me)?|list|get(?:\s+me)?|"
    r"i(?:'m|\s+am)\s+looking\s+for|looking\s+for|i\s+(?:need|want)|give\s+me|"
    r"are\s+there|any)\b"
)

STRONG_SOURCES = {"text", "concept", "intent"}
CONCEPT_FILTERS = {"is_supplier", "industry_context", "supplier_category"}
SWITCH_RE = re.compile(
    r"\b(?:instead|rather|switch(?:\s+it)?\s+to|what\s+about|how\s+about|same\s+for|"
    r"change\s+(?:it|that|this)\s+to|actually)\b"
)

ORDINALS = {
    "first": 1,
    "1st": 1,
    "one": 1,
    "second": 2,
    "2nd": 2,
    "two": 2,
    "third": 3,
    "3rd": 3,
    "three": 3,
    "fourth": 4,
    "4th": 4,
    "four": 4,
    "fifth": 5,
    "5th": 5,
    "five": 5,
}
_ORD = r"(?:first|second|third|fourth|fifth|1st|2nd|3rd|4th|5th)"
_NUM = r"(?:one|two|three|four|five|[1-5])"


@dataclass
class TurnIntent:
    action: str = (
        "search"  # search|more|compare|detail|related_entity|reset|smalltalk|clarify
    )
    refs: list[int] = field(
        default_factory=list
    )  # 0-based positions in the latest list
    ref_mode: str | None = None  # ordinal|all|focus|previous|last
    requested_count: int | None = None
    question_field: str | None = None
    open_link: bool = False
    compare_fields: list[str] = field(default_factory=list)
    target_entity: str | None = None

    set_entity: str | None = None
    entity_changed: bool = False
    set_location: dict[str, Any] | None = None
    remove_location: bool = False
    set_keywords: list[str] | None = None
    add_keywords: list[str] = field(default_factory=list)
    set_filters: dict[str, Any] = field(default_factory=dict)
    remove_filters: list[str] = field(default_factory=list)
    strict_filters: list[str] = field(default_factory=list)
    clear_filters: bool = False
    fresh: bool = False
    smalltalk: str | None = None
    plan: SearchPlan | None = None
    # Conversation-state transition for search turns (see TRANSITIONS).
    transition: str | None = None
    transition_reason: str = ""

    def changes_search(self) -> bool:
        return bool(
            self.set_entity
            or self.set_location
            or self.remove_location
            or self.set_keywords is not None
            or self.add_keywords
            or self.set_filters
            or self.remove_filters
            or self.clear_filters
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "refs": [i + 1 for i in self.refs],
            "question_field": self.question_field,
            "compare_fields": self.compare_fields,
            "set_entity": self.set_entity,
            "set_location": self.set_location,
            "remove_location": self.remove_location,
            "set_keywords": self.set_keywords,
            "add_keywords": self.add_keywords,
            "set_filters": self.set_filters,
            "remove_filters": self.remove_filters,
            "strict_filters": self.strict_filters,
            "clear_filters": self.clear_filters,
            "fresh": self.fresh,
            "transition": self.transition,
            "transition_reason": self.transition_reason,
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _clean(message: str) -> str:
    text = " ".join((message or "").strip().split())
    text = text.replace("’", "'").replace("‘", "'")
    return text


def _low(message: str) -> str:
    return _clean(message).casefold().strip(" .!?")


def singular(token: str) -> str:
    token = token.casefold()
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 4 and token.endswith(("ches", "shes", "sses", "xes")):
        return token[:-2]
    if (
        len(token) > 3
        and token.endswith("s")
        and not token.endswith(("ss", "us", "is"))
    ):
        return token[:-1]
    return token


def normalize_keywords(values: list[str]) -> list[str]:
    out: list[str] = []
    for value in values:
        for token in tokens(value):
            if token in FILLER_WORDS:
                continue
            word = singular(token)
            if word and word not in out:
                out.append(word)
    return out[:12]


def entity_label(entity: str | None, count: int = 2) -> str:
    if not entity:
        return "result" if count == 1 else "results"
    one, many = ENTITY_LABELS.get(entity, (entity, entity + "s"))
    return one if count == 1 else many


# ---------------------------------------------------------------------------
# Action detection
# ---------------------------------------------------------------------------

RESET_RE = re.compile(
    r"^(?:(?:let'?s\s+|please\s+|can\s+we\s+|i\s+want\s+to\s+)?"
    r"(?:start\s+(?:over|again|fresh|a\s+new\s+(?:search|chat|conversation))|"
    r"new\s+(?:search|chat|conversation|topic)|"
    r"reset(?:\s+(?:the\s+)?(?:search|chat|conversation|everything|all))?|"
    r"clear(?:\s+(?:the\s+)?(?:search|chat|conversation|everything|all|context|history))?|"
    r"begin\s+again|forget\s+(?:everything|all\s+(?:of\s+)?that)))"
    r"(?:\s+please)?$"
)

SMALLTALK = {
    r"^(?:hi|hello|hey|hiya|good\s+(?:morning|afternoon|evening))(?:\s+there)?$": "greeting",
    r"^(?:thanks|thank\s+you|thank\s+u|thx|ty|cheers|great,?\s+thanks)(?:\s+(?:so\s+much|a\s+lot))?$": "thanks",
    r"^(?:ok|okay|cool|great|nice|perfect|got\s+it|alright|sounds\s+good)$": "ack",
    r"^(?:help|what\s+can\s+you\s+do|how\s+does\s+this\s+work|who\s+are\s+you)$": "help",
}

MORE_RE = re.compile(
    r"^(?:(?:can\s+you\s+|could\s+you\s+|please\s+)?"
    r"(?:(?:show|give|get|list|load|see|find)(?:\s+me)?\s+(?:some\s+|a\s+few\s+|\d+\s+)?(?:more|others?|additional|next)"
    r"(?:\s+(?:results?|options?|matches|ones|of\s+(?:these|them|those)|like\s+(?:this|these|that)|"
    r"(?P<noun>[a-z]+)))?|"
    r"(?:some\s+)?more(?:\s+(?:results?|options?|matches|please|of\s+them|like\s+(?:these|this)|(?P<noun2>[a-z]+)))?|"
    r"next(?:\s+(?:page|results?|\d+|ones|five|5))?|"
    r"(?:is\s+there\s+)?any(?:thing|\s+more|\s+others?)\s*(?:else)?|"
    r"what\s+else(?:\s+(?:is\s+there|do\s+you\s+have))?|"
    r"(?:show\s+)?additional\s+results|keep\s+going|continue|load\s+more)"
    r")(?:\s+please)?$"
)

COMPARE_RE = re.compile(
    r"\b(?:compare|comparison|contrast|versus|vs\.?|difference(?:s)?\s+between|"
    r"side\s+by\s+side|which\s+(?:one|of\s+(?:these|them|those)|is\s+(?:better|best))|"
    r"which\s+(?:job|company|professional|product|event|one)s?\s+(?:has|have|is|offers?|pays?))\b"
)

REFERENCE_RE = re.compile(
    rf"\b(?:the\s+)?(?:{_ORD})(?:\s+(?:one|result|job|position|professional|profile|candidate|company|"
    rf"product|article|event|award|faq|item|option|match|listing))?\b"
    r"|#\s?[1-5]\b|\b(?:number|no\.?|result|option|item)\s+[1-5]\b"
    r"|\b(?:the\s+)?last\s+(?:one|result|job|item)\b"
    r"|\bthat\s+(?:one|job|result|position|company|profile|product|event|article|award|candidate)\b"
    r"|\bthis\s+(?:one|job|result|position|profile|product|event|article|award|candidate)\b"
    r"|\bthe\s+previous\s+(?:one|job|result|position|profile|company|product|event|article|award)\b"
    r"|\b(?:about|on|of|open)\s+(?:it|that|this)\b"
)

_ROLE_WORDS = {singular(t) for role in PROFESSIONAL_ROLE_TERMS for t in tokens(role)}

DETAIL_VERB_RE = re.compile(
    r"\b(?:tell\s+me\s+(?:more\s+)?about|more\s+(?:about|on|details|info)|details?|"
    r"describe|explain|open|view|show(?:\s+me)?|what(?:'s|\s+is|\s+are)?|who|where|which|"
    r"how\s+much|info(?:rmation)?|summar(?:y|ize|ise)|link|apply)\b"
)

RELATED_COMPANY_RE = re.compile(
    r"\b(?:which|what|who|show|list|find)\b.*\b(?:compan(?:y|ies)|employers?|hotels?|organi[sz]ations?)\b"
    r".*\b(?:hiring|posted|posting|offer(?:s|ing)?|behind|advertis(?:ed|ing)|recruiting|are\s+these|"
    r"is\s+(?:it|this|that))\b"
    r"|\bwho\s+is\s+hiring\s+(?:for\s+)?(?:these|those|them)\b"
)

QUESTION_FIELDS = [
    (
        "company",
        r"\b(?:what|which|who)\b.*\b(?:compan(?:y|ies)|employer|hotel|organi[sz]ation)\b|"
        r"\bwho\s+(?:is\s+)?(?:hiring|posted|offers?)\b|\bthat\s+company\b|\bthe\s+company\b|\bemployer\b",
    ),
    (
        "location",
        r"\bwhere\b|\blocation\b|\bwhich\s+city\b|\bwhat\s+city\b|\baddress\b",
    ),
    (
        "salary",
        r"\bsalary\b|\bpay(?:s|ing)?\b|\bcompensation\b|\bhow\s+much\b|\bwage\b",
    ),
    ("experience", r"\bexperience\b|\byears\b"),
    ("url", r"\blink\b|\burl\b|\bapply\b|\bopen\b|\bwebsite\b"),
    ("date", r"\bwhen\b|\bdate\b|\bposted\b"),
    ("accommodation", r"\baccommodation\b|\bhousing\b"),
    (
        "employment_type",
        r"\bfull[- ]?time\b|\bpart[- ]?time\b|\bcontract\b|\bemployment\s+type\b",
    ),
]

COMPARE_FIELD_TERMS = {
    "salary": r"\bsalar(?:y|ies)\b|\bpay\b|\bcompensation\b",
    "location": r"\blocations?\b|\bcit(?:y|ies)\b|\bwhere\b",
    "experience": r"\bexperience\b|\byears\b",
    "company": r"\bcompan(?:y|ies)\b|\bemployers?\b",
    "level": r"\blevels?\b|\bseniority\b",
    "accommodation": r"\baccommodation\b|\bhousing\b|\bbenefits?\b",
    "employment_type": r"\bemployment\b|\bfull[- ]?time\b|\bpart[- ]?time\b|\bcontract\b",
    "date": r"\bdates?\b|\bposted\b|\bnewest\b|\bmost\s+recent\b",
    "category": r"\bcategor(?:y|ies)\b",
}


def _parse_refs(low: str) -> tuple[list[int], str | None, int | None]:
    """Return (1-based positions, mode, requested count)."""
    # "first three" / "top 3" / "first 2"
    m = re.search(
        rf"\b(?:the\s+)?(?:first|top)\s+({_NUM})\b(?!\s*(?:years?|yrs?))", low
    )
    if m:
        n = ORDINALS.get(m.group(1)) or int(m.group(1))
        return list(range(1, n + 1)), "ordinal", n
    m = re.search(rf"\b(?:the\s+)?last\s+({_NUM})\b", low)
    if m:
        n = ORDINALS.get(m.group(1)) or int(m.group(1))
        return [-i for i in range(n, 0, -1)], "last", n

    positions: list[int] = []
    for match in re.finditer(
        rf"\b({_ORD})\b|#\s?([1-5])\b|\b(?:number|no\.?|result|option|item)\s+([1-5])\b",
        low,
    ):
        word = match.group(1)
        value = ORDINALS[word] if word else int(match.group(2) or match.group(3))
        if value not in positions:
            positions.append(value)
    # "1 and 3", "2 & 4"
    m = re.search(
        r"\b([1-5])\s*(?:,|and|&)\s*([1-5])(?:\s*(?:,|and|&)\s*([1-5]))?\b", low
    )
    if m and not positions:
        positions = [int(g) for g in m.groups() if g]
    if re.search(r"\b(?:the\s+)?last\s+(?:one|result|job|item)\b", low):
        positions.append(-1)
    if positions:
        return positions, "ordinal", len(positions)
    if re.search(
        r"\b(?:the\s+)?previous\s+(?:one|job|result|position|profile|company|product|event|article|award)\b",
        low,
    ):
        return [], "previous", 1
    if re.search(
        r"\b(?:that|this)\s+(?:one|job|result|position|company|profile|product|event|article|award|candidate)\b",
        low,
    ) or re.search(r"\b(?:it|its)\b", low):
        return [], "focus", 1
    if re.search(
        r"\b(?:these|them|those|all(?:\s+of\s+them)?|both|the\s+results)\b", low
    ):
        return [], "all", None
    return [], None, None


def _question_field(low: str) -> str | None:
    for name, pattern in QUESTION_FIELDS:
        if re.search(pattern, low):
            return name
    return None


# ---------------------------------------------------------------------------
# Removals
# ---------------------------------------------------------------------------

_FILTER_TERMS: list[tuple[str, str]] = [
    ("accommodation", r"(?:staff\s+)?(?:accommodation|housing|lodging)"),
    (
        "level",
        r"(?:management|managerial|manager|senior(?:ity)?|junior|mid(?:-|\s)?level|executive|"
        r"entry(?:-|\s)level|intern(?:ship)?|level|seniority)",
    ),
    ("experience", r"(?:experience|years(?:\s+of\s+experience)?)"),
    ("salary_min", r"(?:salary|pay|compensation)"),
    (
        "employment_type",
        r"(?:full[- ]?time|part[- ]?time|remote|contract|temporary|employment\s+type)",
    ),
    ("department", r"department"),
    ("industry", r"industry"),
    ("category", r"category"),
    ("verified", r"verified"),
    ("featured", r"featured"),
    ("location", r"(?:location|city|country|place)"),
]

_REMOVE_VERBS = (
    r"(?:remove|drop|clear|delete|ignore|forget|lose|skip|cancel|undo|without|"
    r"no\s+longer\s+(?:need|require|want)|(?:i\s+)?(?:don'?t|do\s+not|doesn'?t|does\s+not)\s+(?:need|require|want|care\s+about)|"
    r"no\s+need\s+for|(?:don'?t|do\s+not)\s+(?:restrict|limit)(?:\s+(?:it|this|that|them|results?|the\s+search))?\s+to|"
    r"not\s+(?:just|only)|regardless\s+of(?:\s+the)?)"
)

# "any level", "all salaries": removal of a restriction without a verb.
_ANY_FILTER = [
    ("level", r"\b(?:any|all|every)\s+(?:seniority|levels?|experience\s+levels?)\b"),
    ("experience", r"\b(?:any|all)\s+(?:amount\s+of\s+)?experience\b(?!\s+levels?)"),
    ("salary_min", r"\b(?:any|all)\s+salar(?:y|ies)\b"),
    ("employment_type", r"\b(?:any|all)\s+(?:employment|job)\s+types?\b"),
    ("category", r"\b(?:any|all)\s+categor(?:y|ies)\b"),
    ("department", r"\b(?:any|all)\s+departments?\b"),
    ("industry", r"\b(?:any|all)\s+industr(?:y|ies)\b"),
]
_REMOVE_TAIL = (
    r"(?:\s+(?:filter|requirement|restriction|constraint|condition|criteri(?:a|on)|preference|"
    r"level|levels|positions?|roles?|jobs?|ones|option))?"
)


def _place_names() -> str:
    names = sorted(set(CITIES) | set(COUNTRIES), key=len, reverse=True)
    return "|".join(re.escape(n) for n in names)


_PLACES = _place_names()


def _detect_removals(
    low: str, state: dict[str, Any]
) -> tuple[list[str], bool, bool, str]:
    """Return (filter keys removed, location removed, clear all, residual text)."""
    removed: list[str] = []
    remove_location = False
    clear_all = False
    residual = low

    def consume(pattern: str) -> bool:
        nonlocal residual
        new, count = re.subn(pattern, " ", residual)
        residual = new
        return count > 0

    if consume(
        r"\b(?:remove|clear|drop|reset|delete|ignore)\s+(?:all\s+(?:the\s+)?|the\s+|every\s+)?filters?\b"
        r"|\bno\s+filters?\b|\bwithout\s+(?:any\s+)?filters?\b"
    ):
        clear_all = True

    # Location removal ("anywhere", "not just Dubai", "remove the location").
    state_location = state.get("location") or {}
    current_places = [
        p
        for p in (
            state_location.get("city"),
            state_location.get("country"),
            state_location.get("raw"),
        )
        if p
    ]
    place_alt = "|".join(re.escape(p.casefold()) for p in current_places)
    place_pattern = _PLACES + (("|" + place_alt) if place_alt else "")
    location_patterns = [
        rf"\bnot\s+(?:just|only)\s+(?:in\s+)?(?:{place_pattern})\b",
        rf"\b(?:don'?t|do\s+not)\s+(?:restrict|limit)(?:\s+(?:it|this|that|them|results?|the\s+search))?\s+to\s+(?:{place_pattern})\b",
        rf"\b(?:outside(?:\s+of)?|beyond)\s+(?:just\s+)?(?:{place_pattern})\b",
        r"\b(?:remove|drop|clear|delete|ignore|forget|without)\s+(?:the\s+|a\s+|any\s+)?(?:location|city|country|place)"
        r"(?:\s+(?:filter|requirement|restriction|constraint|preference))?\b",
        r"\b(?:any|all|every|no\s+specific|no\s+particular)\s+(?:location|city|country|place)s?\b",
        r"\b(?:location|city|country)\s+(?:doesn'?t|does\s+not)\s+matter\b",
        r"\banywhere(?:\s+(?:else|in\s+the\s+world))?\b|\beverywhere\b|\bworld\s*-?\s*wide\b|\bglobal(?:ly)?\b|\ball\s+over\b",
    ]
    for pattern in location_patterns:
        if consume(pattern):
            remove_location = True

    for key, term in _FILTER_TERMS:
        if key == "location":
            continue
        pattern = rf"\b{_REMOVE_VERBS}\s+(?:the\s+|a\s+|any\s+|an\s+|of\s+the\s+)?{term}{_REMOVE_TAIL}\b"
        if consume(pattern):
            removed.append(key)
    for key, pattern in _ANY_FILTER:
        if consume(pattern) and key not in removed:
            removed.append(key)
    if "salary_min" in removed:
        removed.append("salary_currency")
    return removed, remove_location, clear_all, " ".join(residual.split())


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def interpret(message: str, state: dict[str, Any] | None = None) -> TurnIntent:
    state = state or {}
    text = _clean(message)
    low = _low(message)
    intent = TurnIntent()
    has_search = bool(state.get("entity") or state.get("keywords"))
    last_results = state.get("last_results") or []

    if not low:
        intent.action = "clarify"
        return intent

    if RESET_RE.match(low):
        intent.action = "reset"
        return intent

    for pattern, kind in SMALLTALK.items():
        if re.match(pattern, low):
            intent.action = "smalltalk"
            intent.smalltalk = kind
            return intent

    # "What companies are hiring these chefs?" -> companies behind the results.
    if (
        last_results
        and RELATED_COMPANY_RE.search(low)
        and re.search(
            r"\b(?:these|those|them|this|that|it|the\s+(?:first|second|third|fourth|fifth|last))\b",
            low,
        )
    ):
        refs, mode, count = _parse_refs(low)
        intent.action = "related_entity"
        intent.target_entity = "company"
        intent.refs = (
            [r - 1 if r > 0 else r for r in refs] if mode in {"ordinal", "last"} else []
        )
        intent.ref_mode = (
            mode if mode in {"ordinal", "last", "focus", "previous"} else "all"
        )
        return intent

    # Comparison of results already shown.
    if COMPARE_RE.search(low) and (last_results or re.search(r"\bcompare\b", low)):
        refs, mode, count = _parse_refs(low)
        intent.action = "compare"
        intent.ref_mode = mode or "all"
        intent.requested_count = count
        intent.refs = [r - 1 if r > 0 else r for r in refs]
        intent.compare_fields = [
            name
            for name, pattern in COMPARE_FIELD_TERMS.items()
            if re.search(pattern, low)
        ]
        return intent

    # References to a specific result ("tell me more about the first one").
    has_reference = bool(REFERENCE_RE.search(low))
    bare_reference = bool(
        re.fullmatch(
            rf"(?:and\s+)?(?:the\s+)?(?:{_ORD}|last)(?:\s+(?:one|result|job|item|option))?|#\s?[1-5]|(?:number|result|option)\s+[1-5]",
            low,
        )
    )
    if (
        (last_results or state.get("focus"))
        and has_reference
        and (bare_reference or DETAIL_VERB_RE.search(low))
    ):
        refs, mode, _ = _parse_refs(low)
        if mode in {"ordinal", "last", "focus", "previous"}:
            intent.action = "detail"
            intent.ref_mode = mode
            intent.refs = [r - 1 if r > 0 else r for r in refs][:1]
            intent.question_field = _question_field(low)
            intent.open_link = bool(re.search(r"\b(?:open|link|url|apply|view)\b", low))
            return intent

    # "Show me more" (same search, next results).
    more = MORE_RE.match(low)
    if more:
        noun = more.group("noun") or more.group("noun2")
        if not noun:
            intent.action = "more"
            return intent
        noun_entity = _entity_match(noun)[0]
        if noun_entity == state.get("entity") or (
            noun_entity is None and singular(noun) in _ROLE_WORDS
        ):
            # "more jobs" / "more chefs": same search, next page.
            intent.action = "more"
            return intent
        # "more professionals" while searching jobs -> entity change;
        # "more senior" -> a refinement. Both are handled as searches.

    # Information questions ("How do I apply for a job?") are answered from
    # FAQ records - never turned into a job search or a clarification.
    classification = classify(text)
    if classification.kind == "faq":
        intent.action = "faq"
        intent.transition = "faq"
        intent.transition_reason = classification.reason
        intent.plan = understand(text)
        return intent
    if classification.kind == "facet":
        intent.action = "facet"
        intent.transition = "facet"
        intent.transition_reason = classification.reason
        intent.plan = understand(text)
        return intent

    return _interpret_search(text, low, state, intent, has_search)


def _interpret_search(
    text: str, low: str, state: dict[str, Any], intent: TurnIntent, has_search: bool
) -> TurnIntent:
    intent.action = "search"
    state_entity = state.get("entity")
    pending = state.get("pending") or {}

    removed, remove_location, clear_all, residual = _detect_removals(low, state)
    intent.remove_filters = removed
    intent.remove_location = remove_location
    intent.clear_filters = clear_all

    # "instead" / "switch to" phrasing is a signal but never required.
    residual_text = re.sub(
        r"\b(?:actually|instead|rather|switch(?:\s+it)?\s+to|change\s+(?:it|that|this)\s+to|"
        r"what\s+about|how\s+about|and\s+what\s+about|now|then)\b",
        " ",
        residual,
    )
    residual_text = " ".join(residual_text.split())

    plan = understand(residual_text) if residual_text else SearchPlan()
    intent.plan = plan

    # ---- entity -------------------------------------------------------------
    # Priority: explicit module noun / schema concept / job-intent word >
    # generic noun > role word. Role words ("chef") and generic nouns
    # ("positions") never change an existing entity.
    explicit_entity = plan.entity if plan.entity_source in STRONG_SOURCES else None
    if plan.entity:
        if plan.entity_source in {"role", "weak"}:
            if not state_entity:
                intent.set_entity = plan.entity
        elif plan.entity != state_entity:
            intent.set_entity = plan.entity
            intent.entity_changed = bool(state_entity)

    # ---- location -------------------------------------------------------------
    if plan.city or plan.country:
        city = plan.city
        country = plan.country or (CITY_COUNTRY.get(city) if city else None)
        intent.set_location = {"city": city, "country": country, "raw": None}
    elif plan.location_text:
        raw = plan.location_text.strip()
        raw_tokens = [t for t in tokens(raw) if t not in FILLER_WORDS]
        if raw_tokens:
            intent.set_location = {"city": None, "country": None, "raw": raw.title()}
    elif pending.get("field") == "location" and not remove_location:
        # Reply to "Which location would you prefer?" with a place we do not
        # know ("Antarctica"): still an explicit location, never a keyword.
        candidate = [t for t in tokens(residual_text) if t not in FILLER_WORDS]
        if (
            candidate
            and len(candidate) <= 3
            and not (plan.entity or plan.level or plan.filters)
        ):
            intent.set_location = {
                "city": None,
                "country": None,
                "raw": " ".join(candidate).title(),
            }

    # ---- filters ---------------------------------------------------------------
    filters: dict[str, Any] = dict(plan.filters or {})
    if plan.level:
        filters["level"] = plan.level
    if plan.experience is not None:
        filters["experience"] = plan.experience
    if plan.department:
        filters["department"] = plan.department
    if plan.industry:
        filters["industry"] = plan.industry
    if plan.category:
        filters["category"] = plan.category
    for key in removed:
        filters.pop(key, None)
    # Filter lifetime (schema_map): a filter that the target entity does not
    # have is never applied ("kitchen" department on a company search).
    target_entity = intent.set_entity or state_entity or plan.entity
    demoted_topics: list[str] = []
    if target_entity:
        for key in [k for k in filters if not filter_applies(target_entity, k)]:
            value = filters.pop(key)
            if key in {"industry", "department", "category"} and isinstance(value, str):
                # Not a structured field of this entity, but still the user's
                # topic ("hospitality awards"): search it as a keyword.
                demoted_topics.append(value)
    intent.set_filters = filters

    # ---- keywords ----------------------------------------------------------------
    keyword_source = list(plan.keywords) + demoted_topics
    if intent.set_location and intent.set_location.get("raw"):
        place_tokens = set(tokens(intent.set_location["raw"]))
        keyword_source = [k for k in keyword_source if k not in place_tokens]
    # Level/department words already captured as filters are not keywords,
    # except real role phrases ("executive chef", "hr manager").
    keywords = normalize_keywords(keyword_source)
    if plan.level:
        role_words = {
            singular(t) for role in PROFESSIONAL_ROLE_TERMS for t in tokens(role)
        }
        keywords = [k for k in keywords if k != plan.level or k in role_words]
    if intent.set_location:
        location_words = set(
            tokens(" ".join(v for v in intent.set_location.values() if v))
        )
        keywords = [k for k in keywords if k not in location_words]

    if pending.get("field") == "location" and intent.set_location and not plan.entity:
        keywords = []

    # Keep the user's word order ("sous chef", not "chef sous"): phrase
    # matching in ranking depends on it.
    def _position(word: str) -> int:
        m = re.search(rf"\b{re.escape(word)}", residual_text)
        return m.start() if m else len(residual_text)

    keywords.sort(key=_position)

    refinement = bool(REFINEMENT_LEAD.match(residual_text or low))
    restrictive = bool(RESTRICTIVE.search(low))
    if keywords:
        if pending.get("field") == "keywords":
            intent.set_keywords = keywords
        elif refinement and has_search and not intent.set_entity:
            intent.add_keywords = keywords
        else:
            intent.set_keywords = keywords

    # ---- transition ----------------------------------------------------------
    complete_request = bool(
        SEARCH_VERB.match(residual_text or low) or RECORDS_QUESTION_RE.search(low)
    )
    switch_phrase = bool(SWITCH_RE.search(low))
    own_topic = bool(keywords or set(plan.filters or {}) & CONCEPT_FILTERS)
    if not has_search:
        intent.transition, why = "new_search", "no active search"
    elif explicit_entity and explicit_entity != state_entity:
        if switch_phrase and not own_topic and not refinement:
            intent.transition = "entity_switch"
            why = f"'{plan.entity_term}' switches {state_entity} -> {explicit_entity} keeping compatible context"
        else:
            intent.transition = "new_search"
            why = f"explicit new domain '{plan.entity_term}' ({state_entity} -> {explicit_entity})"
    elif pending.get("field") and not complete_request:
        intent.transition, why = (
            "clarification_answer",
            f"answers pending {pending.get('field')} question",
        )
    elif (
        explicit_entity
        and complete_request
        and not refinement
        and (own_topic or intent.set_location)
    ):
        intent.transition, why = "new_search", f"complete new {explicit_entity} request"
    elif intent.changes_search():
        intent.transition, why = "modification", "refines the current search"
    else:
        intent.transition, why = "continuation", "no explicit change"
    intent.transition_reason = why
    intent.fresh = intent.transition == "new_search" and has_search
    if intent.transition == "new_search" and not intent.set_entity:
        # A new search always states its own entity (it may equal the old one).
        intent.set_entity = explicit_entity or plan.entity or state_entity

    # Strict filters: explicitly required constraints.
    strict: list[str] = []
    if "level" in filters and (
        restrictive or (has_search and not keywords and not intent.set_entity)
    ):
        strict.append("level")
    if filters.get("accommodation") is True:
        strict.append("accommodation")
    intent.strict_filters = strict

    if not intent.changes_search() and not has_search:
        intent.action = "clarify"
        intent.transition = "clarification"
    return intent
