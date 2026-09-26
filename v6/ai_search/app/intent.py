"""Deterministic query classification (runs before any LLM).

Decides, with an explainable reason:
  * information/FAQ question vs. request for records
  * which entity the user asked for (explicit nouns win over role words)
  * schema concepts that are not entities (supplier, supplier category,
    supplier industry) and how they map onto search_documents
  * facet requests ("list supplier categories")

Schema mapping (see schema_map.py, derived from the migration scripts):
  suppliers            -> entity company, filter is_supplier = True
  supplier industry    -> entity company, filter industry_context = "supplier"
  supplier category(s) -> facet over company.supplier_categories.name
  products/marketplace -> entity product
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

ENTITY_NOUNS: dict[str, str] = {
    # explicit, strong module nouns
    r"jobs?|vacanc(?:y|ies)|job\s+openings?": "job",
    r"professionals?|candidates?": "professional",
    r"compan(?:y|ies)|employers?|businesses|organi[sz]ations?": "company",
    r"products?|marketplace|listings?\s+for\s+sale": "product",
    r"articles?|news|blogs?|blog\s+posts?|stories": "article",
    r"events?|conferences?|exhibitions?|summits?|expos?": "event",
    r"awards?": "award",
    r"faqs?|frequently\s+asked(?:\s+questions)?": "faq",
}
# Generic nouns: set the entity only when nothing stronger exists and never
# switch an existing conversation entity.
WEAK_NOUNS: dict[str, str] = {
    r"positions?|roles?|openings?|careers?|employment": "job",
    r"people|persons?|talents?|experts?": "professional",
}

_NOUN_RE = [
    (re.compile(rf"\b(?:{p})\b"), e, "strong") for p, e in ENTITY_NOUNS.items()
] + [(re.compile(rf"\b(?:{p})\b"), e, "weak") for p, e in WEAK_NOUNS.items()]

SUPPLIER_CATEGORY_RE = re.compile(
    r"\bsupplier\s+categor(?:y|ies)\b|\bcategor(?:y|ies)\s+of\s+suppliers?\b|\bsupplier\s+types?\b"
)
SUPPLIER_INDUSTRY_RE = re.compile(
    r"\bsupplier\s+industr(?:y|ies)\b|\bindustr(?:y|ies)\s+(?:of|for)\s+suppliers?\b"
)
SUPPLIER_RE = re.compile(r"\b(?:suppliers?|vendors?)\b")
FACET_LIST_RE = re.compile(
    r"^(?:please\s+)?(?:(?:list|show|find|get|give)(?:\s+me)?(?:\s+(?:all|the))?|what\s+are\s+the|which)\b"
    r"|\b(?:list\s+of|all)\b"
)
OTHER_FACETS = [
    (
        re.compile(
            r"\bproduct\s+categor(?:y|ies)\b|\bmarketplace\s+categor(?:y|ies)\b"
        ),
        "product_category",
    ),
    (
        re.compile(r"\barticle\s+categor(?:y|ies)\b|\bnews\s+categor(?:y|ies)\b"),
        "article_category",
    ),
    (re.compile(r"\baward\s+categor(?:y|ies)\b"), "award_category"),
    (
        re.compile(
            r"\b(?:company|companies)\s+industr(?:y|ies)\b|\bindustries\s+of\s+companies\b"
        ),
        "company_industry",
    ),
]
COMPANIES_HIRING_RE = re.compile(
    r"\b(?:compan(?:y|ies)|employers?|hotels?)\s+(?:that\s+are\s+|who\s+are\s+|currently\s+)?"
    r"(?:hiring|recruiting|looking\s+for)\s+(?P<role>[a-z][a-z &/-]{1,40}?)"
    r"(?=\s+in\b|\s+at\b|\s+near\b|$|[?.!,])"
)

# ---------------------------------------------------------------------------
# Information (FAQ) vs records
# ---------------------------------------------------------------------------

_ENTITY_PLURALS = (
    r"jobs|vacancies|openings|positions|roles|professionals|candidates|companies|employers|"
    r"suppliers|vendors|products|articles|news|events|awards|faqs|hotels|restaurants"
)
# Requests for records: "which chef jobs are available", "what jobs are there",
# "any chef jobs in Dubai", "how many jobs ...".
RECORDS_QUESTION_RE = re.compile(
    rf"^(?:which|what)\b(?:\s+\w+){{0,4}}?\s+(?:{_ENTITY_PLURALS})\b"
    r"(?:\s+\w+){0,4}?\s*(?:are|is)?\s*(?:available|open|there|listed|hiring|posted|near|in|at|for|on)\b"
    rf"|^(?:are|is)\s+there\s+(?:any\s+)?(?:\w+\s+){{0,4}}?(?:{_ENTITY_PLURALS})\b"
    rf"|^how\s+many\s+(?:\w+\s+){{0,3}}?(?:{_ENTITY_PLURALS})\b"
    rf"|^(?:which|what)\s+(?:{_ENTITY_PLURALS})\b"
)
SEARCH_COMMAND_RE = re.compile(
    r"^(?:please\s+)?(?:find|search|show|list|get|give|browse|look\s+up|recommend|suggest)\b"
    r"|^(?:i\s+(?:need|want|am\s+looking\s+for)|i'm\s+looking\s+for|looking\s+for)\b"
    r"|^any\s+\w+"
)
INFO_TERMS = (
    r"process|procedure|policy|policies|requirements?|required|steps?|fees?|charges?|refunds?|"
    r"subscription|membership|plans?|packages?|account|profile|password|log\s*in|login|sign\s*up|"
    r"register|registration|verify|verification|support|contact|privacy|terms|meaning|mean|"
    r"difference|benefits?\s+of|purpose|work|works|hozpitality|platform|website|app|feature|"
    r"credits?|payment|invoice|cv|resume|documents?"
)
ACTION_VERBS = (
    r"apply|register|sign\s*up|create|delete|remove|reset|change|update|edit|upload|post|publish|"
    r"contact|cancel|pay|buy|subscribe|verify|log\s*in|login|use|find\s+out|get\s+verified|"
    r"become|add|hire|advertise|list|nominate|vote|attend|book|withdraw|track"
)
INFO_QUESTION_RES = [
    re.compile(
        rf"^how\s+(?:do|can|could|should|would|to|does|did|is|are|will|long|much)\b(?!\s+many\b)"
    ),
    re.compile(
        rf"^(?:(?:give|show|tell)\s+(?:me\s+)?(?:the\s+)?(?:steps?|process|procedure)\b|(?:steps?|process|procedure)\s+to\b)"
    ),
    re.compile(
        rf"^what\s+(?:is|are|does|do|was|were|'s)\s+(?:the\s+|a\s+|an\s+|my\s+|your\s+)?(?:\w+\s+){{0,3}}?(?:{INFO_TERMS})\b"
    ),
    re.compile(
        r"^what\s+(?:is|'s)\s+hozpitality\b|^what\s+(?:is|are)\s+(?:a\s+)?(?:pro|premium|featured|verified)\b"
    ),
    re.compile(r"^why\b"),
    re.compile(
        rf"^(?:can|may|should|must|do|does)\s+(?:i|we|you|my|companies|professionals)\b.*\b(?:{ACTION_VERBS})\b"
    ),
    re.compile(
        r"^(?:is\s+it\s+possible|do\s+i\s+need|am\s+i\s+(?:able|allowed|eligible)|where\s+(?:can|do|should)\s+i|"
        r"who\s+(?:can|do|should)\s+i|when\s+(?:can|will|do|should)\s+i|what\s+happens\s+(?:if|when))\b"
    ),
    re.compile(
        r"^i\s+(?:can't|cannot|can\s+not|am\s+unable|am\s+not\s+able|forgot|lost|didn't\s+receive|did\s+not\s+receive|"
        r"have\s+a\s+problem|have\s+an\s+issue)\b"
    ),
    re.compile(
        rf"^(?:help|explain|tell\s+me\s+about)\b.*\b(?:{INFO_TERMS}|{ACTION_VERBS})\b"
    ),
    re.compile(
        r"\b(?:faqs?|frequently\s+asked|help\s+cent(?:er|re)|customer\s+support)\b"
    ),
]


# One label per INFO_QUESTION_RES entry (same order), for explainability.
INFO_QUESTION_LABELS = (
    "how-to question",
    "asks what a process/policy/requirement is",
    "asks what Hozpitality or a membership is",
    "why question",
    "asks whether/how they can do an action",
    "possibility/eligibility question",
    "reports a problem",
    "asks for help or an explanation",
    "mentions FAQ/support",
)


def _norm(text: str) -> str:
    text = (text or "").replace("’", "'").casefold()
    return " ".join(text.split()).strip(" ?!.")


def is_information_question(text: str) -> tuple[bool, str]:
    """True when the user asks for information/instructions, not records."""
    low = _norm(text)
    if not low:
        return False, "empty"
    if RECORDS_QUESTION_RE.search(low):
        return False, "asks which/what records exist"
    # Imperative help requests such as "give me steps to apply for a job"
    # are information questions even though they begin with a search-command
    # verb like "give" or "show".
    if re.search(r"^(?:give|show|tell)\s+(?:me\s+)?(?:the\s+)?(?:steps?|process|procedure)\b", low):
        return True, "information question: asks for steps/process"
    if SEARCH_COMMAND_RE.search(low):
        # "Find FAQs about passwords" is a search of the FAQ module (explicit
        # noun), handled by entity classification, not a question.
        return False, "search command"
    for label, pattern in zip(INFO_QUESTION_LABELS, INFO_QUESTION_RES):
        if pattern.search(low):
            return True, f"information question: {label}"
    return False, "no information pattern"


# ---------------------------------------------------------------------------
# Entity classification
# ---------------------------------------------------------------------------


@dataclass
class Classification:
    kind: str = "search"  # search | faq | facet
    entity: str | None = None
    strength: str | None = None  # explicit | concept | intent | weak | role | faq
    term: str | None = None
    reason: str = ""
    filters: dict = field(default_factory=dict)  # schema filters implied by concepts
    facet: str | None = None
    related: str | None = None  # "companies_hiring"
    related_role: str | None = None
    consumed: list[str] = field(
        default_factory=list
    )  # phrases that must not become keywords

    def as_dict(self) -> dict:
        return {
            "kind": self.kind,
            "entity": self.entity,
            "strength": self.strength,
            "term": self.term,
            "reason": self.reason,
            "filters": self.filters,
            "facet": self.facet,
            "related": self.related,
        }


def classify(text: str) -> Classification:
    low = _norm(text)
    info, why = is_information_question(low)
    if info:
        return Classification(kind="faq", entity="faq", strength="faq", reason=why)

    # Facets: concepts stored inside documents ("list supplier categories").
    category = SUPPLIER_CATEGORY_RE.search(low)
    if category:
        tail = low[category.end() :].strip(" :-")
        tail = re.sub(r"^(?:is|called|named|of|=)\s+", "", tail)
        value = (
            ""
            if not tail or re.match(r"^(?:in|near|at|from|for|with)\b", tail)
            else tail
        )
        if value:
            # "suppliers in supplier category kitchen equipment"
            return Classification(
                kind="search",
                entity="company",
                strength="concept",
                term=category.group(0),
                filters={"is_supplier": True, "supplier_category": value},
                reason="supplier category filter on company.supplier_categories.name",
                consumed=[category.group(0), value],
            )
        return Classification(
            kind="facet",
            entity="company",
            strength="concept",
            term=category.group(0),
            facet="supplier_category",
            filters={"is_supplier": True},
            reason="supplier categories are company.supplier_categories (table supplier_category)",
            consumed=[category.group(0)],
        )
    for pattern, facet in OTHER_FACETS:
        match = pattern.search(low)
        if match and FACET_LIST_RE.search(low):
            from .schema_map import FACETS

            return Classification(
                kind="facet",
                entity=FACETS[facet][0],
                strength="concept",
                term=match.group(0),
                facet=facet,
                reason=f"{facet} is a field inside {FACETS[facet][0]} documents",
                consumed=[match.group(0)],
            )

    # "companies hiring chefs" -> companies behind matching job records.
    hiring = COMPANIES_HIRING_RE.search(low)
    if hiring:
        return Classification(
            kind="search",
            entity="company",
            strength="explicit",
            term="companies hiring",
            related="companies_hiring",
            related_role=hiring.group("role").strip(),
            reason="companies linked to matching jobs (job.company.id -> company:<id>)",
            consumed=[hiring.group(0)],
        )

    # Explicit nouns, first occurrence wins ("professionals who are chefs").
    found: list[tuple[int, str, str, str]] = []
    for regex, entity, strength in _NOUN_RE:
        match = regex.search(low)
        if match:
            found.append((match.start(), entity, strength, match.group(0)))
    strong = sorted(f for f in found if f[2] == "strong")

    supplier_industry = SUPPLIER_INDUSTRY_RE.search(low)
    supplier = SUPPLIER_RE.search(low)

    if supplier_industry:
        return Classification(
            kind="search",
            entity="company",
            strength="concept",
            term=supplier_industry.group(0),
            filters={"industry_context": "supplier"},
            reason="supplier industry = company.industries[].context == 'supplier'",
            consumed=[supplier_industry.group(0)],
        )

    if strong:
        _, entity, _, term = strong[0]
        # "products from suppliers" -> product; "suppliers" alone -> company.
        cls = Classification(
            kind="search",
            entity=entity,
            strength="explicit",
            term=term,
            reason=f"explicit module noun '{term}'",
        )
        if supplier and entity == "company":
            cls.filters = {"is_supplier": True}
            cls.reason += "; supplier = company.is_supplier"
            cls.consumed = [supplier.group(0)]
        return cls

    if supplier:
        return Classification(
            kind="search",
            entity="company",
            strength="concept",
            term=supplier.group(0),
            filters={"is_supplier": True},
            reason="suppliers are companies with company.is_supplier (migrate_companies.py)",
            consumed=[supplier.group(0)],
        )

    weak = sorted(f for f in found if f[2] == "weak")
    if weak:
        _, entity, _, term = weak[0]
        return Classification(
            kind="search",
            entity=entity,
            strength="weak",
            term=term,
            reason=f"generic noun '{term}'",
        )

    intent = re.search(
        r"\b(?:hire|hiring|vacancy|vacancies|salary|paying|opening|openings|recruiting)\b",
        low,
    )
    if intent:
        return Classification(
            kind="search",
            entity="job",
            strength="intent",
            term=intent.group(0),
            reason=f"job intent word '{intent.group(0)}'",
        )
    return Classification(kind="search", reason="no explicit entity")
