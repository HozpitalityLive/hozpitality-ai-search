"""Authoritative map of the Hozpitality search_documents schema.

Derived ONLY from the PostgreSQL -> MongoDB migration scripts
(`migrations script/migrate_*.py`). Every path below is written by those
scripts; nothing here is guessed. If a field is not listed, the migration does
not produce it.

Module       _id prefix     PostgreSQL source                       Notes
-----------  -------------  --------------------------------------  -----------------------------------------
job          job:<id>       base_job (+ job_levels, job_role,       live & not deleted only; no URL, slug only
                            departments, industries, employment_type, job_type,
                            base_currency, base_salaryrange, companies/user_accounts)
professional professional:  professionals + user_accounts            no search_keywords/aliases; no URL
company      company:<id>   companies + user_accounts (+ industries,  supplier = company.is_supplier
                            supplier_category)                        (industry.context == "supplier" OR
                                                                       any supplier category)
product      product:<id>   marketplace_product (+ productcategory,   categories = marketplace product categories;
                            countries, seller user_accounts)          links.website/youtube are EXTERNAL links
article      article:<id>   base_article (+ base_category, author     location = countries only (no city)
                            user_accounts, base_article_location)
event        event:<id>     base_event (+ companies, countries)       website/payment_link are EXTERNAL
award        award:<id>     base_awards (+ base_awardcategory)        links.detail = award_detail_url (only real
                                                                       page URL in the whole corpus); location is
                                                                       a STRING, country is top-level
faq          faq:<id>       base_faq                                  question/answer, no title, no URL, no location

Concepts that are NOT entities (no documents of their own):
  supplier            -> company with company.is_supplier == True
  supplier category   -> company.supplier_categories[].name   (table supplier_category)
  industry            -> {job.industries, company.industries, professional.industries}[] with
                         {id, name, context}; context "supplier" marks supplier industries
  product category    -> product.categories[].name            (marketplace_productcategory)
  article category    -> article.category.name                (base_category)
  award category      -> award.categories[].name              (base_awardcategory)
  job level / role / department -> job.levels / job.roles / job.departments [] {id, name}
"""

from __future__ import annotations

from dataclasses import dataclass, field

ENTITY_TYPES = (
    "job",
    "professional",
    "company",
    "product",
    "article",
    "event",
    "award",
    "faq",
)

_COUNTRY_KEYS = ("name", "ac_name", "code", "country_code")


def _country(prefix: str) -> tuple[str, ...]:
    return tuple(f"{prefix}.{key}" for key in _COUNTRY_KEYS)


@dataclass(frozen=True)
class EntitySchema:
    entity: str
    source_table: str
    title: tuple[str, ...]
    description: tuple[str, ...]
    city: tuple[str, ...] = ()
    country: tuple[str, ...] = ()
    # Extra text fields for the regex fallback when $text is unavailable.
    search_fields: tuple[str, ...] = ()
    # Structured filter key -> document paths (arrays are traversed).
    filters: dict[str, tuple[str, ...]] = field(default_factory=dict)
    company_name: tuple[str, ...] = ()
    company_id: str | None = None
    company_slug: str | None = None
    category: tuple[str, ...] = ()
    created: tuple[str, ...] = ()
    # Real page URLs present in the record (only awards have one).
    record_url: tuple[str, ...] = ()
    # External links present in the record: (path, label).
    external_urls: tuple[tuple[str, str], ...] = ()
    images: tuple[str, ...] = ()
    live: tuple[str, ...] = ("is_live",)
    labels: tuple[str, str] = ("result", "results")


BASE_SEARCH_FIELDS = ("title", "search_aliases", "search_keywords")

SCHEMAS: dict[str, EntitySchema] = {
    "job": EntitySchema(
        entity="job",
        source_table="base_job",
        title=("title",),
        description=("summary", "job.description"),
        city=("location.city",),
        country=_country("location.country"),
        search_fields=("job.roles.name", "job.tags"),
        filters={
            "level": ("job.levels.name",),
            "role": ("job.roles.name",),
            "department": ("job.departments.name",),
            "industry": ("job.industries.name",),
            "employment_type": ("job.employment_type.name",),
            "job_type": ("job.job_type.name",),
            "featured": ("metadata.is_featured",),
            "status": ("metadata.job_status",),
        },
        company_name=("company.name",),
        company_id="company.id",
        company_slug="company.slug",
        category=("job.roles.name", "job.departments.name"),
        created=("metadata.created_at",),
        external_urls=(("metadata.spider_url", "Original listing"),),
        images=("media.avatar", "company.profile_url"),
        live=("is_live", "metadata.is_live"),
        labels=("job", "jobs"),
    ),
    "professional": EntitySchema(
        entity="professional",
        source_table="professionals",
        title=("title", "user.name"),
        description=("professional.resume_title", "professional.job_role.name"),
        city=("location.city",),
        country=_country("location.country"),
        search_fields=(
            "professional.job_role.name",
            "professional.resume_title",
            "professional.skills.name",
            "professional.department.name",
        ),
        filters={
            "level": ("professional.job_level.name",),
            "role": ("professional.job_role.name",),
            "department": ("professional.department.name",),
            "industry": ("professional.industries.name",),
            "skill": ("professional.skills.name",),
            "verified": ("metadata.verified",),
            "featured": ("metadata.is_featured",),
            "currently_working": ("metadata.is_working",),
        },
        company_name=(
            "professional.current_company.name",
            "professional.current_company_text",
        ),
        company_id="professional.current_company.id",
        category=("professional.job_role.name", "professional.department.name"),
        created=("created_at",),
        images=("profile_image",),
        labels=("professional", "professionals"),
    ),
    "company": EntitySchema(
        entity="company",
        source_table="companies",
        title=("title", "company.name"),
        description=("summary",),
        city=("location.city",),
        country=_country("location.country"),
        search_fields=("company.industries.name", "company.supplier_categories.name"),
        filters={
            "industry": ("company.industries.name",),
            "industry_context": ("company.industries.context",),
            "is_supplier": ("company.is_supplier",),
            "supplier_category": ("company.supplier_categories.name",),
            "verified": ("metadata.verified",),
            "featured": ("metadata.is_featured",),
            "company_size": ("metadata.no_of_employees",),
        },
        category=("company.supplier_categories.name", "company.industries.name"),
        created=("created_at",),
        external_urls=(("company.website", "Company website"),),
        images=("profile_image",),
        labels=("company", "companies"),
    ),
    "product": EntitySchema(
        entity="product",
        source_table="marketplace_product",
        title=("title",),
        description=("description",),
        city=(
            "location.prime_city",
            "location.current_location",
            "location.other_location",
        ),
        country=_country("location.country")
        + _country("location.available_in_countries"),
        search_fields=("keywords", "categories.name"),
        filters={
            "category": ("categories.name",),
            "product_type": ("product_type",),
            "condition": ("condition",),
            "featured": ("flags.is_featured",),
            "status": ("status",),
        },
        company_name=("seller.company_name", "seller.name"),
        company_id="seller.id",
        company_slug="seller.slug",
        category=("categories.name",),
        created=("dates.created_at",),
        external_urls=(
            ("links.website", "Product website"),
            ("links.youtube", "Video"),
        ),
        images=("media.main_image.path",),
        labels=("product", "products"),
    ),
    "article": EntitySchema(
        entity="article",
        source_table="base_article",
        title=("title",),
        description=("sub_title", "content.text"),
        city=(),
        country=_country("location.countries"),
        search_fields=("sub_title", "category.name"),
        filters={
            "category": ("category.name",),
            "featured": ("metadata.is_featured",),
            "status": ("metadata.status",),
        },
        company_name=("author.name",),
        company_id="author.id",
        company_slug="author.slug",
        category=("category.name",),
        created=("metadata.created_at",),
        external_urls=(("media.youtube_link", "Video"),),
        images=("media.thumbnail_url",),
        labels=("article", "articles"),
    ),
    "event": EntitySchema(
        entity="event",
        source_table="base_event",
        title=("title",),
        description=("description",),
        city=("location.city",),
        country=_country("location.country"),
        filters={
            "event_type": ("event_type",),
            "featured": ("flags.is_feature",),
            "status": ("status",),
        },
        company_name=("company.name",),
        company_id="company.id",
        company_slug="company.slug",
        created=("start_datetime", "created_at"),
        external_urls=(("website", "Event website"), ("payment_link", "Registration")),
        images=("media.banner",),
        labels=("event", "events"),
    ),
    "award": EntitySchema(
        entity="award",
        source_table="base_awards",
        title=("title", "short_title"),
        description=("subtitle", "description"),
        city=("location",),
        country=_country("country"),
        search_fields=("short_title", "subtitle", "categories.name"),
        filters={
            "category": ("categories.name",),
            "year": ("year",),
            "status": ("status",),
        },
        category=("categories.name",),
        created=("dates.created_at", "award_date"),
        record_url=("links.detail",),
        external_urls=(
            ("links.nomination", "Nominate"),
            ("links.vote_now", "Vote"),
            ("links.winners", "Winners"),
        ),
        images=("media.image", "media.avatar"),
        live=("is_active",),
        labels=("award", "awards"),
    ),
    "faq": EntitySchema(
        entity="faq",
        source_table="base_faq",
        title=("question",),
        description=("answer",),
        search_fields=("question", "answer"),
        filters={"status": ("status",)},
        created=("dates.created_at",),
        live=(),
        labels=("FAQ", "FAQs"),
    ),
}


def schema(entity: str | None) -> EntitySchema | None:
    return SCHEMAS.get(entity or "")


def filter_paths(entity: str | None, key: str) -> tuple[str, ...]:
    """Paths for a filter key; across all entities when entity is unknown."""
    if entity in SCHEMAS:
        return SCHEMAS[entity].filters.get(key, ())
    paths: list[str] = []
    for item in SCHEMAS.values():
        for path in item.filters.get(key, ()):
            if path not in paths:
                paths.append(path)
    return tuple(paths)


def filter_applies(entity: str | None, key: str) -> bool:
    """Whether a filter is meaningful for an entity (filter lifetime rules)."""
    if key in {"accommodation", "experience", "salary_min", "salary_currency"}:
        # Evidence/text based job & professional filters (no structured field).
        return (
            entity in {"job"}
            if key != "experience"
            else entity in {"job", "professional"}
        )
    if entity not in SCHEMAS:
        return True
    return key in SCHEMAS[entity].filters


def location_paths(entity: str | None, kind: str) -> tuple[str, ...]:
    if entity in SCHEMAS:
        item = SCHEMAS[entity]
        return item.city if kind == "city" else item.country
    paths: list[str] = []
    for item in SCHEMAS.values():
        for path in item.city if kind == "city" else item.country:
            if path not in paths:
                paths.append(path)
    return tuple(paths)


def search_fields(entity: str | None) -> tuple[str, ...]:
    if entity in SCHEMAS:
        extra = SCHEMAS[entity].search_fields
    else:
        extra = tuple(
            dict.fromkeys(p for s in SCHEMAS.values() for p in s.search_fields)
        )
    return tuple(dict.fromkeys(BASE_SEARCH_FIELDS + extra))


def labels(entity: str | None) -> tuple[str, str]:
    item = SCHEMAS.get(entity or "")
    return item.labels if item else ("result", "results")


# Facets: concepts that live inside documents rather than as documents.
FACETS: dict[str, tuple[str, str, dict]] = {
    # name: (entity, path, extra match)
    "supplier_category": (
        "company",
        "company.supplier_categories.name",
        {"company.is_supplier": True},
    ),
    "company_industry": ("company", "company.industries.name", {}),
    "supplier_industry": (
        "company",
        "company.industries.name",
        {"company.industries.context": "supplier"},
    ),
    "product_category": ("product", "categories.name", {}),
    "article_category": ("article", "category.name", {}),
    "award_category": ("award", "categories.name", {}),
    "job_role": ("job", "job.roles.name", {}),
    "job_department": ("job", "job.departments.name", {}),
}

FACET_LABELS = {
    "supplier_category": "supplier categories",
    "company_industry": "company industries",
    "supplier_industry": "supplier industries",
    "product_category": "product categories",
    "article_category": "article categories",
    "award_category": "award categories",
    "job_role": "job roles",
    "job_department": "job departments",
}
