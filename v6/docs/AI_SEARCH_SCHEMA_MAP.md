# Hozpitality search_documents — schema & entity map

Source of truth: the migration scripts in `migrations script/migrate_*.py`
(PostgreSQL `hozpitality` → MongoDB `mongoAdmin.search_documents`). The code
equivalent is [`ai_search/app/schema_map.py`](../ai_search/app/schema_map.py).
Nothing below is inferred; if a field is not listed, the migration does not
write it.

## Modules (entity types)

| entity_type | `_id` | PostgreSQL source | Title | Description | Slug | Page URL in record |
|---|---|---|---|---|---|---|
| job | `job:<id>` | `base_job` (only `is_live AND NOT is_deleted`) | `title` (job_title) | `summary`, `job.description` | `slug` | — |
| professional | `professional:<id>` | `professionals` + `user_accounts` (active) | `title` = first + last name | `professional.resume_title` | `slug` | — |
| company | `company:<id>` | `companies` + `user_accounts` (active, user_type=company) | `title` = company name | `summary` (about_us) | `slug` | — |
| product | `product:<id>` | `marketplace_product` | `title` | `description` | `slug` | — |
| article | `article:<id>` | `base_article` (all statuses) | `title` | `sub_title`, `content.text` | `slug` | — |
| event | `event:<id>` | `base_event` | `title` | `description` (details) | `slug` | — |
| award | `award:<id>` | `base_awards` | `title`, `short_title` | `subtitle`, `description` | `slug` | **`links.detail`** (award_detail_url) |
| faq | `faq:<id>` | `base_faq` | `question` (may start with "1. ") | `answer` | — | — |

Every document also has `entity_type`, `source.object_id` (PostgreSQL id),
`ai_search_text` (the `$text`-indexed field), and, for all but professionals,
`search_keywords` / `search_aliases`.

## Location

| entity | city | country |
|---|---|---|
| job | `location.city` | `location.country.{name, ac_name, code, country_code}` |
| professional | `location.city` | `location.country.{name, code, country_code}` |
| company | `location.city` | `location.country.{…}` |
| product | `location.prime_city`, `current_location`, `other_location` | `location.country.{…}`, `location.available_in_countries[]` |
| article | — (none) | `location.countries[].{name, code}` |
| event | `location.city` | `location.country.{…}` |
| award | `location` (**a string**) | `country.{…}` (top level) |
| faq | — | — |

A city search on articles uses the city's country (there is no city field).
FAQs ignore location.

## Relationships

| From | Field | To |
|---|---|---|
| job | `company.{id, name, slug}` (`base_job.company_id` → `user_accounts.id`) | company `company:<id>` (`companies.useraccount_ptr_id`) |
| professional | `professional.current_company.{id, name}`, `current_company_text` | company |
| product | `seller.{id, name, user_type, company_name, slug}` | user account (company or person) |
| article | `author.{id, type, name, slug}` | user account |
| event | `company.{id, name, slug}` | company |
| award | `categories[]` | award categories (`base_awardcategory`) |

## Classification fields (not entities)

| Concept | Where it lives | Notes |
|---|---|---|
| **Supplier** | `company.is_supplier` | `True` when the company has an industry with `context == "supplier"` **or** any supplier category. Supplier companies also get aliases like "supplier", "hospitality vendor". **There is no supplier module.** |
| Supplier category | `company.supplier_categories[].name` | table `supplier_category` via `user_accounts_supplier_category` |
| Industry | `company.industries[] {id, name, context}`, `job.industries[]`, `professional.industries[]` | `context` distinguishes supplier industries. A company whose industry *name* contains "Supplier" is not a supplier unless its context is `supplier` or it has supplier categories. |
| Product category | `categories[].name` (product) | `marketplace_productcategory` — unrelated to supplier categories |
| Article category | `category.name` | `base_category` |
| Award category | `categories[].name` (award) | `base_awardcategory` |
| Job level / role / department | `job.levels[]`, `job.roles[]`, `job.departments[]` `{id, name}` | professionals: `professional.job_level`, `job_role`, `department` |
| Employment type / job type | `job.employment_type.name`, `job.job_type.name` | |
| Salary | `job.salary.{description, range.name, currency.code}` | text only; no numeric min/max |
| Accommodation | — | **no structured field**; only the job's own description text |

## Liveness, dates, status

| entity | live | date | status |
|---|---|---|---|
| job | `is_live`, `metadata.is_live` | `metadata.created_at` (ISO string) | `metadata.job_status` |
| professional | `is_live` (= active) | `created_at` | — |
| company | `is_live` | `created_at` | — |
| product | `is_live` (status live and not expired) | `dates.created_at` | `status` |
| article | `is_live` (status publish) | `metadata.created_at` (ISO string) | `metadata.status` |
| event | `is_live` (Upcoming/Ongoing and not ended) | `start_datetime`, `created_at` | `status` |
| award | `is_active` | `dates.created_at`, `award_date` | `status` active/inactive |
| faq | — (`status: "active"`) | `dates.created_at` | `status` |

## External links (never used as the record's page URL)

`job.metadata.spider_url` (original listing of imported jobs),
`company.company.website`, `product.links.website` / `links.youtube`,
`event.website` / `payment_link`, `article.media.youtube_link`,
`award.links.{nomination, vote_now, winners, …}`.

## Page URLs

Only awards store a page URL (`links.detail`). All other modules store a
slug, and the site's route patterns are not in the migration. Operators
configure them with `PUBLIC_URL_TEMPLATES` (placeholders `{slug}`, `{id}`); a
module without a template returns `url: null` — no URL is ever guessed.
