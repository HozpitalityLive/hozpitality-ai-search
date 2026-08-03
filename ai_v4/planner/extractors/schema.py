from dataclasses import dataclass


@dataclass(frozen=True)
class SearchEntity:

    name: str
    description: str


PERSON = SearchEntity(
    name="person_name",
    description="Professional, candidate or person's full name",
)

COMPANY = SearchEntity(
    name="company",
    description="Company, hotel, employer, restaurant or organization",
)

LOCATION = SearchEntity(
    name="location",
    description="City, state, country or geographic location",
)

JOB_TITLE = SearchEntity(
    name="job_title",
    description="Job title, designation or position",
)

DEPARTMENT = SearchEntity(
    name="department",
    description="Department or functional area",
)

SKILL = SearchEntity(
    name="skill",
    description="Professional skill",
)

TECHNOLOGY = SearchEntity(
    name="technology",
    description="Technology, software or platform",
)

SALARY = SearchEntity(
    name="salary",
    description="Salary, pay or compensation",
)

EXPERIENCE = SearchEntity(
    name="experience",
    description="Experience duration",
)

AWARD = SearchEntity(
    name="award",
    description="Award or recognition",
)

EVENT = SearchEntity(
    name="event",
    description="Event name",
)

ARTICLE = SearchEntity(
    name="article",
    description="Article or publication title",
)

LANGUAGE = SearchEntity(
    name="language",
    description="Spoken language",
)

NATIONALITY = SearchEntity(
    name="nationality",
    description="Nationality or citizenship",
)

VISA = SearchEntity(
    name="visa",
    description="Visa or work authorization",
)

EMPLOYMENT_TYPE = SearchEntity(
    name="employment_type",
    description="Employment type such as Full-time, Part-time, Contract, Internship",
)


SEARCH_SCHEMA = [
    PERSON,
    COMPANY,
    LOCATION,
    JOB_TITLE,
    DEPARTMENT,
    SKILL,
    TECHNOLOGY,
    SALARY,
    EXPERIENCE,
    AWARD,
    EVENT,
    ARTICLE,
    LANGUAGE,
    NATIONALITY,
    VISA,
    EMPLOYMENT_TYPE,
]


SEARCH_ENTITY_NAMES = [entity.name for entity in SEARCH_SCHEMA]