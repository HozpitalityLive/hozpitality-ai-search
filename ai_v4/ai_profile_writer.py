import httpx
from fastapi import APIRouter, HTTPException
from ai_v4.config.settings import settings
import logging

logger = logging.getLogger(__name__)

router = APIRouter()


OLLAMA_URL = f"{settings.OLLAMA_URL}/api/generate"
MODEL = "llama3-hoz:latest"


async def _generate(prompt: str) -> str:
    async with httpx.AsyncClient(timeout=90) as client:
        response = await client.post(
            OLLAMA_URL,
            json={
                "model": MODEL,
                "system": """
You are a profile content generator.

The provided profile is the ONLY source of truth.

Never use knowledge from training data.
Never reuse names, companies, employers, projects or examples.
Never complete missing information.
If information is missing, omit it.
Every statement must be directly supported by the profile.
""",
                "prompt": prompt,
                
                "stream": False,
                "keep_alive": 0,
                "options": {
                    "temperature": 0.2,
                    "top_p": 0.8,
                    "repeat_penalty": 1.2,
                    "num_predict": 220,
                    "num_ctx": 4096,
                },
            }
        )

        response.raise_for_status()

        return response.json()["response"].strip()


def build_profile(profile: dict) -> str:
    profile_type = (profile.get("profile_type") or "professional").lower()

    if profile_type == "company":
        fields = [
            ("Business Name", profile.get("business_name")),
            ("Created By", profile.get("created_by")),
            ("Designation", profile.get("designation")),
            ("Website Link", profile.get("website_url")),
            ("Industry", profile.get("industry")),
            ("Supplier Category", profile.get("supplier_category")),
            ("Country", profile.get("country")),
        ]
    else:
        full_name = " ".join(
            filter(
                None,
                [
                    profile.get("firstname"),
                    profile.get("lastname"),
                ],
            )
        )

        fields = [
            ("Hi, I am professional My Name", full_name),
            ("Industry", profile.get("industry")),
            ("Department", profile.get("department")),
            ("Role", profile.get("role")),
            ("Job Level", profile.get("job_level")),
            ("Currently Working in Company", profile.get("working_in_company")),
            ("Languages", profile.get("languages")),
            ("Country", profile.get("country")),
            ("Experience", profile.get("experience")),
            ("Skills", profile.get("skills")),
            ("Education", profile.get("education")),
            ("Achievements", profile.get("achievements")),
            ("Certifications", profile.get("certifications")),
        ]

    return "\n".join(
        f"{label}: {value}"
        for label, value in fields
        if value not in (None, "", [])
    )


async def generate_profile_content(
    profile: dict,
    content_type: str,
) -> str:

    profile_text = build_profile(profile)

    logger.info("Profile: %s", profile)

    
    profile_type = (profile.get("profile_type") or "professional").lower()

    if profile_type == "company":
        name = profile.get("business_name", "")
    else:
        name = " ".join(
            filter(
                None,
                [
                    profile.get("firstname"),
                    profile.get("lastname"),
                ],
            )
        )

    profile_type = profile.get("profile_type", "professional").lower()

    if content_type.lower() == "tagline":
        if profile_type == "company":
            prompt = f"""
You are an expert professional content writer.

Generate ONE company branding statement.

Company Name: {name if name else "Not Provided"}

IMPORTANT:
Use ONLY the information provided.

DO NOT invent:
- products
- services
- customers
- locations
- experience
- history
- expertise
- mission
- values
- certifications
- awards

If information is missing, omit it.

- Maximum 150 characters.
- One sentence.
- If the profile contains limited information, keep it concise.
- Never invent missing details.
- Rephrase the profile professionally.
- Make it attractive but factual.
- Do not exaggerate.
- No quotation marks.

OUTPUT FORMAT (MANDATORY)

Your response MUST contain ONLY the final text.

DO NOT include:
- introductions
- explanations
- notes
- markdown
- quotation marks
- labels
- "Here is..."
- "Certainly"
- "Let me know..."
- "Professional Branding Statement:"
- any text before or after the statement

If you output anything except the statement, your answer is incorrect.

Company Profile:

{profile_text}
"""
        else:
            prompt = f"""
You are an expert professional content writer.

Generate ONE professional branding statement.

Candidate Name: {name if name else "Not Provided"}

IMPORTANT:
The profile below is the ONLY source of truth.

DO NOT:
- invent job titles
- invent experience
- invent seniority
- invent skills
- invent achievements
- invent certifications
- invent employers
- invent industries
- infer missing information

If a field is missing, completely ignore it.

Requirements:
- Length between 100 and 200 characters.
- Write one attractive, natural sentence.
- Rephrase the available information professionally.
- Make it engaging without exaggeration.
- Never use buzzwords such as:
  Results-driven
  Passionate
  Dynamic
  Innovative
  Expert
  World-class
  Best-in-class
  Highly experienced
unless those facts are explicitly provided.
- Do not repeat the same word.
- No quotation marks.

OUTPUT FORMAT (MANDATORY)

Your response MUST contain ONLY the final text.

DO NOT include:
- introductions
- explanations
- notes
- markdown
- quotation marks
- labels
- "Here is..."
- "Certainly"
- "Let me know..."
- "Professional Branding Statement:"
- any text before or after the statement

If you output anything except the statement, your answer is incorrect.

Candidate Profile:

{profile_text}
"""

    elif content_type.lower() == "about":
        if profile_type == "company":
            prompt = f"""
You are an expert business content writer.

Your task is to write an "About Us" section for a company.

Company Name:
{name if name else "Not Provided"}

COMPANY PROFILE

{profile_text}

CRITICAL INSTRUCTIONS

The company profile above is the ONLY source of truth.

Every statement must be supported by the supplied information.

Never use information from:
- previous conversations
- training data
- memory
- assumptions
- common industry knowledge

DO NOT invent or guess:
- products
- services
- customers
- experience
- years in business
- history
- company size
- offices
- branches
- locations
- certifications
- awards
- partnerships
- expertise
- mission
- vision
- values
- technologies

If information is missing, omit it completely.

You MAY:
- professionally rewrite the provided information
- expand the wording without changing the meaning
- combine available facts into natural sentences
- improve readability
- use professional business language

Requirements

- Around 150–350 characters.
- 4–5 sentences.
- If the supplied profile contains limited information, produce a shorter response rather than inventing facts.
- Begin with the company name only if it exists.
- If no company name exists, do not use placeholders.
- Professional and trustworthy tone.
- No bullet points.
- No headings.
- No quotation marks.
- No markdown.
- No emojis.

OUTPUT FORMAT

Return ONLY the About Us section.

Do not include:
- Here is...
- About Us:
- Certainly
- Let me know...
- explanations
- notes
- markdown
- any text before or after the response.
"""
        else:
            prompt = f"""
You are an expert professional profile content writer.

Your task is to write a professional "About" section for a person's profile.

Candidate Name:
{name if name else "Not Provided"}

PROFILE DATA

{profile_text}

CRITICAL INSTRUCTIONS

The profile data above is the ONLY source of truth.

Every statement in your response must be directly supported by the profile.

Never use information from:
- previous conversations
- training examples
- memory
- assumptions
- common industry knowledge

DO NOT invent or guess:
- names
- employers
- companies
- job titles
- years of experience
- seniority
- achievements
- awards
- certifications
- projects
- responsibilities
- technical skills
- leadership
- customer service experience
- communication skills
- expertise
- specializations
- locations other than those provided

If any information is missing, completely omit it.

You MAY:
- professionally rewrite the provided information
- expand the wording without changing the meaning
- connect available facts into natural sentences
- improve readability
- use professional language

Requirements

- 400–450 characters.
- 3–5 sentences.
- Mention the candidate's name only if it exists.
- If no name exists, do not use placeholders.
- Professional, natural and engaging tone.
- No first-person language.
- No bullet points.
- No headings.
- No quotation marks.
- No markdown.
- No emojis.

OUTPUT FORMAT

Return ONLY the About section.

Do not include:
- Here is...
- About:
- Certainly
- Let me know...
- explanations
- notes
- markdown
- any extra text before or after the response.
"""


    return await _generate(prompt)


@router.post("/generate-profile-content")
async def generate_profile(data: dict):

    try:
        result = await generate_profile_content(
            profile=data.get("profile", {}),
            content_type=data.get("content_type", ""),
        )
        
        logger.info("Incoming request: %s", data)
        

        return {
            "success": True,
            "content": result,
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))