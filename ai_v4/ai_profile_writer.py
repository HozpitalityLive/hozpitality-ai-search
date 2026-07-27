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
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.6,
                    "num_predict": 220,
                },
            },
        )

        response.raise_for_status()

        return response.json()["response"].strip()


def build_profile(profile: dict) -> str:
    fields = [
        ("Industry", profile.get("industry")),
        ("Department", profile.get("department")),
        ("Role", profile.get("role")),
        ("Job Level", profile.get("job_level")),
        ("Current Company", profile.get("company")),
        ("Experience", profile.get("experience")),
        ("Skills", profile.get("skills")),
        ("Languages", profile.get("languages")),
        ("Country", profile.get("country")),
        ("Education", profile.get("education")),
        ("Achievements", profile.get("achievements")),
        ("Certifications", profile.get("certifications")),
    ]

    return "\n".join(
        f"{label}: {value}"
        for label, value in fields
        if value
    )


async def generate_profile_content(
    profile: dict,
    content_type: str,
) -> str:

    profile_text = build_profile(profile)

    logger.info("Profile: %s", profile)

    name = (
        profile.get("name")
        or profile.get("full_name")
        or profile.get("company_name")
        or profile.get("company")
        or ""
    )

    profile_type = profile.get("profile_type", "professional").lower()

    if content_type.lower() == "tagline":
        if profile_type == "company":
            prompt = f"""
You are an expert Content writer AI.

Create ONE company tagline.

Company Name: {name if name else "Not Provided"}

STRICT RULES:
- Use ONLY information in the profile.
- NEVER invent products.
- NEVER invent services.
- NEVER invent customers.
- NEVER invent company history.
- NEVER invent expertise.
- NEVER invent values.
- If there is insufficient information, create a simple tagline using only the industry or specialization.
- Maximum 8 words.
- No quotation marks.
- No emojis.
- Return ONLY the tagline.

Company Profile:

{profile_text}
"""
        else:
            prompt = f"""
You are an expert Content writer AI.

Your task is to create Tagline for profile using ONLY the information provided.

Candidate Name: {name if name else "Not Provided"}

STRICT RULES:
- Use ONLY fields present in the profile.
- NEVER infer or guess any missing information.
- NEVER invent:
  - job title
  - experience
  - seniority
  - certifications
  - achievements
  - industry
  - technical skills
  - leadership
  - expertise
  - employer
- If Role is missing, DO NOT create one.
- If Experience is missing, DO NOT mention years or seniority.
- If Skills are missing, DO NOT mention technologies or expertise.
- If Company is missing, DO NOT mention any company.
- Maximum 10 words.
- No quotation marks.
- No emojis.
- No marketing language.
- Return ONLY the headline.

Preferred order:
1. Role
2. Department
3. Industry
4. Skills
5. Country

Candidate Profile:

{profile_text}
"""

    elif content_type.lower() == "about":
        if profile_type == "company":
            prompt = f"""
You are an expert Content writer AI.

Write an About Us section.

Company Name: {name if name else "Not Provided"}

STRICT RULES:
- Begin with the company name ONLY if available.
- Use ONLY information in the profile.
- NEVER invent:
  - products
  - services
  - experience
  - customers
  - locations
  - company history
  - mission
  - vision
  - expertise
  - awards
  - certifications
- If information is missing, simply omit it.
- Never fill gaps with assumptions.
- Write only from available facts.
- 80-120 words.
- Professional tone.
- No bullet points.
- Return ONLY the About Us section.

Company Profile:

{profile_text}
"""
        else:
            prompt = f"""
You are an expert Content writer AI.

Write a professional About section.

Candidate Name: {name if name else "Not Provided"}

STRICT RULES:
- Mention the candidate's name ONLY if available.
- Use ONLY information present in the profile.
- NEVER invent:
  - experience
  - years
  - achievements
  - certifications
  - leadership
  - technical skills
  - communication skills
  - customer service
  - employer
  - projects
  - responsibilities
- If Role is missing, do not create one.
- If Experience is missing, do not mention experience.
- If Skills are missing, do not describe expertise.
- If Education is missing, omit education.
- If Achievements are missing, omit achievements.
- Do not add filler sentences.
- 80-120 words.
- Professional tone.
- No bullet points.
- Return ONLY the About section.

Candidate Profile:

{profile_text}
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