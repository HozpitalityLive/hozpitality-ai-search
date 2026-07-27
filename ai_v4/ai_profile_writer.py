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

Requirements:
- Length between 100 and 200 characters.
- One sentence only.
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
- 400-450 characters.
- Professional tone.
- No bullet points.
- Return ONLY the About Us section.

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
- any text before or after the statement

If you output anything except the statement, your answer is incorrect.

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
- 400-450 characters.
- Professional tone.
- No bullet points.
- Return ONLY the About section.

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
- any text before or after the statement

If you output anything except the statement, your answer is incorrect.

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