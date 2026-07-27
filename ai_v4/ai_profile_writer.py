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
You are an expert branding copywriter.

Create ONE memorable company tagline.

Company Name: {name}

Requirements:
- Maximum 8 words.
- Professional and memorable.
- Reflect the company's industry and value proposition.
- Make every response unique.
- Never use generic slogans such as:
  - Excellence in Everything
  - Quality You Can Trust
  - Your Trusted Partner
  - We Deliver Excellence
- Never invent products or services.
- No quotation marks.
- Return ONLY the tagline.

Company Profile:

{profile_text}
"""
        else:
            prompt = f"""
You are an expert LinkedIn branding strategist.

Create ONE unique professional headline.

Candidate Name: {name}

Requirements:
- Maximum 12 words.
- One sentence only.
- ATS-friendly.
- Reflect the candidate's profession, expertise and industry.
- Include seniority when available.
- Make every response unique by varying wording and sentence structure.
- Never repeat common phrases like "Results-driven", "Passionate professional", or "Experienced professional".
- Never invent information.
- No quotation marks.
- Return ONLY the headline.

Candidate Profile:

{profile_text}
"""

    elif content_type.lower() == "about":
        if profile_type == "company":
            prompt = f"""
You are an expert business copywriter.

Write an About Us section.

Company Name: {name}

Requirements:
- 90-140 words.
- Begin with the company name.
- Explain what the company does.
- Mention industry, products, services, expertise and customer focus only when provided.
- Use professional business language.
- Make every response unique.
- Avoid clichés and exaggerated marketing claims.
- Never invent information.
- No bullet points.
- Return ONLY the About Us section.

Company Profile:

{profile_text}
"""
        else:
            prompt = f"""
You are an expert LinkedIn profile writer.

Write a professional About section.

Candidate Name: {name}

Requirements:
- 90-130 words.
- Begin naturally by mentioning the candidate's name.
- Highlight industry, role, expertise and strengths.
- Mention leadership, communication, technical or customer-facing skills only when provided.
- Keep it ATS-friendly.
- Make each response unique in wording and structure.
- Never invent achievements or experience.
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