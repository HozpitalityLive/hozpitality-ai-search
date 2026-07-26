import httpx
from fastapi import APIRouter, HTTPException

router = APIRouter()

OLLAMA_URL = "http://ollama:11434/api/generate"
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

    if content_type.lower() == "tagline":

        prompt = f"""
You are an expert LinkedIn and resume writer.

Write ONE professional profile tagline.

Rules:
- Maximum 12 words.
- One sentence.
- No quotation marks.
- ATS friendly.
- Mention profession naturally.
- Make it impactful.
- Do not invent information.
- Return ONLY the tagline.

Candidate Profile:

{profile_text}
"""

    elif content_type.lower() == "about":

        prompt = f"""
You are an expert resume and LinkedIn writer.

Write a professional About Me section.

Rules:
- Around 5 short sentences.
- 80-120 words.
- Professional and engaging.
- ATS friendly.
- Mention industry, department, role and strengths.
- Highlight leadership, customer service and technical skills when available.
- Never invent experience.
- No bullet points.
- Return ONLY the description.

Candidate Profile:

{profile_text}
"""

    else:
        raise ValueError("content_type must be 'tagline' or 'about'")

    return await _generate(prompt)


@router.post("/generate-profile-content")
async def generate_profile(data: dict):

    try:
        result = await generate_profile_content(
            profile=data.get("profile", {}),
            content_type=data.get("content_type", ""),
        )

        return {
            "success": True,
            "content": result,
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))