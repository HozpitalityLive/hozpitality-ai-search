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
            ("Business Name", profile.get("companyname")),
            # ("Created By", profile.get("createdBy")),
            # ("Designation", profile.get("currentDesignation")),
            ("Industry", profile.get("industry")),
            ("Supplier Category", profile.get("supplier_category")),
            ("Country", profile.get("country")),
            ("City", profile.get("city")),
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
            ("Name", full_name),
            ("Industry", profile.get("industry")),
            ("Department", profile.get("department")),
            ("Role", profile.get("role")),
            ("City", profile.get("city")),
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
        name = profile.get("companyname", "")
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
Write ONE factual company tagline using ONLY the Company Profile below.

COMPANY PROFILE:
{profile_text}

STRICT RULES:
- Use only facts explicitly present in the profile.
- Do not invent or infer anything.
- Do not add services, products, expertise, customers, solutions, experience, quality, innovation, leadership, mission, or vision unless explicitly provided.
- Include the company name if available.
- Include industry only if available.
- Include supplier category only if available.
- Include city only if available.
- Include country only if available.
- Rewrite the available facts professionally.
- One sentence only.
- Maximum 150 characters.

OUTPUT RULE:
Return ONLY the tagline text.

DO NOT return:
- "Here's your professional tagline:"
- "Here is your tagline:"
- "Professional tagline:"
- "Tagline:"
- "Here is..."
- "Certainly..."
- explanations
- quotes
- markdown
- bullet points
- any introductory or closing text

The first character of your response must be the first character of the tagline.
The last character must be the last character of the tagline.
"""
        else:
            prompt = f"""
Write ONE factual professional tagline using ONLY the Professional Profile below.

PROFESSIONAL PROFILE:
{profile_text}

TAGLINE FORMAT:

Hi, I am a [Industry] professional working as [Role] based in [Location].

STRICT RULES:

- The tagline MUST start with "Hi, I am a".
- Do NOT mention the person's name.
- Use the Industry field when available.
- Use the Role field when available.
- Use City and Country as the location when available.
- If both City and Country are available, format the location as "City, Country".
- If only City is available, use only the City.
- If only Country is available, use only the Country.
- Do not invent or infer any information.
- Do not use information from training data, memory, previous conversations, or general knowledge.
- Do not invent employers, companies, job titles, responsibilities, expertise, achievements, certifications, projects, skills, experience, or locations.
- Only use information explicitly provided in the profile.
- Do not mention the person's name even if it is available.
- Do not mention department, company, skills, experience, education, achievements, or certifications unless they are required to complete the format.
- Keep the wording natural and professional.
- One sentence only.
- Maximum 150 characters.

MISSING INFORMATION:

- If Industry is missing, say "Hi, I am a professional".
- If Role is missing, omit "working as [Role]".
- If Location is missing, omit "based in [Location]".
- Never create placeholder text.
- Never guess missing information.

EXAMPLES:

If Industry, Role, City and Country are available:
Hi, I am a Facility Management professional working as Facility Manager based in Vasai-Virar, India.

If Industry, Role and City are available:
Hi, I am a Facility Management professional working as Facility Manager based in Vasai-Virar.

If Industry and Location are available:
Hi, I am a Facility Management professional based in Vasai-Virar, India.

If Industry and Role are available:
Hi, I am a Facility Management professional working as Facility Manager.

If only Industry is available:
Hi, I am a Facility Management professional.

OUTPUT RULE:

Return ONLY the tagline text.

DO NOT return:

- "Here's your professional tagline:"
- "Here is your professional tagline:"
- "Here's your tagline:"
- "Here is your tagline:"
- "Professional tagline:"
- "Tagline:"
- "Here is..."
- "Certainly..."
- explanations
- notes
- quotes
- markdown
- bullet points
- introductory text
- closing text

The first character of your response must be "H".
"""

            
    elif content_type.lower() == "about":
        if profile_type == "company":
            prompt = f"""
You are writing an About Us section.

The profile below is the ONLY source of truth.

Company Profile

{profile_text}

Rules

Every sentence MUST be supported by the profile.

Never invent:

- services
- products
- expertise
- experience
- customers
- technologies
- solutions
- mission
- vision
- values
- offices
- history
- achievements
- awards
- certifications

If information is missing,
leave it out.

Write naturally.

2-4 short sentences.

Maximum 450 characters.

Begin with the company name if available.

Return ONLY the About Us text.

No headings.

No markdown.

No quotes.

No explanations.
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