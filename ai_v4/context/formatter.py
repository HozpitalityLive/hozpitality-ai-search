from ai_v4.config.logger import logger


class ContextFormatter:

    def format_documents(
        self,
        documents: list
    ) -> str:

        logger.info(
            f"Formatting {len(documents)} documents"
        )

        output = []

        for index, item in enumerate(documents, 1):

            doc = item["document"]

            category = (
                doc.get("category")
                or doc.get("category_text")
                or "general"
            ).lower()

            title = doc.get("title", "")
            company = doc.get("user_name") or doc.get("company") or ""
            location = doc.get("location") or doc.get("location_text") or ""
            salary = doc.get("salary") or ""
            content = (doc.get("content") or "")[:500]
            slug = doc.get("slug") or ""

            if category == "job":

                output.append(
f"""[JOB {index}]
Title: {title}
Company: {company}
Location: {location}
{"Salary: " + salary if salary else ""}
Description: {content}
URL: {slug}
"""
                )

            elif category == "company":

                output.append(
f"""[COMPANY {index}]
Name: {title}
Location: {location}
Description: {content}
URL: {slug}
"""
                )

            elif category == "professional":

                output.append(
f"""[PROFESSIONAL {index}]
Name: {title}
Location: {location}
Profile: {content}
URL: {slug}
"""
                )

            elif category == "article":

                output.append(
f"""[ARTICLE {index}]
Title: {title}
Author: {company}
Summary: {content}
URL: {slug}
"""
                )

            elif category == "product":

                output.append(
f"""[PRODUCT {index}]
Name: {title}
Seller: {company}
Location: {location}
Description: {content}
URL: {slug}
"""
                )

            elif category == "event":

                output.append(
f"""[EVENT {index}]
Title: {title}
Location: {location}
Details: {content}
URL: {slug}
"""
                )

            elif category == "award":

                output.append(
f"""[AWARD {index}]
Title: {title}
Description: {content}
URL: {slug}
"""
                )

            else:

                output.append(
f"""[RESULT {index}]
Title: {title}
Category: {category}
Location: {location}
Description: {content}
URL: {slug}
"""
                )

        return "\n".join(output)