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

        for index, item in enumerate(documents, start=1):
            doc = item["document"]
            output.append(

f"""
Result {index}

Title:
{doc.get("title","")}

Category:
{doc.get("category_text","")}

Location:
{doc.get("location_text","")}

User:
{doc.get("user_name","")}

Content:
{doc.get("content","")}

URL:
{doc.get("slug","")}
"""
            )

        return "\n".join(output)