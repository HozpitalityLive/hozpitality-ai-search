from ai_v4.config.logger import logger


class SearchMerger:

    def merge(
        self,
        elastic_results: list,
        postgres_results: list
    ) -> list:

        logger.info("Merging search results")

        merged = []
        seen = set()
      
        for item in elastic_results:

            doc = item["document"]
            unique_key = (
                doc.get("content_type_id"),
                doc.get("object_id")
            )

            if unique_key in seen:
                continue

            seen.add(unique_key)
            merged.append(item)
 
        for item in postgres_results:

            doc = item["document"]
            unique_key = (
                doc.get("content_type_id"),
                doc.get("object_id")
            )

            if unique_key in seen:
                continue

            seen.add(unique_key)
            merged.append(item)

        return merged