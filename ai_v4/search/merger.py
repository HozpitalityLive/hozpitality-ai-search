from ai_v4.config.logger import logger


class SearchMerger:

    def merge(
        self,
        elastic_results: list,
        postgres_results: list
    ) -> list:

        logger.info("=" * 80)
        logger.info("MERGING SEARCH RESULTS")
        logger.info(f"Elastic Results : {len(elastic_results)}")
        logger.info(f"Postgres Results: {len(postgres_results)}")
        logger.info("=" * 80)

        merged = []
        seen = set()

        logger.info("Processing Elastic Results")

        for index, item in enumerate(elastic_results, start=1):

            doc = item["document"]

            unique_key = (
                doc.get("content_type_id"),
                doc.get("object_id")
            )

            logger.info(
                f"[ES {index}] "
                f"title={doc.get('title')} | "
                f"id={doc.get('id')} | "
                f"content_type_id={doc.get('content_type_id')} | "
                f"object_id={doc.get('object_id')} | "
                f"key={unique_key}"
            )

            if unique_key in seen:

                logger.warning(
                    f"Duplicate Elastic Result Skipped -> {unique_key}"
                )

                continue

            seen.add(unique_key)
            merged.append(item)

        logger.info("Processing Postgres Results")

        for index, item in enumerate(postgres_results, start=1):

            doc = item["document"]

            unique_key = (
                doc.get("content_type_id"),
                doc.get("object_id")
            )

            logger.info(
                f"[PG {index}] "
                f"title={doc.get('title')} | "
                f"id={doc.get('id')} | "
                f"content_type_id={doc.get('content_type_id')} | "
                f"object_id={doc.get('object_id')} | "
                f"key={unique_key}"
            )

            if unique_key in seen:

                logger.warning(
                    f"Duplicate Postgres Result Skipped -> {unique_key}"
                )

                continue

            seen.add(unique_key)
            merged.append(item)

        logger.info("=" * 80)
        logger.info(f"FINAL MERGED RESULTS : {len(merged)}")
        logger.info("=" * 80)

        return merged