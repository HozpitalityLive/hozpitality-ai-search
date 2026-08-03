from ai_v4.planner.extractors.langextract_extractor import LangExtractExtractor
import asyncio


async def main():

    extractor = LangExtractExtractor()

    queries = [

        "Find Raj Bhatt profile",
        "Show chef profiles",
        "Candidates with Opera PMS",
        "French speaking receptionist",

        "Waiter jobs in Dubai",
        "Housekeeping supervisor jobs",
        "Front Office Manager UAE",
        "Jobs paying AED 10000",

        "Marriott hotels",
        "Hilton companies",
        "Restaurants in Qatar",

        "Latest hospitality articles",

        "Hotel awards 2025",

        "Hospitality events in Dubai",

    ]

    for query in queries:

        print("=" * 100)
        print(query)

        entities = await extractor.extract(query)

        print(entities)


asyncio.run(main())