import langextract as lx


EXAMPLES = [

    lx.data.ExampleData(

        text="Find waiter jobs in Dubai",

        extractions=[

            lx.data.Extraction(
                extraction_class="job_title",
                extraction_text="waiter",
            ),

            lx.data.Extraction(
                extraction_class="location",
                extraction_text="Dubai",
            ),

        ]

    ),

    lx.data.ExampleData(

        text="Find Raj Bhatt profile",

        extractions=[

            lx.data.Extraction(
                extraction_class="person_name",
                extraction_text="Raj Bhatt",
            )

        ]

    ),

    lx.data.ExampleData(

        text="Marriott hotels",

        extractions=[

            lx.data.Extraction(
                extraction_class="company",
                extraction_text="Marriott",
            )

        ]

    ),

]