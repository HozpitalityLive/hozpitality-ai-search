from ai_search.app.typo import correct_tokens


def test_typo_correction():
    corrected, changes = correct_tokens(
        ["excutive", "chef"],
        ["executive", "chef", "manager"],
        threshold=80,
    )
    assert corrected == ["executive", "chef"]
    assert changes[0]["from"] == "excutive"
    assert changes[0]["to"] == "executive"


def test_plural_role_is_not_corrected_to_unrelated_term():
    corrected, changes = correct_tokens(
        ["chefs"],
        ["chef", "chefs-kitchen", "chefs"],
        threshold=80,
    )
    assert corrected == ["chefs"]
    assert changes == []
