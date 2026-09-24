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
