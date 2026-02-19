import sktk


def test_top_level_imports():
    new_symbols = [
        "MockKernel",
        "LLMScenario",
        "PIIFilter",
        "PromptInjectionFilter",
        "ContentSafetyFilter",
        "TokenBudgetFilter",
        "SQLiteHistory",
        "InMemoryHistory",
        "InMemoryBlackboard",
        "InMemoryKnowledgeBackend",
        "TextSource",
        "fixed_size_chunker",
    ]
    for name in new_symbols:
        assert hasattr(sktk, name), f"sktk.{name} not found"
        assert name in sktk.__all__, f"{name} not in sktk.__all__"
