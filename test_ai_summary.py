import src.ai_summary as ai_summary


FACTS = {
    "risk_band": "High",
    "probability": 0.8123,
    "threshold": 0.60,
    "top_risk_factors": ["Contract", "tenure"],
    "recommended_actions": ["Review contract options"],
    "limitations": ["Model explanations describe associations, not causes."],
}


def test_default_ollama_model_matches_installed_model():
    assert ai_summary.ollama_model() == "qwen3.5:4b"


def test_fallback_summary_is_available_without_external_provider(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(ai_summary, "ollama_available", lambda: False)

    assert ai_summary.ai_summary_available() is True
    assert ai_summary.summary_provider() == "fallback"

    summary = ai_summary.generate_business_summary(FACTS)

    assert set(summary) == {"summary", "risk_factors", "recommended_next_steps", "caveat"}
    assert "81.2%" in summary["summary"]
    assert summary["risk_factors"] == ["Contract", "tenure"]
    assert summary["recommended_next_steps"] == ["Review contract options"]
    assert "built-in verified-facts template" in summary["caveat"]


def test_provider_failure_returns_fallback(monkeypatch):
    monkeypatch.setattr(ai_summary, "ollama_available", lambda: True)
    monkeypatch.setattr(ai_summary, "openai_available", lambda: False)

    def fail_ollama(_facts):
        raise RuntimeError("model is not installed")

    monkeypatch.setattr(ai_summary, "_generate_with_ollama", fail_ollama)

    summary = ai_summary.generate_business_summary(FACTS)

    assert "built-in verified-facts template" in summary["caveat"]
    assert summary["recommended_next_steps"] == ["Review contract options"]
