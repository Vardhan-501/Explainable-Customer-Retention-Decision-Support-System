import json
import os
from typing import Any

import requests

SUMMARY_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "risk_factors": {"type": "array", "items": {"type": "string"}},
        "recommended_next_steps": {"type": "array", "items": {"type": "string"}},
        "caveat": {"type": "string"},
    },
    "required": ["summary", "risk_factors", "recommended_next_steps", "caveat"],
    "additionalProperties": False,
}

DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_OLLAMA_MODEL = "qwen3.5:4b"
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"


def _setting(name: str, default: str | None = None) -> str | None:
    """Read a setting from the environment or Streamlit secrets.

    Streamlit Cloud secrets are not guaranteed to be exposed through
    ``os.environ`` in every runtime, so the app checks ``st.secrets`` as well.
    The import is intentionally lazy so this module remains usable in tests
    and from non-Streamlit scripts.
    """
    value = os.getenv(name)
    if value:
        return value.strip()

    try:
        import streamlit as st

        secret_value = st.secrets.get(name)
        if secret_value is not None:
            return str(secret_value).strip()
    except Exception:
        # Streamlit is optional when the module is imported by unit tests, and
        # missing/unconfigured secrets should behave like missing environment
        # variables rather than breaking the prediction app.
        pass

    return default


def ollama_url() -> str:
    return _setting("OLLAMA_URL", DEFAULT_OLLAMA_URL) or DEFAULT_OLLAMA_URL


def ollama_model() -> str:
    return _setting("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL) or DEFAULT_OLLAMA_MODEL


def openai_model() -> str:
    return _setting("OPENAI_MODEL", DEFAULT_OPENAI_MODEL) or DEFAULT_OPENAI_MODEL


def build_verified_facts(result: dict[str, Any], explanation_df, actions: list[dict[str, str]], threshold: float = 0.60) -> dict[str, Any]:
    risk_factors = explanation_df.loc[explanation_df["impact"] > 0, "display_feature"].head(3).tolist()
    next_steps = [item["action"] for item in actions[:3]]
    return {
        "risk_band": result["risk_band"],
        "probability": round(float(result["probability"]), 4),
        "threshold": threshold,
        "top_risk_factors": risk_factors,
        "recommended_actions": next_steps,
        "limitations": [
            "Model explanations describe associations, not causes.",
            "Prediction should be reviewed by a human.",
        ],
    }


def ollama_available() -> bool:
    try:
        response = requests.get(f"{ollama_url()}/api/tags", timeout=3)
        return response.ok
    except requests.RequestException:
        return False


def openai_available() -> bool:
    return bool(_setting("OPENAI_API_KEY"))


def summary_provider() -> str:
    """Return the active provider: ``ollama``, ``openai``, or ``fallback``."""
    if ollama_available():
        return "ollama"
    if openai_available():
        return "openai"
    return "fallback"


def ai_summary_available() -> bool:
    """Return whether the app can generate a summary.

    A deterministic local fallback is always available, so the UI should keep
    the Generate button enabled even when no external language model is set up.
    """
    return True


def _prompt(facts: dict[str, Any]) -> str:
    return (
        "You are a careful customer-retention analyst. Use only the verified facts below. "
        "Do not invent metrics, causes, discounts, or customer history. State that model "
        "explanations describe associations rather than causation. Return JSON with exactly "
        "the fields summary, risk_factors, recommended_next_steps, and caveat.\n\n"
        + json.dumps(facts)
    )


def _validate_summary(summary: dict[str, Any]) -> dict[str, Any]:
    required = {"summary", "risk_factors", "recommended_next_steps", "caveat"}
    if set(summary) != required:
        raise ValueError("The summary did not match the required JSON fields")
    if not isinstance(summary["summary"], str) or not isinstance(summary["caveat"], str):
        raise ValueError("Summary and caveat must be strings")
    if not isinstance(summary["risk_factors"], list) or not isinstance(summary["recommended_next_steps"], list):
        raise ValueError("Risk factors and next steps must be arrays")
    if not all(isinstance(item, str) for item in summary["risk_factors"] + summary["recommended_next_steps"]):
        raise ValueError("Risk factors and next steps must contain only strings")
    return summary


def _generate_with_ollama(facts: dict[str, Any], model: str | None = None) -> dict[str, Any]:
    response = requests.post(
        f"{ollama_url()}/api/chat",
        json={
            "model": model or ollama_model(),
            "messages": [{"role": "user", "content": _prompt(facts)}],
            "stream": False,
            "format": SUMMARY_SCHEMA,
            "options": {"temperature": 0},
        },
        timeout=120,
    )
    response.raise_for_status()
    return _validate_summary(json.loads(response.json()["message"]["content"]))


def _generate_with_openai(facts: dict[str, Any], model: str | None = None) -> dict[str, Any]:
    from openai import OpenAI

    api_key = _setting("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not configured")

    client_kwargs: dict[str, str] = {"api_key": api_key}
    base_url = _setting("OPENAI_BASE_URL")
    if base_url:
        client_kwargs["base_url"] = base_url

    response = OpenAI(**client_kwargs).chat.completions.create(
        model=model or openai_model(),
        messages=[
            {"role": "system", "content": "You are a careful customer-retention analyst. Output JSON only."},
            {"role": "user", "content": _prompt(facts)},
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "retention_summary", "strict": True, "schema": SUMMARY_SCHEMA},
        },
        max_completion_tokens=800,
    )
    content = response.choices[0].message.content
    if not content:
        raise ValueError("The OpenAI response did not contain summary content")
    return _validate_summary(json.loads(content))


def _generate_fallback(facts: dict[str, Any], provider_errors: list[str] | None = None) -> dict[str, Any]:
    """Create a useful summary without inventing facts or requiring a model."""
    probability = float(facts.get("probability", 0.0))
    risk_band = str(facts.get("risk_band", "Unknown"))
    probability_text = f"{probability:.1%}"
    risk_factors = [str(item) for item in facts.get("top_risk_factors", []) if item]
    recommended_actions = [str(item) for item in facts.get("recommended_actions", []) if item]

    if risk_factors:
        factor_text = ", ".join(risk_factors)
        summary = (
            f"The churn model estimates a {probability_text} churn probability, which is in the "
            f"{risk_band} risk band. The main model-associated risk signals are: {factor_text}."
        )
    else:
        summary = (
            f"The churn model estimates a {probability_text} churn probability, which is in the "
            f"{risk_band} risk band. No positive model-associated risk signals were returned."
        )

    if not recommended_actions:
        recommended_actions = ["Review the customer profile and confirm the appropriate next step with a human reviewer."]

    caveat = (
        "No external language-model provider is configured, so this summary was generated by the "
        "built-in verified-facts template. Model explanations describe associations, not causes, "
        "and the result should be reviewed by a human."
    )
    if provider_errors:
        caveat = (
            "An external language-model provider was unavailable, so the built-in verified-facts "
            "template was used instead. Model explanations describe associations, not causes, "
            "and the result should be reviewed by a human."
        )

    return _validate_summary(
        {
            "summary": summary,
            "risk_factors": risk_factors or ["No positive model-associated risk signals were returned."],
            "recommended_next_steps": recommended_actions,
            "caveat": caveat,
        }
    )


def generate_business_summary(facts: dict[str, Any]) -> dict[str, Any]:
    """Use Ollama first, OpenAI second, and a safe local fallback last."""
    provider_errors: list[str] = []

    if ollama_available():
        try:
            return _generate_with_ollama(facts)
        except Exception as error:
            provider_errors.append(f"ollama:{type(error).__name__}")

    if openai_available():
        try:
            return _generate_with_openai(facts)
        except Exception as error:
            provider_errors.append(f"openai:{type(error).__name__}")

    return _generate_fallback(facts, provider_errors)
