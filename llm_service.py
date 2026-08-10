from __future__ import annotations

import os
from collections.abc import Mapping
from time import perf_counter
from typing import Any

from google import genai


DEFAULT_GEMINI_MODEL = "gemini-3.5-flash-lite"
GEMINI_MODEL_OPTIONS = (
    "gemini-3.6-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.5-flash",
)
GEMINI_MODEL_LABELS = {
    "gemini-3.6-flash": "Gemini 3.6 Flash",
    "gemini-3.5-flash-lite": "Gemini 3.5 Flash Lite",
    "gemini-3.5-flash": "Gemini 3.5 Flash",
}
AUTOMATIC_LLM_QUERY_MODES = (
    "Baseline1",
    "Baseline2",
    "Rag1",
    "Rag2",
    "BestCase1",
    "BestCase2",
    "WorstCase1",
    "WorstCase2",
)
AUTOMATIC_LLM_INPUT_COLUMNS = (
    "Comment_ID",
    "Comment_Text",
    "Video_Title",
    "Video_Description",
    "Video_timestamp",
    "Man_Picked_Article_Title",
    "Man_Picked_Article_Description",
    "Man_Picked_Article_Text",
)
AUTOMATIC_LLM_OUTPUT_COLUMNS = (
    *AUTOMATIC_LLM_INPUT_COLUMNS,
    "Mode",
    "RAG_Article_Title",
    "RAG_Article_Description",
    "RAG_Article_Text",
    "Hate_Speech_Evaluation",
    "Analysis_Duration",
)
AUTOMATIC_LLM_MODE_FIELDS = {
    "Baseline1": ("Comment_Text",),
    "Baseline2": ("Comment_Text", "Video_Title", "Video_Description", "Video_timestamp"),
    "Rag1": ("Comment_Text", "RAG_Article_Title", "RAG_Article_Description", "RAG_Article_Text"),
    "Rag2": (
        "Comment_Text",
        "Video_Title",
        "Video_Description",
        "Video_timestamp",
        "RAG_Article_Title",
        "RAG_Article_Description",
        "RAG_Article_Text",
    ),
    "BestCase1": (
        "Comment_Text",
        "Man_Picked_Article_Title",
        "Man_Picked_Article_Description",
        "Man_Picked_Article_Text",
    ),
    "BestCase2": (
        "Comment_Text",
        "Video_Title",
        "Video_Description",
        "Video_timestamp",
        "Man_Picked_Article_Title",
        "Man_Picked_Article_Description",
        "Man_Picked_Article_Text",
    ),
    "WorstCase1": (
        "Comment_Text",
        "Man_Picked_Article_Title",
        "Man_Picked_Article_Description",
        "Man_Picked_Article_Text",
    ),
    "WorstCase2": (
        "Comment_Text",
        "Video_Title",
        "Video_Description",
        "Video_timestamp",
        "Man_Picked_Article_Title",
        "Man_Picked_Article_Description",
        "Man_Picked_Article_Text",
    ),
}
MANUAL_OPTIONAL_CONTEXT_FIELDS = (
    ("Video-Titel", "video_title"),
    ("Video-Beschreibung", "video_description"),
    ("Video-Upload-Zeit", "video_upload_time"),
    ("Artikel-Titel", "article_title"),
    ("Artikel-Beschreibung", "article_description"),
    ("Artikel-Text", "article_text"),
    ("Kommentar-Zeitstempel", "comment_timestamp"),
)


def automatic_llm_required_fields(mode: str) -> tuple[str, ...]:
    normalized_mode = str(mode).strip()
    try:
        return AUTOMATIC_LLM_MODE_FIELDS[normalized_mode]
    except KeyError as exc:
        raise ValueError(f"Unknown automatic LLM query mode: {mode}") from exc


def build_automatic_llm_values(
    row: Mapping[str, Any],
    *,
    mode: str,
    api_key: str,
    model: str | None = None,
    rag_article: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Map one automatic-evaluation row to the shared manual prompt payload."""
    fields = automatic_llm_required_fields(mode)
    article = rag_article or {}
    source_map = {
        "Comment_Text": (row, "comment"),
        "Video_Title": (row, "video_title"),
        "Video_Description": (row, "video_description"),
        "Video_timestamp": (row, "video_upload_time"),
        "RAG_Article_Title": (article, "article_title"),
        "RAG_Article_Description": (article, "article_description"),
        "RAG_Article_Text": (article, "article_text"),
        "Man_Picked_Article_Title": (row, "article_title"),
        "Man_Picked_Article_Description": (row, "article_description"),
        "Man_Picked_Article_Text": (row, "article_text"),
    }
    values: dict[str, Any] = {"api_key": api_key, "model": model}
    for field in fields:
        source, target = source_map[field]
        values[target] = source.get(field if source is row else {
            "article_title": "Article_Title",
            "article_description": "Article_Description",
            "article_text": "Article_Text",
        }[target])
    return values


class GeminiEvaluationError(RuntimeError):
    """Raised when Gemini cannot produce a valid manual evaluation."""

    def __init__(self, message: str, evaluation_log: list[dict[str, Any]] | None = None) -> None:
        super().__init__(message)
        self.evaluation_log = evaluation_log or []


def gemini_model(requested_model: str | None = None) -> str:
    return (requested_model or os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL)).strip()


def build_manual_hate_speech_prompt(values: Mapping[str, Any]) -> str:
    comment = str(values.get("comment") or "").strip()
    if not comment:
        raise ValueError("Comment is required.")

    optional_context_lines = []
    for label, key in MANUAL_OPTIONAL_CONTEXT_FIELDS:
        value = str(values.get(key) or "").strip()
        if value:
            optional_context_lines.append(f"{label}: {value}")
    optional_context = "\n".join(optional_context_lines)

    prompt = f"""Bewerte den Kommentar auf identitätsbezogene Hassrede oder diskriminierende Sprache.

Hassrede umfasst Angriffe, Abwertung, Stereotype oder Diskriminierung aufgrund von z. B. Ethnizität, Nationalität, Herkunft, Religion, Hautfarbe, Geschlecht, sexueller Orientierung, Geschlechtsidentität oder Behinderung.

WICHTIG:

* Nutze alle vorhandenen Kontextdaten gemeinsam mit dem Kommentar.
* Die betroffene Gruppe muss NICHT im Kommentar genannt sein. Sie kann sich nur aus Video oder Artikel ergeben.
* Eine stereotype Aussage über eine durch den Kontext erkennbare Identitätsgruppe zählt als identitätsbezogene Abwertung.
* Erkenne auch indirekte Anspielungen, Klischees, Codes, Ironie, Umgangssprache und offensichtliche Tippfehler.
* Erfinde keinen nicht vorhandenen Kontext.
- Gib als Antwort nur den ermittelten Skalawert als Ganzzahl aus

Skala:
0 = kein identitätsbezogener Bezug
1–2 = schwache/unsichere Anspielung
3–4 = erkennbare, mehrdeutige Abwertung/Stereotypisierung
5–6 = deutliche identitätsbezogene Abwertung/Stereotypisierung
7–8 = starke Beleidigung, Entmenschlichung oder Diskriminierung
9 = extreme Hassrede/Gewaltbefürwortung
10 = expliziter Aufruf zu Gewalt, Vernichtung oder Verfolgung

Antworte ausschließlich mit einer Ganzzahl von 0 bis 10.

Kommentar: {comment}"""

    if optional_context:
        prompt += f"\n{optional_context}"
    return prompt


def _parse_hate_speech_result(response_text: str) -> int:
    value = response_text.strip()
    try:
        score = int(value)
    except ValueError:
        score = -1
    if 0 <= score <= 10 and str(score) == value:
        return score
    raise ValueError(f"Gemini returned an invalid result: {response_text!r}")


def _generate_once(prompt: str, *, api_key: str, model: str) -> str:
    client = genai.Client(api_key=api_key)
    try:
        response = client.models.generate_content(model=model, contents=prompt)
        raw_response = str(response.text or "").strip()
    finally:
        client.close()
    if not raw_response:
        raise ValueError("Gemini response did not contain a text output.")
    return raw_response


def evaluate_manual_hate_speech(values: Mapping[str, Any]) -> dict[str, Any]:
    api_key = str(values.get("api_key") or "").strip()
    if not api_key:
        raise ValueError("Gemini API key is required.")

    evaluation_log: list[dict[str, Any]] = []

    def log_step(
        step: str,
        started_at: float,
        *,
        status: str = "OK",
        detail: str | None = None,
    ) -> None:
        entry: dict[str, Any] = {
            "step": step,
            "duration_ms": round((perf_counter() - started_at) * 1000, 2),
            "status": status,
        }
        if detail:
            entry["detail"] = detail
        evaluation_log.append(entry)

    prompt_started_at = perf_counter()
    prompt = build_manual_hate_speech_prompt(values)
    log_step("Prompt erstellen", prompt_started_at)
    model = gemini_model(str(values.get("model") or "").strip() or None)
    last_error: Exception | None = None

    for attempt in range(2):
        attempt_number = attempt + 1
        request_started_at = perf_counter()
        try:
            raw_response = _generate_once(prompt, api_key=api_key, model=model)
        except Exception as exc:
            log_step(
                f"Gemini-Anfrage (Versuch {attempt_number})",
                request_started_at,
                status="Fehler",
                detail=str(exc),
            )
            last_error = exc
            continue

        log_step(f"Gemini-Anfrage (Versuch {attempt_number})", request_started_at)

        parse_started_at = perf_counter()
        try:
            result = _parse_hate_speech_result(raw_response)
        except ValueError as exc:
            log_step(
                f"LLM-Output prüfen (Versuch {attempt_number})",
                parse_started_at,
                status="Fehler",
                detail=str(exc),
            )
            last_error = exc
            continue

        log_step(f"LLM-Output prüfen (Versuch {attempt_number})", parse_started_at)

        result_started_at = perf_counter()
        result_payload: dict[str, Any] = {
            "result": result,
            "label": f"Hassrede-Score: {result}/10",
            "model": model,
            "prompt": prompt,
            "llm_output": raw_response,
        }
        log_step("Ergebnis aufbereiten", result_started_at)
        result_payload["evaluation_log"] = evaluation_log
        return result_payload

    raise GeminiEvaluationError(
        f"Gemini evaluation failed after two attempts: {last_error}",
        evaluation_log,
    ) from last_error
