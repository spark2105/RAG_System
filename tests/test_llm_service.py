import unittest
from unittest.mock import patch

from llm_service import (
    GEMINI_MODEL_LABELS,
    GEMINI_MODEL_OPTIONS,
    GeminiEvaluationError,
    _generate_once,
    build_manual_hate_speech_prompt,
    evaluate_manual_hate_speech,
)


class LlmServiceTest(unittest.TestCase):
    def test_frontend_model_options_are_configured(self):
        self.assertEqual(
            GEMINI_MODEL_OPTIONS,
            (
                "gemini-3.6-flash",
                "gemini-3.5-flash-lite",
                "gemini-3.5-flash",
            ),
        )
        self.assertEqual(GEMINI_MODEL_LABELS["gemini-3.6-flash"], "Gemini 3.6 Flash")

    def test_prompt_contains_comment_and_omits_empty_optional_fields(self):
        prompt = build_manual_hate_speech_prompt(
            {
                "comment": "Das ist ein Kommentar.",
                "video_title": "",
                "article_text": None,
            }
        )
        self.assertIn("Kommentar: Das ist ein Kommentar.", prompt)
        self.assertIn("Hassrede umfasst Angriffe, Abwertung, Stereotype", prompt)
        self.assertIn("* Nutze alle vorhandenen Kontextdaten gemeinsam mit dem Kommentar.", prompt)
        self.assertIn("* Erkenne auch indirekte Anspielungen", prompt)
        self.assertIn("1–2 = schwache/unsichere Anspielung", prompt)
        self.assertIn("10 = expliziter Aufruf zu Gewalt, Vernichtung oder Verfolgung", prompt)
        self.assertIn("Antworte ausschließlich mit einer Ganzzahl von 0 bis 10.", prompt)
        self.assertNotIn("Video-Titel:", prompt)
        self.assertNotIn("Artikel-Text:", prompt)

    def test_prompt_includes_only_filled_optional_context_lines(self):
        prompt = build_manual_hate_speech_prompt(
            {
                "comment": "Okay",
                "video_title": "Videotitel",
                "video_description": "",
                "article_text": "Artikeltext",
                "comment_timestamp": "2026-08-10 12:00",
            }
        )
        self.assertIn("Video-Titel: Videotitel", prompt)
        self.assertIn("Artikel-Text: Artikeltext", prompt)
        self.assertIn("Kommentar-Zeitstempel: 2026-08-10 12:00", prompt)
        self.assertNotIn("Video-Beschreibung:", prompt)
        self.assertNotIn("Artikel-Titel:", prompt)

    def test_comment_is_required(self):
        with self.assertRaises(ValueError):
            build_manual_hate_speech_prompt({"comment": "  "})

    def test_api_key_is_required_for_evaluation(self):
        with self.assertRaises(ValueError):
            evaluate_manual_hate_speech({"comment": "Kommentar"})

    @patch("llm_service._generate_once", return_value="6")
    def test_requested_model_is_used(self, generate):
        result = evaluate_manual_hate_speech(
            {
                "comment": "Kommentar",
                "api_key": "test-key",
                "model": "gemini-2.5-flash-lite",
            }
        )
        self.assertEqual(result["model"], "gemini-2.5-flash-lite")
        self.assertEqual(generate.call_args.kwargs["model"], "gemini-2.5-flash-lite")

    @patch("llm_service.genai.Client")
    def test_gemini_sdk_request_uses_api_key(self, client_factory):
        client = client_factory.return_value
        client.models.generate_content.return_value.text = "8"

        output = _generate_once(
            "Prompt",
            api_key="secret-key",
            model="gemini-3.5-flash-lite",
        )

        self.assertEqual(output, "8")
        client_factory.assert_called_once_with(api_key="secret-key")
        client.models.generate_content.assert_called_once_with(
            model="gemini-3.5-flash-lite",
            contents="Prompt",
        )
        client.close.assert_called_once_with()

    @patch("llm_service._generate_once", side_effect=["invalid", "8"])
    def test_retries_once_and_returns_score_label(self, generate):
        result = evaluate_manual_hate_speech({"comment": "Kommentar", "api_key": "test-key"})
        self.assertEqual(result["result"], 8)
        self.assertEqual(result["label"], "Hassrede-Score: 8/10")
        self.assertEqual(result["llm_output"], "8")
        self.assertIn("Kommentar: Kommentar", result["prompt"])
        self.assertEqual(
            [entry["status"] for entry in result["evaluation_log"]],
            ["OK", "OK", "Fehler", "OK", "OK", "OK"],
        )
        self.assertTrue(all(entry["duration_ms"] >= 0 for entry in result["evaluation_log"]))
        self.assertEqual(generate.call_count, 2)

    @patch("llm_service._generate_once", side_effect=[ValueError("first"), ValueError("second")])
    def test_raises_after_second_failed_attempt(self, generate):
        with self.assertRaises(GeminiEvaluationError) as raised:
            evaluate_manual_hate_speech({"comment": "Kommentar", "api_key": "test-key"})
        self.assertEqual(generate.call_count, 2)
        self.assertEqual(len(raised.exception.evaluation_log), 3)
        self.assertTrue(all(entry["duration_ms"] >= 0 for entry in raised.exception.evaluation_log))


if __name__ == "__main__":
    unittest.main()
