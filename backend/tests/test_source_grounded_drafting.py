import sys
import unittest
from types import SimpleNamespace, ModuleType
from unittest.mock import Mock, patch

sys.path.insert(0, "backend")
from app.ai import (draft_procedure, ollama, PROCEDURE_RESPONSE_SCHEMA, TITLE_REPAIR_SCHEMA,
                    CONSISTENCY_RESPONSE_SCHEMA, LocalModelError, EvidenceMismatchError)
from app.transcription import transcribe_video
from app.source_evidence import load_source_excerpts
from app.db import Base
from app.models import Source, KnowledgeChunk
from app.video_sop import _select_prompt_visuals
from sqlalchemy import create_engine
from sqlalchemy.orm import Session


class DraftGroundingTests(unittest.TestCase):
    def setUp(self):
        self.excerpts = [{
            "text": "ドリルの電源を切り、チャックとドリルビットに損傷がないか確認します。ドリルの電源を切り、チャックとドリルビットに損傷がないか確認します。操作前に周囲を確認します。",
            "source_ref": "DrillMachine.mp4 · 00:12",
            "timestamp": "00:12",
            "source_id": 21,
        }]

    def _soldering_result(self, title="Inspect the solder joint", *, steps=None):
        return {
            "evidence_quality": "limited", "insufficient_evidence": False, "reason": "Only the shown inspection is covered.",
            "title": title, "description": "Inspect the solder joint after the iron is removed.",
            "tools": [], "warnings": [],
            "steps": steps if steps is not None else [{
                "instruction": "Inspect the solder joint after the iron is removed.",
                "timestamp": "00:12", "source_ref": "soldering.mp4 · 00:12", "evidence_type": "source_supported",
            }],
        }

    def _soldering_evidence(self):
        return [{"text": "Inspect the solder joint after the iron is removed.",
            "source_ref": "soldering.mp4 · 00:12", "timestamp": "00:12", "evidence_type": "source_supported"}]

    @patch("app.ai.requests.post")
    def test_ollama_receives_the_procedure_json_schema(self, post):
        response = post.return_value
        response.json.return_value = {"response": "{}"}
        self.assertEqual(ollama("Return procedure JSON.", json_mode=True,
            format_schema=PROCEDURE_RESPONSE_SCHEMA), "{}")
        payload = post.call_args.kwargs["json"]
        self.assertIs(payload["format"], PROCEDURE_RESPONSE_SCHEMA)
        self.assertFalse(payload["stream"])

    @patch("app.ai.requests.post")
    def test_ollama_falls_back_once_when_server_rejects_json_schema(self, post):
        import requests
        first, second = Mock(), Mock()
        post.side_effect = [first, second]
        first.raise_for_status.side_effect = requests.HTTPError(
            "schema unsupported", response=SimpleNamespace(status_code=400))
        second.json.return_value = {"response": "{}"}
        self.assertEqual(ollama("Return procedure JSON.", json_mode=True,
            format_schema=PROCEDURE_RESPONSE_SCHEMA), "{}")
        self.assertEqual(post.call_count, 2)
        self.assertIs(post.call_args_list[0].kwargs["json"]["format"], PROCEDURE_RESPONSE_SCHEMA)
        self.assertEqual(post.call_args_list[1].kwargs["json"]["format"], "json")

    @patch("app.ai.ollama")
    def test_full_video_structured_output_accepts_valid_title_and_schema(self, ollama):
        import json
        ollama.return_value = json.dumps(self._soldering_result())
        result = draft_procedure("soldering.mp4", self._soldering_evidence(), "en",
            output_schema=PROCEDURE_RESPONSE_SCHEMA, fallback_title="soldering.mp4")
        self.assertEqual(result["title"], "Inspect the solder joint")
        self.assertNotIn("title_fallback_used", result)
        self.assertEqual(ollama.call_count, 1)
        self.assertIs(ollama.call_args.kwargs["format_schema"], PROCEDURE_RESPONSE_SCHEMA)
        self.assertIn("title", PROCEDURE_RESPONSE_SCHEMA["required"])

    @patch("app.ai.ollama")
    def test_missing_title_gets_one_controlled_title_repair(self, ollama):
        import json
        ollama.side_effect = [json.dumps(self._soldering_result("")), '{"title":"Inspect the solder joint"}']
        result = draft_procedure("soldering.mp4", self._soldering_evidence(), "en",
            output_schema=PROCEDURE_RESPONSE_SCHEMA, fallback_title="soldering.mp4")
        self.assertEqual(ollama.call_count, 2)
        self.assertIs(ollama.call_args_list[0].kwargs["format_schema"], PROCEDURE_RESPONSE_SCHEMA)
        self.assertIs(ollama.call_args_list[1].kwargs["format_schema"], TITLE_REPAIR_SCHEMA)
        self.assertIn("VALIDATED PROCEDURE CONTENT", ollama.call_args_list[1].args[0])
        self.assertEqual(result["title"], "Inspect the solder joint")
        self.assertNotIn("title_fallback_used", result)

    @patch("app.ai.ollama")
    def test_missing_title_after_repair_uses_source_name_fallback(self, ollama):
        import json
        ollama.side_effect = [json.dumps(self._soldering_result("")), '{"title":""}']
        result = draft_procedure("soldering.mp4", self._soldering_evidence(), "en",
            output_schema=PROCEDURE_RESPONSE_SCHEMA, fallback_title="Soldering_Tutorial.mp4")
        self.assertEqual(ollama.call_count, 2)
        self.assertEqual(result["title"], "Soldering Tutorial — SOP Draft")
        self.assertTrue(result["title_fallback_used"])
        self.assertTrue(any("TITLE REVIEW REQUIRED" in warning for warning in result["warnings"]))

    @patch("app.ai.ollama")
    def test_missing_title_with_no_actionable_steps_is_refused_without_retry(self, ollama):
        import json
        ollama.return_value = json.dumps(self._soldering_result("", steps=[]))
        with self.assertRaisesRegex(ValueError, "did not support any reliable procedure steps"):
            draft_procedure("soldering.mp4", self._soldering_evidence(), "en",
                output_schema=PROCEDURE_RESPONSE_SCHEMA, fallback_title="soldering.mp4")
        ollama.assert_called_once()

    @patch("app.ai.ollama")
    def test_unsupported_content_is_rejected_before_missing_title_repair(self, ollama):
        import json
        unrelated = self._soldering_result("", steps=[{"instruction": "Open a command line as administrator and configure Python."}])
        unrelated["description"] = "Create a Python virtual environment."
        unrelated["tools"] = ["Python"]
        ollama.side_effect = [json.dumps(unrelated), '{"supported":false,"reason":"The source discusses soldering, not Python."}']
        with self.assertRaisesRegex(ValueError, "did not sufficiently match"):
            draft_procedure("soldering.mp4", self._soldering_evidence(), "en",
                output_schema=PROCEDURE_RESPONSE_SCHEMA, fallback_title="soldering.mp4",
                force_consistency_check=True)
        self.assertEqual(ollama.call_count, 2)
        self.assertIn("Check source-to-output consistency", ollama.call_args_list[1].args[0])
        self.assertNotIn("Supply only a concise, meaningful title", ollama.call_args_list[1].args[0])

    @patch("app.ai.ollama")
    def test_malformed_output_and_invalid_step_structure_are_rejected(self, ollama):
        import json
        with self.subTest("malformed JSON"):
            ollama.return_value = '{"title":'
            with self.assertRaisesRegex(LocalModelError, "invalid structured output"):
                draft_procedure("soldering.mp4", self._soldering_evidence(), "en",
                    output_schema=PROCEDURE_RESPONSE_SCHEMA, fallback_title="soldering.mp4")
            ollama.assert_called_once()

        with self.subTest("empty instruction"):
            ollama.reset_mock()
            ollama.return_value = json.dumps(self._soldering_result(steps=[{"instruction": ""}]))
            with self.assertRaisesRegex(LocalModelError, "malformed step"):
                draft_procedure("soldering.mp4", self._soldering_evidence(), "en",
                    output_schema=PROCEDURE_RESPONSE_SCHEMA, fallback_title="soldering.mp4")
            ollama.assert_called_once()

    @patch("app.ai.ollama")
    def test_selected_evidence_is_sent_and_timestamp_is_retained(self, ollama):
        ollama.return_value = '{"title":"ドリルビットの確認","description":"電源を切り、チャックとビットを確認します。","tools":[],"warnings":[],"steps":[{"instruction":"電源を切り、チャックとドリルビットを確認します。","timestamp":"00:12","source_ref":"DrillMachine.mp4 · 00:12"}]}'
        result = draft_procedure("DrillMachine.mp4", self.excerpts, "ja")
        prompt = ollama.call_args.args[0]
        self.assertIn(self.excerpts[0]["text"], prompt)
        self.assertIn("SOURCE REF: DrillMachine.mp4 · 00:12", prompt)
        self.assertEqual(result["steps"][0]["timestamp"], "00:12")
        self.assertEqual(result["steps"][0]["source_ref"], "DrillMachine.mp4 · 00:12")

    @patch("app.ai.ollama")
    def test_video_prompt_carries_timestamp_window_links_and_context_stays_non_evidentiary(self, ollama):
        import json
        ollama.return_value = json.dumps(self._soldering_result())
        evidence = [{"text": "Move the soldering iron across the joint until solder flows.",
            "source_id": 21, "source_ref": "soldering.mp4 · 00:31", "timestamp": "00:31",
            "timestamp_start": 31, "timestamp_end": 35, "window_id": 8, "window_ids": [8],
            "alignment_status": "timestamp_aligned_to_action_window", "evidence_type": "source_supported"}]
        context_only = [{"text": "This is just an Opening logo", "source_id": 21,
            "source_ref": "soldering.mp4 · expert answer · 00:09", "timestamp": "00:09",
            "window_id": 3, "evidence_type": "expert_confirmed"},
            {"text": "An unrelated reference describes inspection after cooling.", "source_id": 99,
             "source_ref": "other-manual.pdf · page 4", "evidence_type": "approved_sop_reference"}]
        draft_procedure("soldering.mp4", evidence, "en", context_excerpts=context_only)
        prompt = ollama.call_args.args[0]
        self.assertIn("SOURCE ID: 21", prompt)
        self.assertIn("ACTION WINDOW IDS: 8", prompt)
        self.assertIn("TIMESTAMP_RANGE_SECONDS: 31–35", prompt)
        self.assertIn("RELATED / UNALIGNED CONTEXT (not procedural evidence)", prompt)
        self.assertIn("This is just an Opening logo", prompt)
        self.assertIn("DO NOT USE TO SUPPORT A PROCEDURE CLAIM OR STEP", prompt)
        self.assertIn("Related documents and approved SOPs are comparison context", prompt)

    @patch("app.ai.ollama")
    def test_actionable_transcript_gets_one_limited_draft_retry_after_visual_coverage_refusal(self, ollama):
        import json
        refused = {"evidence_quality": "unusable", "insufficient_evidence": True,
            "reason": "Visual coverage is incomplete.", "title": "", "description": "",
            "tools": [], "warnings": [], "steps": []}
        limited = {"evidence_quality": "limited", "insufficient_evidence": False,
            "reason": "The transcript does not establish setup or acceptance criteria.",
            "title": "Solder a joint", "description": "Follow the narrated soldering actions; expert verification is required.",
            "tools": ["Soldering iron"], "warnings": ["Verify setup and acceptance criteria with an expert."],
            "steps": [{"instruction": "Move the soldering iron across the joint until solder flows.",
                "timestamp": "00:31", "source_ref": "soldering.mp4 · 00:31", "evidence_type": "source_supported"}]}
        ollama.side_effect = [json.dumps(refused), json.dumps(limited), '{"supported":true,"reason":"The action is in the transcript."}']
        evidence = [{"text": "Move the soldering iron across the joint until solder flows.",
            "source_ref": "soldering.mp4 · 00:31", "timestamp": "00:31", "timestamp_start": 31,
            "timestamp_end": 35, "source_id": 21, "window_id": 8, "window_ids": [8],
            "alignment_status": "timestamp_aligned_to_action_window", "evidence_type": "source_supported"},
            {"text": "UNVERIFIED AI VISUAL OBSERVATION: A hand is holding the soldering iron.",
             "source_ref": "soldering.mp4 · frame 4 · 00:32", "timestamp": "00:32", "timestamp_start": 32,
             "timestamp_end": 32, "source_id": 21, "window_id": 8, "window_ids": [8],
             "alignment_status": "selected_action_moment", "evidence_type": "visual_observation"}]
        result = draft_procedure("soldering.mp4", evidence, "en", force_consistency_check=True,
            output_schema=PROCEDURE_RESPONSE_SCHEMA, fallback_title="soldering.mp4")
        self.assertEqual(result["evidence_quality"], "limited")
        self.assertEqual(len(result["steps"]), 1)
        self.assertEqual(ollama.call_count, 3)  # initial, one generation retry, then consistency check
        self.assertIn("Assess these timestamped transcript segments directly", ollama.call_args_list[1].args[0])
        self.assertIn("ACTION WINDOW IDS: 8", ollama.call_args_list[1].args[0])
        self.assertIn("Move the soldering iron across the joint", ollama.call_args_list[1].args[0])
        self.assertNotIn("A hand is holding the soldering iron", ollama.call_args_list[1].args[0])

    @patch("app.ai.ollama")
    def test_visual_evidence_can_trigger_reassessment_without_expert_answers(self, ollama):
        import json
        from app.ai import _has_actionable_source_cue
        visual = [{"text": "The operator clamps the part and moves the tool along its edge.",
            "source_id": 28, "source_ref": "clip.mp4 · frame 12 · 00:24", "timestamp": "00:24",
            "timestamp_start": 24, "timestamp_end": 24, "window_ids": [6],
            "alignment_status": "timestamp_aligned_to_action_window", "evidence_type": "visual_observation"}]
        self.assertTrue(_has_actionable_source_cue(visual))
        refused = {"evidence_quality": "unusable", "insufficient_evidence": True,
            "reason": "No narration was available.", "title": "", "description": "", "tools": [], "warnings": [], "steps": []}
        limited = {"evidence_quality": "limited", "insufficient_evidence": False,
            "reason": "Only visible actions are supported.", "title": "Position the part",
            "description": "Provisional visual sequence.", "tools": [], "warnings": ["Verify settings."],
            "steps": [{"instruction": "Secure the part before moving the tool along its edge.",
                "timestamp": "00:24", "source_ref": visual[0]["source_ref"]}]}
        ollama.side_effect = [json.dumps(refused), json.dumps(limited), '{"supported":true,"reason":"The visual source depicts the action."}']
        result = draft_procedure("clip.mp4", visual, "en", output_schema=PROCEDURE_RESPONSE_SCHEMA)
        self.assertEqual(result["evidence_quality"], "limited")
        self.assertEqual(ollama.call_count, 3)
        self.assertIn("direct timestamped visual evidence", ollama.call_args_list[1].args[0])

    @patch("app.ai.ollama")
    def test_unusable_non_actionable_source_does_not_get_a_generation_retry(self, ollama):
        import json
        ollama.return_value = json.dumps({"evidence_quality": "unusable", "insufficient_evidence": True,
            "reason": "The source only introduces a subject.", "title": "", "description": "",
            "tools": [], "warnings": [], "steps": []})
        evidence = [{"text": "Welcome to this short introduction.", "source_id": 4,
            "source_ref": "intro.mp4 · 00:01", "timestamp": "00:01", "evidence_type": "source_supported"}]
        with self.assertRaisesRegex(ValueError, "only introduces a subject"):
            draft_procedure("intro.mp4", evidence, "en", output_schema=PROCEDURE_RESPONSE_SCHEMA,
                fallback_title="intro.mp4")
        ollama.assert_called_once()

    @patch("app.ai.ollama")
    def test_unrelated_subject_is_rejected(self, ollama):
        ollama.side_effect = [
            '{"title":"User-Facing SOP for Setting Up a Virtual Environment","description":"Create a Python virtual environment.","tools":["Python"],"warnings":[],"steps":[{"instruction":"Open a command-line interface as administrator and navigate directories."}]}',
            '{"supported":false,"reason":"The source describes drill inspection, not software setup."}',
        ]
        with self.assertRaisesRegex(ValueError, "did not sufficiently match"):
            draft_procedure("DrillMachine.mp4", self.excerpts, "en")

    @patch("app.ai.ollama")
    def test_insufficient_evidence_fails_without_fallback_sop(self, ollama):
        ollama.return_value = '{"insufficient_evidence":true,"reason":"Transcript is unclear."}'
        with self.assertRaisesRegex(ValueError, "Transcript is unclear"):
            draft_procedure("DrillMachine.mp4", self.excerpts, "en")

    def test_empty_evidence_does_not_call_model(self):
        with patch("app.ai.ollama") as ollama:
            with self.assertRaisesRegex(ValueError, "no usable"):
                draft_procedure("empty.mp4", [{"text": "", "source_ref": "00:01"}])
            ollama.assert_not_called()

    @patch("app.ai.ollama")
    def test_short_japanese_transcript_gets_quality_assessment(self, ollama):
        segments = [
            "ドリルの仕組みはとてもシンプルです", "ここがチャックです",
            "把をしっかり固定します", "モーターで高速回転させます", "これで穴は開きます",
        ]
        short_source = [{"text": text, "source_ref": f"DrillMachine.mp4 · 00:{i * 2:02d}", "timestamp": f"00:{i * 2:02d}"} for i, text in enumerate(segments)]
        ollama.return_value = '{"evidence_quality":"unusable","insufficient_evidence":true,"reason":"The action is not sufficiently clear to draft a reliable instruction."}'
        with self.assertRaisesRegex(ValueError, "not sufficiently clear"):
            draft_procedure("clip.mp4", short_source, "ja")
        ollama.assert_called_once()

    @patch("app.ai.ollama")
    def test_limited_supported_step_is_retained_as_limited(self, ollama):
        ollama.return_value = '{"evidence_quality":"limited","reason":"Only the described action is covered; setup and safety checks are missing.","title":"チャックを確認します","description":"作業前にチャックを確認します。","tools":[],"warnings":[],"steps":[{"instruction":"作業前にチャックを確認します。","timestamp":"00:03","source_ref":"clip.mp4 · 00:03"}]}'
        source = [{"text": "作業前にチャックを確認します。", "source_ref": "clip.mp4 · 00:03", "timestamp": "00:03"}]
        result = draft_procedure("clip.mp4", source, "ja")
        self.assertEqual(result["evidence_quality"], "limited")
        self.assertEqual(result["steps"][0]["timestamp"], "00:03")

    @patch("app.ai.ollama")
    def test_forced_video_consistency_check_and_source_timestamp_normalization(self, ollama):
        ollama.side_effect = [
            '{"evidence_quality":"limited","title":"Move the iron across the joint","description":"A limited soldering action.","tools":[],"warnings":[],"steps":[{"instruction":"Move the soldering iron across the joint until solder flows.","timestamp":"00:32","source_ref":"model-invented reference","evidence_type":"ai_inferred"}]}',
            '{"supported":true,"reason":"The step matches the transcript."}',
        ]
        evidence = [{"text":"Move the soldering iron across the joint until solder flows.",
            "source_ref":"soldering.mp4 · 00:31","timestamp":"00:31","evidence_type":"source_supported"}]
        result = draft_procedure("soldering.mp4", evidence, "en", force_consistency_check=True)
        self.assertEqual(ollama.call_count, 2)
        self.assertIs(ollama.call_args_list[1].kwargs["format_schema"], CONSISTENCY_RESPONSE_SCHEMA)
        self.assertEqual(result["steps"][0]["timestamp"], "00:31")
        self.assertEqual(result["steps"][0]["source_ref"], "soldering.mp4 · 00:31")
        self.assertEqual(result["steps"][0]["evidence_type"], "source_supported")

    @patch("app.ai.ollama")
    def test_full_video_marks_a_useful_partial_timeline_after_one_coverage_retry(self, ollama):
        import json
        tail_only = json.dumps({
            "evidence_quality": "sufficient", "insufficient_evidence": False,
            "reason": "The described actions are supported.",
            "title": "Solder a component", "description": "A sequence of soldering actions.",
            "tools": [], "warnings": [],
            "steps": [{"instruction": "Clean the iron.", "timestamp": "01:30",
                       "source_ref": "soldering.mp4 · 01:30"}],
        })
        evidence = [
            {"text": text, "source_ref": f"soldering.mp4 · {stamp}", "timestamp": stamp,
             "timestamp_start": seconds, "timestamp_end": seconds + 1,
             "evidence_type": "source_supported"}
            for text, stamp, seconds in [
                ("Clean the tip before work.", "00:10", 10),
                ("Bend the component leads.", "00:50", 50),
                ("Do not grab the hot joint.", "03:20", 200),
            ]
        ]
        ollama.side_effect = [tail_only, tail_only, '{"supported":true,"reason":"The cited tail action is supported."}']
        result = draft_procedure("soldering.mp4", evidence, "en", output_schema=PROCEDURE_RESPONSE_SCHEMA,
            require_full_video_coverage=True, force_consistency_check=True)
        self.assertEqual(result["evidence_quality"], "limited")
        self.assertTrue(any("TIMELINE COVERAGE REVIEW REQUIRED" in warning for warning in result["warnings"]))
        self.assertEqual(ollama.call_count, 3)
        self.assertIn("Timeline coverage anchors", ollama.call_args_list[1].args[0])

    @patch("app.ai.ollama")
    def test_full_video_retries_once_and_accepts_supported_beginning_and_end_steps(self, ollama):
        import json
        tail_only = json.dumps({
            "evidence_quality": "sufficient", "insufficient_evidence": False,
            "reason": "The action is supported.", "title": "Solder a component",
            "description": "A sequence of soldering actions.", "tools": [], "warnings": [],
            "steps": [{"instruction": "Bend the component leads.", "timestamp": "00:50",
                       "source_ref": "soldering.mp4 · 00:50"}],
        })
        full_timeline = json.dumps({
            "evidence_quality": "limited", "insufficient_evidence": False,
            "reason": "The source does not cover every setup and verification detail.",
            "title": "Solder a component", "description": "A limited sequence of narrated soldering actions.",
            "tools": [], "warnings": [], "steps": [
                {"instruction": "Clean the tip before work.", "timestamp": "00:10",
                 "source_ref": "soldering.mp4 · 00:10"},
                {"instruction": "Do not grab the hot joint.", "timestamp": "03:20",
                 "source_ref": "soldering.mp4 · 03:20"},
            ],
        })
        ollama.side_effect = [tail_only, full_timeline, '{"supported":true,"reason":"Both steps match source evidence."}']
        evidence = [
            {"text": text, "source_ref": f"soldering.mp4 · {stamp}", "timestamp": stamp,
             "timestamp_start": seconds, "timestamp_end": seconds + 1,
             "evidence_type": "source_supported"}
            for text, stamp, seconds in [
                ("Clean the tip before work.", "00:10", 10),
                ("Bend the component leads.", "00:50", 50),
                ("Do not grab the hot joint.", "03:20", 200),
            ]
        ]
        result = draft_procedure("soldering.mp4", evidence, "en",
            force_consistency_check=True, output_schema=PROCEDURE_RESPONSE_SCHEMA,
            require_full_video_coverage=True)
        self.assertEqual([step["timestamp"] for step in result["steps"]], ["00:10", "03:20"])
        self.assertEqual(ollama.call_count, 3)

    def test_all_aligned_visual_observations_are_available_for_source_assessment(self):
        visuals = [
            {"text": "UNVERIFIED AI VISUAL OBSERVATION: A title card is visible.",
             "timestamp_start": 1, "alignment_status": "timestamp_aligned_to_action_window", "window_id": 7},
            {"text": "UNVERIFIED AI VISUAL OBSERVATION: A hand is near the object.",
             "timestamp_start": 4, "alignment_status": "timestamp_aligned_to_action_window", "window_id": 8},
            {"text": "UNVERIFIED AI VISUAL OBSERVATION: No window link.",
             "timestamp_start": 6, "alignment_status": "unaligned"},
        ]
        selected = _select_prompt_visuals(visuals)
        self.assertEqual([item["timestamp_start"] for item in selected], [1, 4])
        self.assertIn("title card", selected[0]["text"])

    @patch("app.ai.ollama")
    def test_short_actionable_timeline_retries_for_coverage_without_duration_cutoff(self, ollama):
        import json
        tail_only = {"evidence_quality": "limited", "insufficient_evidence": False,
            "reason": "Only one action was included.", "title": "Inspect the component",
            "description": "A limited check.", "tools": [], "warnings": [],
            "steps": [{"instruction": "Check the component.", "timestamp": "00:08",
                "source_ref": "clip.mp4 · 00:08"}]}
        full = {**tail_only, "steps": [
            {"instruction": "Position the component.", "timestamp": "00:02", "source_ref": "clip.mp4 · 00:02"},
            {"instruction": "Check the component.", "timestamp": "00:08", "source_ref": "clip.mp4 · 00:08"},
        ]}
        ollama.side_effect = [json.dumps(tail_only), json.dumps(full), '{"supported":true,"reason":"Both timeline anchors match."}']
        evidence = [{"text": "Position the component.", "source_ref": "clip.mp4 · 00:02", "timestamp": "00:02",
                "timestamp_start": 2, "evidence_type": "source_supported"},
            {"text": "Check the component.", "source_ref": "clip.mp4 · 00:08", "timestamp": "00:08",
                "timestamp_start": 8, "evidence_type": "source_supported"}]
        result = draft_procedure("clip.mp4", evidence, "en", output_schema=PROCEDURE_RESPONSE_SCHEMA,
            require_full_video_coverage=True, force_consistency_check=True)
        self.assertEqual([step["timestamp"] for step in result["steps"]], ["00:02", "00:08"])
        self.assertEqual(ollama.call_count, 3)
        self.assertIn("FULL SELECTED-SOURCE TIMELINE", ollama.call_args_list[0].args[0])

    @patch("app.ai.ollama")
    def test_long_video_evidence_is_summarized_in_packets_without_dropping_timeline_items(self, ollama):
        import json
        from app import config
        evidence = [
            {"text": "Position the workpiece before the next operation. " + ("detail " * 45),
             "source_ref": "video.mp4 · 00:01", "timestamp": "00:01", "timestamp_start": 1,
             "source_id": 300, "evidence_type": "source_supported"},
            {"text": "Check the assembly after the movement. " + ("observation " * 45),
             "source_ref": "video.mp4 · 00:02", "timestamp": "00:02", "timestamp_start": 2,
             "source_id": 300, "evidence_type": "source_supported"},
            {"text": "The opening title card is visible. " + ("scene " * 45),
             "source_ref": "video.mp4 · 00:03", "timestamp": "00:03", "timestamp_start": 3,
             "source_id": 300, "evidence_type": "visual_observation"},
        ]
        summaries = [
            {"items": [{"item_id": 0, "relevance": "action", "summary": "Position the workpiece."}]},
            {"items": [{"item_id": 1, "relevance": "action", "summary": "Check the assembly."}]},
            {"items": [{"item_id": 2, "relevance": "context", "summary": "An opening title card is visible."}]},
        ]
        generation = {"evidence_quality": "limited", "insufficient_evidence": False,
            "reason": "The recording does not establish setup or acceptance criteria.",
            "title": "Position and inspect the assembly", "description": "A limited sequence of evidenced actions.",
            "tools": [], "warnings": [], "steps": [
                {"instruction": "Position the workpiece.", "timestamp": "00:01", "source_ref": "video.mp4 · 00:01"},
                {"instruction": "Check the assembly.", "timestamp": "00:02", "source_ref": "video.mp4 · 00:02"},
            ]}
        with patch.object(config, "VIDEO_SOP_CONTEXT_CHAR_LIMIT", 100), \
             patch.object(config, "VIDEO_SOP_EVIDENCE_PACKET_CHAR_LIMIT", 300):
            ollama.side_effect = [*(json.dumps(row) for row in summaries), json.dumps(generation),
                                  '{"supported":true,"reason":"Both source-linked actions are supported."}']
            result = draft_procedure("video.mp4", evidence, "en", output_schema=PROCEDURE_RESPONSE_SCHEMA,
                require_full_video_coverage=True, force_consistency_check=True, diagnostic_source_id=300)
        self.assertEqual(ollama.call_count, 5)
        self.assertTrue(all("PACKET " in call.args[0] for call in ollama.call_args_list[:3]))
        final_prompt = ollama.call_args_list[3].args[0]
        for source_ref in ("video.mp4 · 00:01", "video.mp4 · 00:02", "video.mp4 · 00:03"):
            self.assertIn(source_ref, final_prompt)
        self.assertIn("CONTEXT ONLY", final_prompt)
        self.assertIn("LONG VIDEO TIMELINE REVIEW REQUIRED", result["warnings"][-1])
        self.assertEqual([step["timestamp"] for step in result["steps"]], ["00:01", "00:02"])

    @patch("app.ai.ollama")
    def test_frame_label_is_not_accepted_as_an_actionable_step(self, ollama):
        import json
        ollama.return_value = json.dumps({"evidence_quality": "sufficient", "insufficient_evidence": False,
            "reason": "The output matches the source.", "title": "Solder a component",
            "description": "A sequence of soldering actions.", "tools": [], "warnings": [],
            "steps": [{"instruction": "Frame 132", "timestamp": "00:21",
                       "source_ref": "soldering.mp4 · frame 132 · 00:21"}]})
        evidence = [{"text": "Apply solder to the joint.", "source_id": 21,
            "source_ref": "soldering.mp4 · frame 132 · 00:21", "timestamp": "00:21",
            "evidence_type": "visual_observation"}]
        with self.assertRaisesRegex(EvidenceMismatchError, "frame or timeline label"):
            draft_procedure("soldering.mp4", evidence, "en", output_schema=PROCEDURE_RESPONSE_SCHEMA)
        ollama.assert_called_once()

    @patch("app.ai.ollama")
    def test_video_consistency_refusal_blocks_unsupported_output(self, ollama):
        ollama.side_effect = [
            '{"evidence_quality":"limited","title":"Move the iron","description":"Limited action.","tools":["Python"],"warnings":[],"steps":[{"instruction":"Move the soldering iron across the joint until solder flows.","source_ref":"soldering.mp4 · 00:31"}]}',
            '{"supported":false,"reason":"Python is unrelated to the soldering evidence."}',
        ]
        evidence = [{"text":"Move the soldering iron across the joint until solder flows.",
            "source_ref":"soldering.mp4 · 00:31","timestamp":"00:31","evidence_type":"source_supported"}]
        with self.assertRaisesRegex(ValueError, "did not sufficiently match"):
            draft_procedure("soldering.mp4", evidence, "en", force_consistency_check=True)

    def test_japanese_selection_reaches_whisper_and_timestamps_are_kept(self):
        calls = {}
        class FakeModel:
            def __init__(self, name, **kwargs):
                calls["model"] = (name, kwargs)
            def transcribe(self, path, **kwargs):
                calls["transcribe"] = (path, kwargs)
                return ([SimpleNamespace(text=" ドリルを確認します ", start=62.4, end=64.8)], SimpleNamespace(language="ja", language_probability=0.9))
        fake_module = ModuleType("faster_whisper")
        fake_module.WhisperModel = FakeModel
        with patch.dict(sys.modules, {"faster_whisper": fake_module}):
            segments, info = transcribe_video("/tmp/input.mp4", "ja", "DrillMachine.mp4")
        self.assertEqual(calls["transcribe"][1]["language"], "ja")
        self.assertEqual(calls["model"][1], {"device": "cpu", "compute_type": "int8"})
        self.assertEqual((segments[0]["start_seconds"], segments[0]["end_seconds"]), (62, 64))
        self.assertEqual(segments[0]["source_ref"], "DrillMachine.mp4 · 01:02")
        self.assertEqual(info.language, "ja")

    def test_source_id_isolation_uses_only_selected_chunks(self):
        engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(engine)
        with Session(engine) as db:
            db.add_all([
                Source(id=7, source_type="video", filename="first.mp4", file_path="first.mp4"),
                Source(id=11, source_type="video", filename="second.mp4", file_path="second.mp4"),
            ])
            db.flush()
            db.add_all([
                KnowledgeChunk(source_id=7, text="first transcript", source_ref="first · 00:01", start_seconds=1, end_seconds=2, embedding_json="[]"),
                KnowledgeChunk(source_id=11, text="second transcript", source_ref="second · 00:03", start_seconds=3, end_seconds=4, embedding_json="[]"),
            ])
            db.flush()
            excerpts = load_source_excerpts(db, 11)
        self.assertEqual([item["text"] for item in excerpts], ["second transcript"])
        self.assertEqual(excerpts[0]["source_id"], 11)
        self.assertEqual(excerpts[0]["timestamp"], "00:03")
        engine.dispose()


if __name__ == "__main__":
    unittest.main()
