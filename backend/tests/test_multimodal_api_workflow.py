"""Isolated API workflow tests using a temporary SQLite database, never user data."""
import json
import os
import sys
import tempfile
import unittest

_TEMP = tempfile.TemporaryDirectory(prefix="skillbridge-multimodal-tests-")
os.environ["DATABASE_URL"] = f"sqlite:///{_TEMP.name}/test.db"
os.environ["STORAGE_DIR"] = f"{_TEMP.name}/storage"
sys.path.insert(0, "backend")

from app.db import Base, engine, SessionLocal  # noqa: E402
from app.db import get_db  # noqa: E402
from app.main import (  # noqa: E402
    app, create_draft, create_sop_proposal, generate_video_expert_questions, review_procedure,
    save_video_expert_answer, submit_sop_proposal, video_sop_drafts,
)
from app.models import (  # noqa: E402
    KnowledgeChunk, Procedure, SOPImprovementProposal, Source, VideoActionWindow,
    VideoAnalysis, VideoCandidate, VideoExpertQuestion, VideoFrame, VideoPipelineRun, SOPComparison,
    VideoSequence, VideoSOPDraftVersion,
)
from app.schemas import ReviewRequest, VideoExpertAnswerRequest  # noqa: E402
from app.ai import LocalModelError, PROCEDURE_RESPONSE_SCHEMA  # noqa: E402
from app.video_jobs import _sop_fingerprint  # noqa: E402
from app.video_jobs import _approved_procedure_references  # noqa: E402
from app.video_jobs import _run  # noqa: E402
from unittest.mock import patch  # noqa: E402
import numpy as np  # noqa: E402
from fastapi import HTTPException  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402


class MultimodalWorkflowTests(unittest.TestCase):
    def setUp(self):
        Base.metadata.drop_all(bind=engine)
        Base.metadata.create_all(bind=engine)

    def _request_evidence_timeline(self, source_id):
        def override_get_db():
            with SessionLocal() as db:
                yield db

        app.dependency_overrides[get_db] = override_get_db
        try:
            return TestClient(app).get(f"/api/sources/{source_id}/evidence-timeline")
        finally:
            app.dependency_overrides.pop(get_db, None)

    def test_evidence_timeline_endpoint_returns_existing_run_and_sequence_data(self):
        with SessionLocal() as db:
            source = Source(id=151, source_type="video", filename="timeline.mp4", file_path="/unused/timeline.mp4")
            db.add(source)
            db.flush()
            run = VideoPipelineRun(source_id=source.id, input_sha256="a" * 64, config_sha256="b" * 64,
                strategy_version="test", status="completed", stage="completed")
            db.add(run)
            db.flush()
            frame = VideoFrame(source_id=source.id, timestamp_seconds=4.0, frame_number=40,
                frame_path="/unused/frame.jpg", analysis_status="completed",
                analysis_json=json.dumps({"description": "A workpiece is visible."}))
            db.add(frame)
            db.flush()
            candidate = VideoCandidate(run_id=run.id, source_id=source.id, frame_id=frame.id,
                timestamp_seconds=4.0, selected_for_detail=True)
            window = VideoActionWindow(run_id=run.id, source_id=source.id, timestamp_start=3.0,
                timestamp_end=5.0, transcript_json=json.dumps([{"text": "Check the workpiece."}]),
                summary=json.dumps({"sequence_summary": "A workpiece is visible."}), status="analyzed")
            sequence = VideoSequence(source_id=source.id, timestamp_start=4.0, timestamp_end=6.0,
                evidence_json=json.dumps({"sequence_summary": "The workpiece remains in view."}), model_name="test")
            chunk = KnowledgeChunk(source_id=source.id, text="Check the workpiece.",
                source_ref="timeline.mp4 · 00:03", start_seconds=3, end_seconds=5)
            db.add_all([candidate, window, sequence, chunk])
            db.commit()
            source_id = source.id

        response = self._request_evidence_timeline(source_id)

        self.assertEqual(response.status_code, 200)
        entries = response.json()
        self.assertEqual(len(entries), 4)
        self.assertEqual({entry["source_id"] for entry in entries}, {source_id})
        self.assertIn("visual_sequence", [kind for entry in entries for kind in entry["evidence_types"]])
        self.assertTrue(any(entry.get("kind") == "action_window" for entry in entries))
        self.assertTrue(any(entry.get("frame_id") is not None for entry in entries))
        self.assertTrue(any(entry.get("audio_text") == "Check the workpiece." for entry in entries))

    def test_evidence_timeline_endpoint_returns_empty_array_without_analysis_data(self):
        with SessionLocal() as db:
            source = Source(id=152, source_type="video", filename="empty-timeline.mp4", file_path="/unused/empty.mp4")
            db.add(source)
            db.commit()
            source_id = source.id

        response = self._request_evidence_timeline(source_id)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), [])

    def test_source_video_draft_history_is_read_without_changing_approval_state(self):
        with SessionLocal() as db:
            source = Source(source_type="video", filename="history.mp4", file_path="/unused/history.mp4")
            db.add(source)
            db.flush()
            run = VideoPipelineRun(source_id=source.id, input_sha256="e" * 64, config_sha256="f" * 64,
                strategy_version="test", status="completed", stage="completed")
            db.add(run)
            db.flush()
            procedure = Procedure(title="Historical approved SOP", description="Stored procedure.",
                source_id=source.id, status="approved")
            db.add(procedure)
            db.flush()
            version = VideoSOPDraftVersion(source_id=source.id, run_id=run.id, procedure_id=procedure.id,
                version=1, evidence_sha256="a" * 64, evidence_manifest_json="[]", warnings_json="[]",
                review_status="approved")
            db.add(version)
            db.commit()
            result = video_sop_drafts(source.id, db)
            db.refresh(procedure)
            self.assertEqual(result["video_sop_drafts"][0]["procedure"]["status"], "approved")
            self.assertEqual(result["video_sop_drafts"][0]["run_id"], run.id)
            self.assertEqual(procedure.status, "approved")

    def test_video_sop_generation_uses_only_transcript_and_does_not_require_analysis(self):
        with SessionLocal() as db:
            source = Source(source_type="video", filename="work-demo.mp4", file_path="/unused/work.mp4")
            other = Source(source_type="video", filename="other.mp4", file_path="/unused/other.mp4")
            db.add_all([source, other])
            db.flush()
            transcript = KnowledgeChunk(source_id=source.id,
                text="Clamp the workpiece securely in the vise.", source_ref="work-demo.mp4 · 00:12",
                start_seconds=12, end_seconds=15)
            visual_chunk = KnowledgeChunk(source_id=source.id,
                text="A hand is visible near a metal object.", source_ref="visual frame · 00:12",
                start_seconds=12, end_seconds=12)
            unrelated = KnowledgeChunk(source_id=other.id,
                text="Open a command prompt.", source_ref="other.mp4 · 00:02", start_seconds=2, end_seconds=4)
            db.add_all([transcript, visual_chunk, unrelated])
            db.commit()
            result_data = {"title": "Clamp the workpiece", "description": "Transcript-supported draft.",
                "tools": ["Vise"], "warnings": ["Verify clamping method with an expert."],
                "evidence_quality": "limited", "steps": [{"instruction": "Clamp the workpiece securely in the vise.",
                    "timestamp": "00:12", "source_ref": "work-demo.mp4 · 00:12", "evidence_type": "source_supported"}]}
            with patch("app.main.draft_procedure", return_value=result_data) as draft:
                result = create_draft(source.id, "en", db)
            args, kwargs = draft.call_args
            self.assertEqual(args[0], "uploaded video transcript")
            self.assertEqual([item["text"] for item in args[1]], [transcript.text])
            self.assertEqual(kwargs["context_excerpts"], [])
            self.assertEqual(result["status"], "draft")
            version = db.query(VideoSOPDraftVersion).filter_by(source_id=source.id).one()
            self.assertIsNone(version.run_id)
            self.assertEqual(json.loads(version.evidence_manifest_json), [{
                "kind": "transcript_chunk", "id": transcript.id, "source_id": source.id,
                "source_ref": transcript.source_ref, "timestamp": "00:12",
            }])

    def test_transcript_sop_digest_ignores_visual_analysis_and_analysis_status(self):
        with SessionLocal() as db:
            source, _, _, run = self._add_video_evidence(db, 155)
            from app.video_sop import assemble_transcript_sop_evidence
            before = assemble_transcript_sop_evidence(db, source)
            run.status = "running"
            db.add(VideoFrame(source_id=source.id, timestamp_seconds=200.0, frame_number=200,
                frame_path="/unused/new-frame.jpg", analysis_status="completed",
                analysis_json=json.dumps({"description": "New unrelated visual analysis."})))
            db.commit()
            after = assemble_transcript_sop_evidence(db, source)
            self.assertEqual(before["evidence_sha256"], after["evidence_sha256"])
            self.assertEqual(before["action_evidence"], after["action_evidence"])
            self.assertEqual(before["counts"], {"transcript_segments": 2})

    def test_expert_answer_is_saved_without_creating_individual_proposal_or_procedure(self):
        with SessionLocal() as db:
            source = Source(id=91, source_type="video", filename="training.mp4", file_path="/unused/training.mp4")
            db.add(source)
            db.flush()
            run = VideoPipelineRun(source_id=source.id, input_sha256="a" * 64, config_sha256="b" * 64,
                                   strategy_version="test", status="completed", stage="completed")
            db.add(run)
            db.flush()
            window = VideoActionWindow(run_id=run.id, source_id=source.id, timestamp_start=12.0,
                timestamp_end=16.0, transcript_json=json.dumps([{"text": "Check the component before continuing.",
                    "start_seconds": 13, "end_seconds": 15, "source_ref": "training.mp4 · 00:13"}]),
                summary=json.dumps({"sequence_summary": "A component is visible."}), status="analyzed")
            db.add(window)
            db.commit()

            first = generate_video_expert_questions(source.id, db)
            second = generate_video_expert_questions(source.id, db)
            self.assertEqual(len(first["video_questions"]), 1)
            self.assertEqual(len(second["video_questions"]), 1)
            question = db.query(VideoExpertQuestion).one()
            self.assertIn("Check the component", question.question)
            self.assertEqual(json.loads(question.evidence_json)["source_id"], source.id)

            save_video_expert_answer(question.id,
                VideoExpertAnswerRequest(answer="The operator checks the component is seated before continuing.", status="answered"), db)
            db.refresh(question)
            self.assertEqual(question.answer_status, "answered")
            self.assertIn("checks the component", question.answer)
            self.assertEqual(db.query(SOPImprovementProposal).count(), 0)
            self.assertEqual(db.query(Procedure).count(), 0)
            with self.assertRaises(HTTPException) as caught:
                create_sop_proposal(question.id, db)
            self.assertEqual(caught.exception.status_code, 410)
            self.assertEqual(db.query(SOPImprovementProposal).count(), 0)

    def _add_video_evidence(self, db, source_id=94):
        source = Source(id=source_id, source_type="video", filename="soldering-demo.mp4", file_path="/unused/video.mp4")
        unrelated = Source(id=source_id + 1, source_type="video", filename="other-video.mp4", file_path="/unused/other.mp4")
        manual = Source(id=source_id + 2, source_type="document", filename="approved-manual.pdf", file_path="/unused/manual.pdf")
        db.add_all([source, unrelated, manual])
        db.flush()
        run = VideoPipelineRun(source_id=source.id, input_sha256="c" * 64, config_sha256="d" * 64,
            strategy_version="test", status="partially_completed", stage="completed_with_gaps",
            metrics_json=json.dumps({"detailed_windows_covered": 1, "candidate_windows_not_detailed": 1}),
            warnings_json=json.dumps(["One candidate window was not analyzed in detail."]))
        db.add(run)
        db.flush()
        window = VideoActionWindow(run_id=run.id, source_id=source.id, timestamp_start=30.0, timestamp_end=38.0,
            transcript_json=json.dumps([{"text": "Move the soldering iron across the joint until solder flows.",
                "start_seconds": 31, "end_seconds": 35, "source_ref": "soldering-demo.mp4 · 00:31"}]),
            frame_ids_json="[]", summary=json.dumps({"sequence_summary": "The iron approaches a joint; AI observation is unverified."}),
            status="analyzed")
        db.add(window)
        db.flush()
        db.add(VideoAnalysis(source_id=source.id, status="partially_completed", progress=100))
        db.add_all([
            KnowledgeChunk(source_id=source.id, text="Move the soldering iron across the joint until solder flows.",
                source_ref="soldering-demo.mp4 · 00:31", start_seconds=31, end_seconds=35),
            KnowledgeChunk(source_id=source.id, text="Remove the iron after the solder wets the connection.",
                source_ref="soldering-demo.mp4 · 00:36", start_seconds=36, end_seconds=39),
            KnowledgeChunk(source_id=unrelated.id, text="Open a command line and configure Python.",
                source_ref="other-video.mp4 · 00:15", start_seconds=15, end_seconds=19),
            KnowledgeChunk(source_id=manual.id, text="Inspect the joint under magnification after cooling.",
                source_ref="approved-manual.pdf · page 3", embedding_json="[1,0]")
        ])
        frame = VideoFrame(source_id=source.id, timestamp_seconds=33.0, frame_number=330,
            frame_path="/unused/frame.jpg", analysis_status="completed",
            analysis_json=json.dumps({"description": "A soldering iron tip is visible near a joint.",
                "observable_actions": ["The tip moves along the joint."], "uncertainties": ["Exact hand action unverified"]}))
        db.add(frame)
        db.flush()
        window.frame_ids_json = json.dumps([frame.id])
        db.add(VideoCandidate(run_id=run.id, source_id=source.id, frame_id=frame.id,
            timestamp_seconds=33.0, selected_for_detail=True))
        manual_chunk = db.query(KnowledgeChunk).filter_by(source_id=manual.id).one()
        db.add(SOPComparison(run_id=run.id, source_id=source.id, window_id=window.id,
            relationship="human_comparison_required", explanation="Compare this manual passage with the demonstrated action.",
            references_json=json.dumps([{"source_id": manual.id, "filename": manual.filename,
                "reference": manual_chunk.source_ref, "chunk_id": manual_chunk.id,
                "excerpt": manual_chunk.text}]), evidence_json="{}"))
        db.add_all([
            VideoExpertQuestion(run_id=run.id, source_id=source.id, window_id=window.id,
                timestamp_seconds=9.0, category="scene_context", question="What is visible at this moment?",
                answer="This is just an Opening logo", answer_status="answered"),
            VideoExpertQuestion(run_id=run.id, source_id=source.id, window_id=window.id,
                timestamp_seconds=35.0, category="process", question="When should the iron be removed?",
                answer="Remove it after the solder wets the connection.", answer_status="answered"),
            VideoExpertQuestion(run_id=run.id, source_id=source.id, window_id=window.id,
                timestamp_seconds=37.0, category="quality_check", question="What confirms the joint is covered?",
                answer="Solder covers the joint across the connection.", answer_status="answered"),
        ])
        db.commit()
        return source, unrelated, manual, run

    def test_full_video_draft_aggregates_source_evidence_excludes_logo_as_step_and_reuses_version(self):
        with SessionLocal() as db:
            source, unrelated, manual, run = self._add_video_evidence(db)
            calls = []

            def fake_draft(name, excerpts, output_language="en", **kwargs):
                calls.append((name, excerpts, output_language, kwargs))
                return {"title": "Solder the joint", "description": "Limited draft from the analyzed demonstration.",
                    "tools": [], "warnings": [], "evidence_quality": "limited",
                    "validation_explanation": "Preparation and final quality criteria are not shown.",
                    "steps": [{"instruction": "Move the iron across the joint until solder flows.",
                        "timestamp": "00:31", "source_ref": "soldering-demo.mp4 · 00:31",
                        "evidence_type": "source_supported"}]}

            with patch("app.main.draft_procedure", side_effect=fake_draft), \
                 patch("app.knowledge.knowledge_index.search", return_value=[db.query(KnowledgeChunk).filter_by(source_id=manual.id).one()]), \
                 patch("app.video_jobs._approved_procedure_references", return_value=[]):
                first = create_draft(source.id, "en", db)
                second = create_draft(source.id, "en", db)

            self.assertEqual(len(calls), 1)
            excerpts = calls[0][1]
            self.assertTrue(any("Remove the iron" in item["text"] for item in excerpts))
            self.assertFalse(any("frame 1" in item["source_ref"] for item in excerpts))
            self.assertFalse(any(item["source_id"] == manual.id for item in excerpts))
            self.assertFalse(any(item["source_id"] == unrelated.id for item in excerpts))
            transcript_item = next(item for item in excerpts if "Move the soldering iron" in item["text"])
            self.assertEqual(transcript_item["source_id"], source.id)
            self.assertFalse(any(item.get("evidence_type") == "visual_observation" for item in excerpts))
            self.assertFalse(any("Opening logo" in item["text"] for item in excerpts))
            context = calls[0][3]["context_excerpts"]
            self.assertEqual(context, [])
            self.assertTrue(calls[0][3]["transcript_only"])
            self.assertIs(calls[0][3]["output_schema"], PROCEDURE_RESPONSE_SCHEMA)
            self.assertEqual(calls[0][3]["fallback_title"], source.filename)
            self.assertEqual(first["status"], "draft")
            self.assertIsNone(first["generation"]["run_id"])
            self.assertEqual(first["generation"]["version"], 1)
            self.assertEqual(first["generation"]["coverage_warnings"], [])
            self.assertTrue(second["generation"]["reused_existing_draft"])
            self.assertEqual(first["id"], second["id"])
            self.assertEqual(db.query(Procedure).filter_by(source_id=source.id).count(), 1)
            stored_version = db.query(VideoSOPDraftVersion).filter_by(source_id=source.id).one()
            manifest = json.loads(stored_version.evidence_manifest_json)
            self.assertTrue(manifest)
            self.assertTrue(all(item["kind"] == "transcript_chunk" for item in manifest))

    def test_actionable_video_can_draft_when_all_expert_questions_are_dismissed(self):
        with SessionLocal() as db:
            source, _, _, run = self._add_video_evidence(db, 101)
            for question in db.query(VideoExpertQuestion).filter_by(source_id=source.id, run_id=run.id):
                question.answer = ""
                question.answer_status = "dismissed"
            db.commit()
            data = {"title": "Solder the joint", "description": "Provisional actions visible in the analyzed video.",
                "tools": [], "warnings": ["Verify preparation and acceptance criteria with an expert."],
                "evidence_quality": "limited", "validation_explanation": "The demonstration omits preparation.",
                "steps": [{"instruction": "Move the iron across the joint until solder flows.",
                    "timestamp": "00:31", "source_ref": "soldering-demo.mp4 · 00:31",
                    "evidence_type": "source_supported"}]}
            with patch("app.main.draft_procedure", return_value=data) as draft, \
                 patch("app.knowledge.knowledge_index.search", return_value=[]), \
                 patch("app.video_jobs._approved_procedure_references", return_value=[]):
                result = create_draft(source.id, "en", db)
            passed_evidence = draft.call_args.args[1]
            self.assertFalse(any(item.get("evidence_type") == "expert_confirmed" for item in passed_evidence))
            self.assertTrue(any(item.get("evidence_type") == "source_supported" for item in passed_evidence))
            self.assertEqual(result["status"], "draft")
            self.assertIsNone(result["generation"]["run_id"])
            self.assertEqual(db.query(Procedure).filter_by(source_id=source.id).one().status, "draft")

    def test_timestamped_saved_sequence_is_relinked_to_current_run_window(self):
        with SessionLocal() as db:
            source, _, _, run = self._add_video_evidence(db, 102)
            window = db.query(VideoActionWindow).filter_by(source_id=source.id, run_id=run.id).one()
            sequence = VideoSequence(source_id=source.id, timestamp_start=32.0, timestamp_end=34.0,
                evidence_json=json.dumps({"sequence_summary": "The soldering iron tip is visible beside the joint."}),
                model_name="legacy-vision")
            db.add(sequence)
            db.commit()
            from app.video_sop import assemble_video_sop_evidence
            with patch("app.knowledge.knowledge_index.search", return_value=[]), \
                 patch("app.video_jobs._approved_procedure_references", return_value=[]):
                evidence = assemble_video_sop_evidence(db, source)
            saved = next(item for item in evidence["snapshot"]["visual_observations"] if item.get("sequence_id") == sequence.id)
            self.assertEqual(saved["window_id"], window.id)
            self.assertEqual(saved["timestamp"], "00:32")
            self.assertEqual(saved["alignment_status"], "timestamp_aligned_to_action_window")
            self.assertTrue(any(item.get("sequence_id") == sequence.id for item in evidence["action_evidence"]))
            self.assertTrue(any(item["kind"] == "video_sequence" and item["id"] == sequence.id for item in evidence["manifest"]))

    def test_unaligned_visual_observation_is_retained_but_not_used_as_step_evidence(self):
        with SessionLocal() as db:
            source, _, _, run = self._add_video_evidence(db, 103)
            frame = VideoFrame(source_id=source.id, timestamp_seconds=300.0, frame_number=9000,
                frame_path="/unused/unaligned.jpg", analysis_status="completed",
                analysis_json=json.dumps({"description": "A title card is visible."}))
            db.add(frame)
            db.flush()
            db.add(VideoCandidate(run_id=run.id, source_id=source.id, frame_id=frame.id,
                timestamp_seconds=frame.timestamp_seconds, selected_for_detail=True))
            db.commit()
            from app.video_sop import assemble_video_sop_evidence
            with patch("app.knowledge.knowledge_index.search", return_value=[]), \
                 patch("app.video_jobs._approved_procedure_references", return_value=[]):
                evidence = assemble_video_sop_evidence(db, source)
            self.assertFalse(any(item.get("frame_id") == frame.id for item in evidence["action_evidence"]))
            context = [item for item in evidence["contextual_evidence"] if item.get("frame_id") == frame.id]
            self.assertEqual(len(context), 1)
            self.assertEqual(context[0]["alignment_status"], "unaligned")
            self.assertTrue(any("excluded as step evidence" in warning for warning in evidence["warnings"]))

    def test_full_video_draft_refuses_model_insufficient_evidence_without_persisting(self):
        with SessionLocal() as db:
            source = Source(id=104, source_type="video", filename="introduction.mp4", file_path="/unused/intro.mp4")
            db.add(source)
            db.flush()
            run = VideoPipelineRun(source_id=source.id, input_sha256="1" * 64, config_sha256="2" * 64,
                strategy_version="test", status="completed", stage="completed")
            db.add(run)
            db.flush()
            db.add_all([
                VideoActionWindow(run_id=run.id, source_id=source.id, timestamp_start=0, timestamp_end=5,
                    transcript_json="[]", summary="{}", status="analyzed"),
                VideoAnalysis(source_id=source.id, status="completed", progress=100),
                KnowledgeChunk(source_id=source.id, text="This video introduces the workshop.",
                    source_ref="introduction.mp4 · 00:01", start_seconds=1, end_seconds=2),
            ])
            db.commit()
            with patch("app.main.draft_procedure", side_effect=ValueError(
                    "The source did not support any reliable procedure steps. No draft was created. Model assessment: only scene context.")), \
                 patch("app.knowledge.knowledge_index.search", return_value=[]), \
                 patch("app.video_jobs._approved_procedure_references", return_value=[]):
                with self.assertRaises(HTTPException) as caught:
                    create_draft(source.id, "en", db)
            self.assertEqual(caught.exception.status_code, 422)
            self.assertIn("transcript segments=1", caught.exception.detail)
            self.assertIn("source-evidence/generation refusal", caught.exception.detail)
            self.assertNotIn("resume analysis", caught.exception.detail)
            self.assertEqual(db.query(Procedure).filter_by(source_id=source.id).count(), 0)
            self.assertEqual(db.query(VideoSOPDraftVersion).filter_by(source_id=source.id).count(), 0)

    def test_model_refusal_with_actionable_transcript_does_not_persist_a_transcript_dump(self):
        with SessionLocal() as db:
            source, _, _, run = self._add_video_evidence(db, 105)
            refusal = ValueError("The model rejected the visual coverage as incomplete.")
            with patch("app.main.draft_procedure", side_effect=refusal) as draft, \
                 patch("app.knowledge.knowledge_index.search", return_value=[]), \
                 patch("app.video_jobs._approved_procedure_references", return_value=[]):
                with self.assertRaises(HTTPException) as caught:
                    create_draft(source.id, "en", db)
            self.assertEqual(draft.call_count, 1)
            self.assertEqual(caught.exception.status_code, 422)
            self.assertIn("source-evidence/generation refusal", caught.exception.detail)
            self.assertNotIn("resume analysis", caught.exception.detail)
            self.assertEqual(db.query(Procedure).filter_by(source_id=source.id).count(), 0)
            self.assertEqual(db.query(VideoSOPDraftVersion).filter_by(source_id=source.id).count(), 0)

    def test_partial_analysis_with_usable_model_output_is_saved_as_limited(self):
        with SessionLocal() as db:
            source, _, _, run = self._add_video_evidence(db, 153)
            model_draft = {"title": "Solder a component", "description": "A source-supported sequence.",
                "tools": [], "warnings": [], "evidence_quality": "sufficient", "validation_explanation": "",
                "steps": [{"instruction": "Move the soldering iron across the joint until solder flows.",
                           "timestamp": "00:31", "source_ref": "soldering-demo.mp4 · 00:31",
                           "evidence_type": "source_supported"}]}
            with patch("app.main.draft_procedure", return_value=model_draft), \
                 patch("app.knowledge.knowledge_index.search", return_value=[]), \
                 patch("app.video_jobs._approved_procedure_references", return_value=[]):
                result = create_draft(source.id, "en", db)
            self.assertIsNone(result["generation"]["run_id"])
            self.assertEqual(result["status"], "draft")
            self.assertFalse(any("partially complete" in str(warning).lower() for warning in result["warnings"]))
            self.assertEqual(result["generation"]["evidence_counts"], {"transcript_segments": 2})

    def test_video_model_failure_is_reported_as_service_error_not_insufficient_evidence(self):
        with SessionLocal() as db:
            source, _, _, _ = self._add_video_evidence(db, 109)
            with patch("app.main.draft_procedure", side_effect=LocalModelError("The local SOP model did not return a result.")), \
                 patch("app.knowledge.knowledge_index.search", return_value=[]), \
                 patch("app.video_jobs._approved_procedure_references", return_value=[]):
                with self.assertRaises(HTTPException) as caught:
                    create_draft(source.id, "en", db)
            self.assertEqual(caught.exception.status_code, 502)
            self.assertIn("local SOP model", caught.exception.detail)
            self.assertEqual(db.query(Procedure).filter_by(source_id=source.id).count(), 0)

    def test_japanese_drill_segments_reach_full_source_generator_while_logo_answer_stays_context_only(self):
        segments = [
            "ドリルの仕組みはとてもシンプルです", "ここがチャックです", "把をしっかり固定します",
            "モーターで高速回転させます", "これで穴は開きます",
        ]
        with SessionLocal() as db:
            source = Source(id=119, source_type="video", filename="DrillMachine.mp4", file_path="/unused/drill.mp4")
            db.add(source)
            db.flush()
            run = VideoPipelineRun(source_id=source.id, input_sha256="e" * 64, config_sha256="f" * 64,
                strategy_version="test", status="completed", stage="completed")
            db.add(run)
            db.flush()
            window = VideoActionWindow(run_id=run.id, source_id=source.id, timestamp_start=0, timestamp_end=8,
                transcript_json="[]", summary="{}", status="analyzed")
            db.add(window)
            db.add(VideoAnalysis(source_id=source.id, status="completed", progress=100))
            for index, text in enumerate(segments):
                db.add(KnowledgeChunk(source_id=source.id, text=text,
                    source_ref=f"DrillMachine.mp4 · 00:{index * 2:02d}", start_seconds=index * 2, end_seconds=index * 2 + 1))
            db.flush()
            db.add(VideoExpertQuestion(run_id=run.id, source_id=source.id, window_id=window.id,
                timestamp_seconds=9.0, question="What is shown at the beginning?",
                answer="This is just an Opening logo", answer_status="answered"))
            db.commit()
            seen = {}

            def refuse_limited(name, excerpts, output_language="en", **kwargs):
                seen.update({"name": name, "excerpts": excerpts, "language": output_language, **kwargs})
                raise ValueError("The source did not support any reliable procedure steps. No draft was created. The component wording needs expert clarification.")

            with patch("app.main.draft_procedure", side_effect=refuse_limited), \
                 patch("app.knowledge.knowledge_index.search", return_value=[]), \
                 patch("app.video_jobs._approved_procedure_references", return_value=[]):
                with self.assertRaises(HTTPException) as caught:
                    create_draft(source.id, "en", db)

            self.assertEqual(caught.exception.status_code, 422)
            self.assertEqual(len(seen["excerpts"]), 5)
            self.assertTrue(all(any(segment in item["text"] for item in seen["excerpts"]) for segment in segments))
            self.assertEqual(seen["context_excerpts"], [])
            self.assertTrue(seen["transcript_only"])
            self.assertNotIn("Opening logo", " ".join(item["text"] for item in seen["excerpts"]))
            self.assertEqual(seen["language"], "en")
            self.assertEqual(db.query(Procedure).filter_by(source_id=source.id).count(), 0)

    def test_logo_only_context_is_not_enough_to_create_a_video_procedure(self):
        with SessionLocal() as db:
            source = Source(id=129, source_type="video", filename="intro.mp4", file_path="/unused/intro.mp4")
            db.add(source)
            db.flush()
            run = VideoPipelineRun(source_id=source.id, input_sha256="9" * 64, config_sha256="8" * 64,
                strategy_version="test", status="completed", stage="completed")
            db.add(run)
            db.flush()
            window = VideoActionWindow(run_id=run.id, source_id=source.id, timestamp_start=0, timestamp_end=4,
                transcript_json="[]", summary="{}", status="analyzed")
            db.add(window)
            db.add(VideoAnalysis(source_id=source.id, status="completed", progress=100))
            db.flush()
            db.add(VideoExpertQuestion(run_id=run.id, source_id=source.id, window_id=window.id,
                timestamp_seconds=1, question="What is this?", answer="This is just an Opening logo", answer_status="answered"))
            db.commit()
            with patch("app.main.draft_procedure") as draft:
                with self.assertRaises(HTTPException) as caught:
                    create_draft(source.id, "en", db)
            self.assertEqual(caught.exception.status_code, 422)
            self.assertIn("no usable stored transcript", caught.exception.detail)
            draft.assert_not_called()
            self.assertEqual(db.query(Procedure).filter_by(source_id=source.id).count(), 0)

    def test_saving_answer_preserves_legacy_approved_proposal_and_procedure(self):
        with SessionLocal() as db:
            source = Source(id=139, source_type="video", filename="legacy.mp4", file_path="/unused/legacy.mp4")
            db.add(source)
            db.flush()
            run = VideoPipelineRun(source_id=source.id, input_sha256="7" * 64, config_sha256="6" * 64,
                strategy_version="test", status="completed", stage="completed")
            db.add(run)
            db.flush()
            window = VideoActionWindow(run_id=run.id, source_id=source.id, timestamp_start=2, timestamp_end=5,
                status="analyzed")
            db.add(window)
            db.flush()
            question = VideoExpertQuestion(run_id=run.id, source_id=source.id, window_id=window.id,
                timestamp_seconds=3, question="What is this moment?", answer="This is just a logo", answer_status="answered")
            procedure = Procedure(title="Legacy approved procedure", description="Historical approval",
                source_id=source.id, status="approved")
            db.add_all([question, procedure])
            db.flush()
            proposal = SOPImprovementProposal(run_id=run.id, source_id=source.id, window_id=window.id,
                procedure_id=procedure.id, title="Legacy question answer", proposed_text="This is just a logo",
                evidence_json=json.dumps({"question_id": question.id}), status="approved")
            db.add(proposal)
            db.commit()
            old_text = proposal.proposed_text

            save_video_expert_answer(question.id,
                VideoExpertAnswerRequest(answer="This remains opening context only.", status="answered"), db)
            db.refresh(proposal)
            db.refresh(procedure)
            self.assertEqual(proposal.status, "approved")
            self.assertEqual(proposal.proposed_text, old_text)
            self.assertEqual(procedure.status, "approved")

    def test_video_draft_refuses_obvious_model_subject_mismatch_without_persisting(self):
        with SessionLocal() as db:
            source, _, _, _ = self._add_video_evidence(db, 114)
            mismatch = json.dumps({"evidence_quality": "sufficient", "insufficient_evidence": False, "reason": "",
                "title": "Set up a Python virtual environment",
                "description": "Create a software environment.", "tools": ["Python"], "warnings": [],
                "steps": [{"instruction": "Open a command line as administrator and navigate directories.",
                    "timestamp": None, "source_ref": None, "evidence_type": "source_supported"}]})
            # The first model response omits step timestamps, so the generator makes
            # one controlled full-procedure retry before the consistency result.
            with patch("app.ai.ollama", side_effect=[mismatch, mismatch,
                    '{"supported":false,"reason":"The source does not support Python setup."}']), \
                 patch("app.knowledge.knowledge_index.search", return_value=[]), \
                 patch("app.video_jobs._approved_procedure_references", return_value=[]):
                with self.assertRaises(HTTPException) as caught:
                    create_draft(source.id, "en", db)
            self.assertEqual(caught.exception.status_code, 422)
            self.assertIn("No draft was created", caught.exception.detail)
            self.assertEqual(db.query(Procedure).filter_by(source_id=source.id).count(), 0)
            self.assertEqual(db.query(VideoSOPDraftVersion).filter_by(source_id=source.id).count(), 0)

    def test_document_source_stays_on_existing_rag_draft_path(self):
        with SessionLocal() as db:
            source = Source(id=124, source_type="document", filename="manual.pdf", file_path="/unused/manual.pdf")
            db.add(source)
            db.add(KnowledgeChunk(source_id=source.id, text="Inspect the joint after cooling.", source_ref="manual.pdf · page 3"))
            db.commit()
            result_data = {"title": "Inspect joint", "description": "From the manual.", "tools": [], "warnings": [],
                "steps": [{"instruction": "Inspect the joint after cooling.", "source_ref": "manual.pdf · page 3"}]}
            with patch("app.main.draft_procedure", return_value=result_data) as draft, \
                 patch("app.main.assemble_transcript_sop_evidence") as video_assembler:
                result = create_draft(source.id, "en", db)
            self.assertEqual(result["status"], "draft")
            draft.assert_called_once()
            video_assembler.assert_not_called()
            self.assertEqual(db.query(VideoSOPDraftVersion).count(), 0)

    def test_regeneration_after_evidence_change_preserves_approved_version(self):
        with SessionLocal() as db:
            source, _, _, _ = self._add_video_evidence(db, 134)
            data = {"title": "Solder the joint", "description": "Limited draft.", "tools": [], "warnings": [],
                "evidence_quality": "limited", "steps": [{"instruction": "Move the iron across the joint.",
                    "timestamp": "00:31", "source_ref": "soldering-demo.mp4 · 00:31", "evidence_type": "source_supported"}]}
            with patch("app.main.draft_procedure", return_value=data) as draft, \
                 patch("app.knowledge.knowledge_index.search", return_value=[]), \
                 patch("app.video_jobs._approved_procedure_references", return_value=[]):
                first = create_draft(source.id, "en", db)
                review_procedure(first["id"], ReviewRequest(action="approve"), db)
                chunk = db.query(KnowledgeChunk).filter_by(source_id=source.id).order_by(KnowledgeChunk.id).first()
                chunk.text += " The operator checks the joint under magnification."
                db.commit()
                second = create_draft(source.id, "en", db)
            self.assertNotEqual(first["id"], second["id"])
            self.assertEqual(db.get(Procedure, first["id"]).status, "approved")
            self.assertEqual(db.get(Procedure, second["id"]).status, "draft")
            versions = db.query(VideoSOPDraftVersion).filter_by(source_id=source.id).order_by(VideoSOPDraftVersion.version).all()
            self.assertEqual([item.version for item in versions], [1, 2])
            self.assertEqual([item.review_status for item in versions], ["approved", "draft"])
            self.assertEqual(draft.call_count, 2)

    def test_sop_fingerprint_changes_when_indexed_evidence_changes(self):
        with SessionLocal() as db:
            source = Source(id=92, source_type="document", filename="manual.pdf", file_path="/unused/manual.pdf")
            db.add(source)
            db.flush()
            chunk = KnowledgeChunk(source_id=source.id, text="Inspect the part.", source_ref="manual.pdf · page 1")
            db.add(chunk)
            db.commit()
            before = _sop_fingerprint(db)
            chunk.text = "Inspect the part and verify its position."
            db.commit()
            after = _sop_fingerprint(db)
        self.assertNotEqual(before, after)

    def test_approved_procedure_step_is_retrievable_with_existing_embedder(self):
        class FakeEmbedder:
            def encode(self, texts, **kwargs):
                return np.asarray([[1.0, 0.0, 0.0] for _ in texts], dtype="float32")
        with SessionLocal() as db:
            source = Source(id=93, source_type="document", filename="manual.pdf", file_path="/unused/manual.pdf")
            db.add(source)
            db.flush()
            procedure = Procedure(title="Position a workpiece", description="Approved procedure",
                steps_json=json.dumps([{"instruction": "Check the workpiece position before operation."}]),
                source_id=source.id, status="approved")
            db.add(procedure)
            db.commit()
            with patch("app.knowledge.get_embedder", return_value=FakeEmbedder()):
                result = _approved_procedure_references(db, "Check workpiece position")
        self.assertEqual(result[0]["procedure_id"], procedure.id)
        self.assertIn("step 1", result[0]["reference"])
        self.assertIn("Check the workpiece", result[0]["excerpt"])

    def test_pipeline_persists_windows_grouped_observations_and_progress(self):
        import subprocess
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as temp:
            video = os.path.join(temp, "clip.mp4")
            subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i",
                "color=c=black:s=320x180:r=10:d=6", "-f", "lavfi", "-i", "color=c=white:s=20x20:r=10:d=6",
                "-filter_complex", "[0:v][1:v]overlay=x='10+60*t':y=50:shortest=1", "-an", "-c:v", "mpeg4", "-y", video],
                check=True, capture_output=True)
            with SessionLocal() as db:
                source = Source(source_type="video", filename="motion.mp4", file_path=video)
                db.add(source)
                db.flush()
                db.add(VideoAnalysis(source_id=source.id, status="pending"))
                db.add(KnowledgeChunk(source_id=source.id, text="Check this component before continuing.",
                    source_ref="motion.mp4 · 00:02", start_seconds=2, end_seconds=4))
                db.commit()
                source_id = source.id

            def screen(_sheet, candidates, _source_id):
                return [{"frame_id": str(row["frame_id"]), "reason": "Potential work evidence", "uncertainty": ""} for row in candidates]

            def analyze(frames, source_id):
                return {"observations": [{"frame_id": str(frame["frame_id"]),
                    "description": "A small component is visible.", "uncertainties": []} for frame in frames],
                    "sequence_summary": "The sampled images show a component.", "visible_changes": [], "uncertainties": [],
                    "source_id": source_id}

            with patch("app.video_jobs.screen_contact_sheet", side_effect=screen), \
                 patch("app.video_jobs.analyze_frame_batch", side_effect=analyze), \
                 patch("app.main.make_chunks", return_value=1), \
                 patch("app.knowledge.knowledge_index.search", return_value=[]), \
                 patch("app.video_jobs._approved_procedure_references", return_value=[]):
                _run(source_id)

            with SessionLocal() as db:
                job = db.query(VideoAnalysis).filter_by(source_id=source_id).one()
                run = db.query(VideoPipelineRun).filter_by(source_id=source_id).one()
                frames = db.query(VideoFrame).filter_by(source_id=source_id).all()
                windows = db.query(VideoActionWindow).filter_by(run_id=run.id).all()
                candidates = db.query(VideoCandidate).filter_by(run_id=run.id).all()
            self.assertIn(job.status, {"completed", "partially_completed"})
            self.assertGreaterEqual(job.progress, 95)
            self.assertTrue(candidates)
            self.assertTrue(frames)
            self.assertTrue(any(frame.analysis_status == "completed" for frame in frames))
            self.assertTrue(windows)
            self.assertTrue(any(window.summary for window in windows))


if __name__ == "__main__":
    unittest.main()
