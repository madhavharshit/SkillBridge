import sys
import tempfile
import unittest
import json
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, "backend")
from app.frame_extraction import extract_frames, extract_frames_at
from app.vision_analysis import (
    FRAME_SCHEMA, _FRAME_ARRAYS, _extract_model_content, _first_complete_sentence, _parse_json_object,
    _call_vision, analyze_frame, analyze_frame_batch, screen_contact_sheet,
)
from app.video_candidates import (scan_motion, build_candidates, build_windows,
    adaptive_motion_scan_fps, adaptive_candidate_baseline_interval, adaptive_contact_budget, plan_detail_frames,
    select_dense_timestamps)
from app.contact_sheets import create_contact_sheet


class FrameExtractionTests(unittest.TestCase):
    def test_sampling_limit_and_timestamps(self):
        with tempfile.TemporaryDirectory() as temp:
            video = Path(temp) / "clip.mp4"
            import subprocess
            subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i", "color=c=blue:s=640x360:d=7:r=10", "-pix_fmt", "yuv420p", str(video)], check=True, capture_output=True)
            frames = extract_frames(str(video), str(Path(temp) / "frames"), interval=2, max_frames=3, max_dimension=480)
            self.assertEqual([f["timestamp_seconds"] for f in frames], [0, 2, 4])
            self.assertLessEqual(len(frames), 3)
            self.assertTrue(all(Path(f["frame_path"]).is_file() for f in frames))

    def test_requested_frame_records_ffmpeg_pts_and_original_request(self):
        import subprocess
        with tempfile.TemporaryDirectory() as temp:
            video = Path(temp) / "clip.mp4"
            subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i",
                            "color=c=blue:s=320x180:d=5:r=10", "-c:v", "mpeg4", str(video)],
                           check=True, capture_output=True)
            frame = extract_frames_at(str(video), str(Path(temp) / "selected"), [1.37], 480)[0]
            self.assertTrue(Path(frame["frame_path"]).is_file())
        self.assertEqual(frame["requested_timestamp_seconds"], 1.37)
        self.assertGreaterEqual(frame["timestamp_seconds"], 1.2)
        self.assertLessEqual(frame["timestamp_seconds"], 1.5)

    def test_missing_video_fails_clearly(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(ValueError, "unavailable"):
                extract_frames(str(Path(temp) / "missing.mp4"), str(Path(temp) / "frames"))


class VisionSchemaTests(unittest.TestCase):
    def test_visual_description_keeps_only_first_complete_sentence(self):
        self.assertEqual(
            _first_complete_sentence("A tool is visible. It may be used for a soldering operation."),
            "A tool is visible.",
        )
        self.assertEqual(
            _first_complete_sentence("ドリルのチャックが見えます。次の動作は不明です。"),
            "ドリルのチャックが見えます。",
        )
        self.assertEqual(_first_complete_sentence("No punctuation in this response"), "No punctuation in this response")

    @patch("app.vision_analysis._call_text")
    def test_frame_uses_direct_text_and_backend_attaches_timestamp(self, call):
        call.return_value = "A drill chuck is visible; the hand-object contact is unclear."
        with tempfile.TemporaryDirectory() as temp:
            image = Path(temp) / "frame.jpg"
            image.write_bytes(b"jpeg")
            result = analyze_frame(str(image), 4.0, 9)
        self.assertEqual(result["source_id"], 9)
        self.assertEqual(result["timestamp_seconds"], 4.0)
        self.assertIn("contact is unclear", result["description"])
        call.assert_called_once()

    @patch("app.vision_analysis._call_text")
    def test_empty_frame_output_is_not_converted_to_an_observation(self, call):
        call.side_effect = ValueError("empty visual response")
        with tempfile.TemporaryDirectory() as temp:
            image = Path(temp) / "frame.jpg"
            image.write_bytes(b"jpeg")
            with self.assertRaisesRegex(ValueError, "empty visual response"):
                analyze_frame(str(image), 0, 1)
        call.assert_called_once()


def valid_frame_content():
    return json.dumps({
        "objects": ["machine"], "observable_actions": [], "machine_states": [],
        "visible_text": [], "safety_observations": [], "uncertainties": [],
        "description": "A machine is visible.",
    })


class VisionResponseParsingTests(unittest.TestCase):
    def test_chat_content_is_unwrapped_from_ollama_envelope(self):
        self.assertEqual(_extract_model_content({"message": {"content": "{\"ok\": true}"}}), '{"ok": true}')

    def test_legacy_generate_envelope_remains_compatible(self):
        self.assertEqual(_extract_model_content({"response": "{\"ok\": true}"}), '{"ok": true}')

    def test_parser_handles_whitespace_markdown_fences_and_surrounding_text(self):
        parsed = _parse_json_object('  ```json\n{"description":"visible"}\n```  ')
        self.assertEqual(parsed["description"], "visible")
        parsed = _parse_json_object('Result follows:\n{"description":"visible"}\nDone.')
        self.assertEqual(parsed["description"], "visible")

    def test_missing_required_fields_and_wrong_types_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            image = Path(temp) / "frame.jpg"
            image.write_bytes(b"jpeg")
            with patch("app.vision_analysis._post_chat", return_value='{"objects":[]}'):
                with self.assertRaisesRegex(ValueError, "omitted required"):
                    _call_vision("describe", [str(image)], FRAME_SCHEMA, _FRAME_ARRAYS, ("description",), 3)
            invalid_type = json.loads(valid_frame_content())
            invalid_type["objects"] = "machine"
            with patch("app.vision_analysis._post_chat", return_value=json.dumps(invalid_type)):
                with self.assertRaisesRegex(ValueError, "incorrect type"):
                    _call_vision("describe", [str(image)], FRAME_SCHEMA, _FRAME_ARRAYS, ("description",), 3)

    def test_empty_malformed_and_truncated_output_get_one_retry_then_actionable_error(self):
        with tempfile.TemporaryDirectory() as temp:
            image = Path(temp) / "frame.jpg"
            image.write_bytes(b"jpeg")
            for bad in ("", "not json", '{"objects": ['):
                with self.subTest(bad=bad), patch("app.vision_analysis._post_chat", side_effect=[bad, bad]) as call:
                    with self.assertRaisesRegex(ValueError, "after one retry"):
                        _call_vision("describe", [str(image)], FRAME_SCHEMA, _FRAME_ARRAYS, ("description",), 4)
                    self.assertEqual(call.call_count, 2)

    def test_valid_output_and_chat_payload(self):
        class Response:
            status_code = 200
            def json(self):
                return {"message": {"content": valid_frame_content()}}

        with tempfile.TemporaryDirectory() as temp:
            image = Path(temp) / "frame.jpg"
            image.write_bytes(b"jpeg bytes")
            with patch("app.vision_analysis.requests.post", return_value=Response()) as post:
                result = _call_vision("describe", [str(image)], FRAME_SCHEMA, _FRAME_ARRAYS, ("description",), 5)
            self.assertEqual(result["objects"], ["machine"])
            args, kwargs = post.call_args
            self.assertTrue(args[0].endswith("/api/chat"))
            import base64
            self.assertEqual(kwargs["json"]["messages"][0]["images"], [base64.b64encode(b"jpeg bytes").decode("ascii")])
            self.assertIsInstance(kwargs["json"]["format"], dict)
            self.assertFalse(kwargs["json"]["stream"])
            self.assertGreaterEqual(kwargs["json"]["options"]["num_predict"], 128)

    def test_truncated_generation_retries_once_with_bounded_output_budget(self):
        class Truncated:
            status_code = 200
            def json(self):
                return {"message": {"content": '{"objects": ['}, "done": True, "done_reason": "length"}
        class Good:
            status_code = 200
            def json(self):
                return {"message": {"content": valid_frame_content()}, "done": True, "done_reason": "stop"}
        with tempfile.TemporaryDirectory() as temp:
            image = Path(temp) / "frame.jpg"
            image.write_bytes(b"jpeg")
            with patch("app.vision_analysis.requests.post", side_effect=[Truncated(), Good()]) as post:
                result = _call_vision("describe", [str(image)], FRAME_SCHEMA, _FRAME_ARRAYS, ("description",), 19)
        self.assertEqual(result["description"], "A machine is visible.")
        self.assertEqual(post.call_count, 2)

    def test_api_errors_and_timeouts_fail_without_retries(self):
        class ErrorResponse:
            status_code = 503
        with tempfile.TemporaryDirectory() as temp:
            image = Path(temp) / "frame.jpg"
            image.write_bytes(b"jpeg")
            with patch("app.vision_analysis.requests.post", return_value=ErrorResponse()) as post:
                with self.assertRaisesRegex(RuntimeError, "HTTP 503"):
                    _call_vision("describe", [str(image)], FRAME_SCHEMA, _FRAME_ARRAYS, ("description",), 6)
                post.assert_called_once()
            import requests
            with patch("app.vision_analysis.requests.post", side_effect=requests.Timeout) as post:
                with self.assertRaisesRegex(RuntimeError, "timed out"):
                    _call_vision("describe", [str(image)], FRAME_SCHEMA, _FRAME_ARRAYS, ("description",), 6)
                post.assert_called_once()

    def test_schema_incompatibility_falls_back_once_to_json_mode(self):
        class BadSchemaResponse:
            status_code = 400
        class GoodResponse:
            status_code = 200
            def json(self):
                return {"message": {"content": valid_frame_content()}}
        with tempfile.TemporaryDirectory() as temp:
            image = Path(temp) / "frame.jpg"
            image.write_bytes(b"jpeg")
            with patch("app.vision_analysis.requests.post", side_effect=[BadSchemaResponse(), GoodResponse()]) as post:
                result = _call_vision("describe", [str(image)], FRAME_SCHEMA, _FRAME_ARRAYS, ("description",), 7)
            self.assertEqual(result["description"], "A machine is visible.")
            self.assertEqual(post.call_count, 2)
            self.assertEqual(post.call_args.kwargs["json"]["format"], "json")

    def test_batch_output_maps_each_requested_frame_id(self):
        with tempfile.TemporaryDirectory() as temp:
            frames = []
            for frame_id in (17, 18):
                image = Path(temp) / f"{frame_id}.jpg"
                image.write_bytes(b"jpeg")
                frames.append({"frame_id": frame_id, "timestamp_seconds": frame_id / 10, "frame_path": str(image)})
            response = {"observations": [{"frame_id": "17", "description": "A component is visible.", "uncertainties": []},
                                         {"frame_id": "18", "description": "A hand is nearby.", "uncertainties": ["contact unclear"]}],
                        "sequence_summary": "The visible images differ.", "visible_changes": [], "uncertainties": []}
            with patch("app.vision_analysis._post_chat", return_value=json.dumps(response)) as call:
                result = analyze_frame_batch(frames, source_id=12)
        self.assertEqual([row["frame_id"] for row in result["observations"]], ["17", "18"])
        self.assertEqual(result["timestamps_seconds"], [1.7, 1.8])
        self.assertEqual(call.call_count, 1)

    def test_batch_with_unrequested_frame_id_is_rejected_after_bounded_retry(self):
        with tempfile.TemporaryDirectory() as temp:
            image = Path(temp) / "17.jpg"
            image.write_bytes(b"jpeg")
            frames = [{"frame_id": 17, "timestamp_seconds": 1.7, "frame_path": str(image)}]
            bad = {"observations": [{"frame_id": "999", "description": "Invented frame", "uncertainties": []}],
                   "sequence_summary": "", "visible_changes": [], "uncertainties": []}
            with patch("app.vision_analysis._post_chat", return_value=json.dumps(bad)) as call:
                with self.assertRaisesRegex(ValueError, "after one retry"):
                    analyze_frame_batch(frames, source_id=12)
        self.assertEqual(call.call_count, 2)


class MotionCandidateTests(unittest.TestCase):
    def make_clip(self, path: Path, filter_value: str):
        import subprocess
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "lavfi",
                        "-i", "color=c=black:s=320x180:r=10:d=3", "-vf", filter_value,
                        "-an", "-c:v", "mpeg4", "-y", str(path)], check=True, capture_output=True)

    def test_static_scene_has_no_motion_peaks_and_keeps_sample_count(self):
        with tempfile.TemporaryDirectory() as temp:
            video = Path(temp) / "static.mp4"
            self.make_clip(video, "null")
            result = scan_motion(str(video), 3, fps=5, width=160, height=90)
        self.assertEqual(result["sample_count"], 15)
        self.assertEqual(result["peaks"], [])

    def test_localized_movement_is_detected_against_mostly_static_background(self):
        import subprocess
        with tempfile.TemporaryDirectory() as temp:
            video = Path(temp) / "localized.mp4"
            subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i",
                            "color=c=black:s=320x180:r=10:d=3", "-f", "lavfi", "-i",
                            "color=c=white:s=20x20:r=10:d=3", "-filter_complex",
                            "[0:v][1:v]overlay=x='10+60*t':y=50:shortest=1", "-an", "-c:v", "mpeg4",
                            "-y", str(video)], check=True, capture_output=True)
            result = scan_motion(str(video), 3, fps=5, width=160, height=90, local_threshold=0.02)
        self.assertTrue(result["peaks"])
        self.assertTrue(all("timestamp_seconds" in item for item in result["peaks"]))
        self.assertTrue(any(item["localized_change"] > item["global_change"] for item in result["peaks"]))

    def test_candidate_windows_retain_baseline_and_transcript_reasons(self):
        candidates = build_candidates(12, [], [{"text": "確認します", "start_seconds": 5, "end_seconds": 6}], 2)
        self.assertTrue(any("uniform_baseline" in item["score_components"] for item in candidates))
        self.assertTrue(any("transcript_boundary" in item["score_components"] for item in candidates))
        windows = build_windows(candidates, 12)
        self.assertTrue(windows)
        dense = select_dense_timestamps([{"start_seconds": 3, "end_seconds": 5, "score": 1}], budget=6, fps=3)
        self.assertEqual(dense[0][0], 3)

    def test_detail_plan_scales_with_duration_and_covers_static_timeline(self):
        windows = [{"start_seconds": start, "end_seconds": start + 8, "score": 0.15,
                    "baseline": True, "priority": False}
                   for start in range(0, 300, 10)]
        plan = plan_detail_frames(windows, duration=300, maximum=24, coverage_interval_seconds=30)
        timestamps = sorted(timestamp for values in plan["timestamps_by_window"].values() for timestamp in values)
        self.assertGreater(plan["budget"], 3)
        self.assertLessEqual(len(timestamps), plan["budget"])
        self.assertLess(timestamps[0], 20)
        self.assertGreater(timestamps[-1], 280)
        gaps = [right - left for left, right in zip(timestamps, timestamps[1:])]
        self.assertLess(max(gaps), 70)

    def test_many_candidate_windows_stay_inside_configured_budget_and_rank_consistently(self):
        duration = 240
        windows = [{"start_seconds": float(start), "end_seconds": float(start + 4),
                    "score": 0.1 + start / duration, "priority": True, "baseline": False}
                   for start in range(0, duration, 8)]
        windows.extend({"start_seconds": float(start), "end_seconds": float(start + 3),
                        "score": 0.15, "priority": False, "baseline": True}
                       for start in range(4, duration, 30))
        plan = plan_detail_frames(windows, duration, maximum=15, coverage_interval_seconds=30)
        selected = set(plan["selected_window_indices"])
        timestamps = [timestamp for values in plan["timestamps_by_window"].values() for timestamp in values]
        self.assertLessEqual(len(timestamps), 15)
        self.assertTrue(selected)
        top_priority = max(range(len(windows)), key=lambda index: windows[index]["score"])
        self.assertIn(top_priority, selected)
        self.assertTrue(set(plan["screened_out_indices"]).isdisjoint(selected))
        self.assertTrue(set(plan["unresolved_important_indices"]).isdisjoint(selected))

    def test_contact_screening_budget_uses_video_length_and_candidate_signals(self):
        candidates = [{"timestamp_seconds": float(index), "priority": index % 4 == 0}
                      for index in range(100)]
        short_budget = adaptive_contact_budget(12, candidates, maximum=27, coverage_interval_seconds=30)
        long_budget = adaptive_contact_budget(600, candidates, maximum=27, coverage_interval_seconds=30)
        self.assertLessEqual(short_budget, long_budget)
        self.assertLessEqual(long_budget, 27)

    def test_coarse_scan_rate_respects_duration_and_source_frame_rate(self):
        short_clip = adaptive_motion_scan_fps(12, 30, maximum_fps=5, target_samples=3600)
        long_clip = adaptive_motion_scan_fps(3600, 30, maximum_fps=5, target_samples=3600)
        low_frame_rate_clip = adaptive_motion_scan_fps(60, 2, maximum_fps=5, target_samples=3600)
        self.assertEqual(short_clip, 5)
        self.assertEqual(long_clip, 1)
        self.assertEqual(low_frame_rate_clip, 2)

    def test_extremely_long_clip_scan_rate_stays_within_sample_ceiling(self):
        duration = 24 * 60 * 60
        fps = adaptive_motion_scan_fps(duration, 30, maximum_fps=5, target_samples=3600)
        self.assertLessEqual(duration * fps, 3600)
        self.assertGreater(fps, 0)

    def test_uniform_candidates_are_bounded_for_very_long_clips(self):
        interval = adaptive_candidate_baseline_interval(100_000, requested_interval=2, maximum_baseline_candidates=1800)
        candidates = build_candidates(100_000, [], [], baseline_interval=interval)
        self.assertLessEqual(len(candidates), 1800)
        self.assertGreater(candidates[-1]["timestamp_seconds"], 99_900)

    def test_baseline_windows_follow_configured_clip_coverage_spacing(self):
        candidates = build_candidates(120, [], [], baseline_interval=2)
        dense_coverage = build_windows(candidates, 120, baseline_coverage_interval_seconds=20)
        sparse_coverage = build_windows(candidates, 120, baseline_coverage_interval_seconds=60)
        self.assertGreater(len(dense_coverage), len(sparse_coverage))
        self.assertLessEqual(dense_coverage[0]["start_seconds"], sparse_coverage[0]["start_seconds"])
        self.assertGreaterEqual(dense_coverage[-1]["end_seconds"], 120)


class ContactSheetTests(unittest.TestCase):
    def test_contact_sheet_keeps_frame_mapping_and_rejects_missing_frames(self):
        import subprocess
        with tempfile.TemporaryDirectory() as temp:
            image = Path(temp) / "frame.jpg"
            subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "lavfi",
                            "-i", "color=c=blue:s=320x180", "-frames:v", "1", str(image)], check=True, capture_output=True)
            output = Path(temp) / "sheet.jpg"
            created = create_contact_sheet([{"frame_id": 31, "timestamp_seconds": 1.25, "frame_path": str(image)}], str(output))
            self.assertEqual(created, str(output))
            self.assertGreater(output.stat().st_size, 100)
            with self.assertRaisesRegex(ValueError, "missing"):
                create_contact_sheet([{"frame_id": 44, "timestamp_seconds": 3, "frame_path": str(Path(temp) / "missing.jpg")}], str(Path(temp) / "bad.jpg"))


if __name__ == "__main__":
    unittest.main()
