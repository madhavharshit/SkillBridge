"""Resumable local video pipeline with cheap motion screening and bounded vision calls."""
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import hashlib
import json
import logging
from pathlib import Path
from threading import Lock
import time

import numpy as np

from . import config
from .config import STORAGE_DIR, VISION_MODEL
from .db import SessionLocal
from .frame_extraction import extract_frames_at
from .contact_sheets import create_contact_sheet
from .models import (Source, VideoAnalysis, VideoFrame, VideoPipelineRun, VideoCandidate,
                     VideoActionWindow, SOPComparison, KnowledgeChunk)
from .video_candidates import (probe_video, scan_motion, build_candidates, build_windows,
                               adaptive_motion_scan_fps, adaptive_contact_budget,
                               adaptive_candidate_baseline_interval,
                               select_contact_candidates, plan_detail_frames)
from .vision_analysis import screen_contact_sheet, analyze_frame_batch, analyze_frame, VisionResponseError

logger = logging.getLogger(__name__)
_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="video-analysis")
_lock = Lock()
_running: set[int] = set()


@lru_cache(maxsize=256)
def _approved_procedure_vector(content_key: str, text: str):
    """Reuse the existing local embedding model and cache each procedure step by content version."""
    from .knowledge import get_embedder
    return get_embedder().encode([text], convert_to_numpy=True, normalize_embeddings=True)[0].astype("float32")


def _approved_procedure_references(db, query: str, top_k: int = 3) -> list[dict]:
    from .knowledge import get_embedder
    from .models import Procedure
    procedures = db.query(Procedure).filter_by(status="approved").order_by(Procedure.id).all()
    passages = []
    for procedure in procedures:
        try:
            steps = json.loads(procedure.steps_json or "[]")
        except (TypeError, json.JSONDecodeError):
            steps = []
        if not isinstance(steps, list):
            steps = []
        common = f"{procedure.title}. {procedure.description}".strip()
        content_key_base = hashlib.sha256((str(procedure.updated_at) + common + procedure.steps_json).encode()).hexdigest()
        supported_steps = [step for step in steps if isinstance(step, dict) and isinstance(step.get("instruction"), str) and step["instruction"].strip()]
        if supported_steps:
            for index, step in enumerate(supported_steps, start=1):
                passage = f"{common}. Step {index}: {step['instruction']}"
                passages.append((content_key_base + f":{index}", procedure, index, passage))
        elif common:
            passages.append((content_key_base + ":description", procedure, None, common))
    if not passages or not query.strip():
        return []
    query_vector = get_embedder().encode([query], convert_to_numpy=True, normalize_embeddings=True)[0].astype("float32")
    scored = []
    for cache_key, procedure, step_number, text in passages:
        vector = _approved_procedure_vector(cache_key, text)
        score = float(np.dot(query_vector, vector))
        scored.append((score, procedure, step_number, text))
    return [{"procedure_id": procedure.id, "title": procedure.title,
             "reference": f"Approved procedure {procedure.id}" + (f" · step {step_number}" if step_number else " · description"),
             "excerpt": text[:500], "similarity": round(score, 4)}
            for score, procedure, step_number, text in sorted(scored, key=lambda item: item[0], reverse=True)[:top_k]]


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _configuration():
    keys = ("VISION_MODEL", "VISION_TIMEOUT_SECONDS", "VISION_MAX_OUTPUT_TOKENS", "VIDEO_FRAME_INTERVAL", "VIDEO_MOTION_SCAN_FPS", "VIDEO_MOTION_SCAN_MAX_SAMPLES", "VIDEO_BASELINE_CANDIDATE_LIMIT",
            "VIDEO_MOTION_WIDTH", "VIDEO_MOTION_HEIGHT", "VIDEO_MOTION_LOCAL_THRESHOLD",
            "VIDEO_WINDOW_RADIUS_SECONDS", "VIDEO_WINDOW_MERGE_GAP_SECONDS", "VIDEO_WINDOW_MAX_SECONDS",
            "VIDEO_FRAME_MAX_DIMENSION", "VIDEO_DETAIL_FPS", "VIDEO_DETAIL_FRAME_BUDGET", "VIDEO_CONTACT_CANDIDATE_LIMIT",
            "VIDEO_CONTACT_SHEET_SIZE", "VIDEO_DETAIL_COVERAGE_INTERVAL_SECONDS",
            "VIDEO_VISION_BATCH_SIZE", "VIDEO_STRATEGY_VERSION", "VIDEO_PROMPT_VERSION")
    return {key: getattr(config, key) for key in keys}


def _sop_fingerprint(db) -> str:
    """Fingerprint indexed manuals and approved procedures for comparison cache invalidation."""
    from .models import Procedure
    documents = db.query(Source).filter_by(source_type="document").order_by(Source.id).all()
    chunks = []
    for document in documents:
        for chunk in (db.query(KnowledgeChunk).filter_by(source_id=document.id)
                      .order_by(KnowledgeChunk.id).all()):
            chunks.append((document.id, chunk.id, chunk.source_ref, chunk.text))
    procedures = db.query(Procedure).filter_by(status="approved").order_by(Procedure.id).all()
    approved = [(item.id, item.updated_at.isoformat() if item.updated_at else "", item.title,
                 item.description, item.tools_json, item.warnings_json, item.steps_json) for item in procedures]
    raw = json.dumps({"chunks": chunks, "procedures": approved}, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _set_stage(db, job, run, stage: str, progress: int, metrics: dict | None = None, status: str = "running"):
    job.status, job.progress = status, max(0, min(99, int(progress)))
    run.status, run.stage = status, stage
    if metrics is not None:
        run.metrics_json = json.dumps(metrics, ensure_ascii=False)
    db.commit()


def _transcript_for_source(db, source_id: int) -> list[dict]:
    chunks = (db.query(KnowledgeChunk).filter_by(source_id=source_id)
              .filter(KnowledgeChunk.start_seconds.is_not(None))
              .order_by(KnowledgeChunk.start_seconds, KnowledgeChunk.id).all())
    return [{"chunk_id": chunk.id, "text": chunk.text, "start_seconds": chunk.start_seconds,
             "end_seconds": chunk.end_seconds if chunk.end_seconds is not None else chunk.start_seconds,
             "source_ref": chunk.source_ref} for chunk in chunks
            if not (chunk.source_ref or "").startswith("visual frame ·")]


def _candidate_objects(db, run_id: int):
    return db.query(VideoCandidate).filter_by(run_id=run_id).order_by(VideoCandidate.timestamp_seconds).all()


def _frame_at_timestamp(db, source_id: int, run_id: int, timestamp: float):
    candidate = (db.query(VideoCandidate).filter_by(run_id=run_id, source_id=source_id)
                 .filter(VideoCandidate.frame_id.is_not(None))
                 .order_by(VideoCandidate.timestamp_seconds).all())
    for item in candidate:
        if abs(item.timestamp_seconds - timestamp) <= 0.02:
            frame = db.get(VideoFrame, item.frame_id)
            if frame and Path(frame.frame_path).is_file():
                return frame
    return None


def _ensure_frame_rows(db, source, run, rows: list[VideoCandidate], directory: Path) -> dict[int, VideoFrame]:
    missing = []
    output = {}
    for row in rows:
        frame = db.get(VideoFrame, row.frame_id) if row.frame_id else None
        if frame and Path(frame.frame_path).is_file():
            output[row.id] = frame
        else:
            missing.append(row)
    if missing:
        extracted = extract_frames_at(source.file_path, str(directory), [row.timestamp_seconds for row in missing],
                                      config.VIDEO_FRAME_MAX_DIMENSION)
        by_request = {round(item["requested_timestamp_seconds"], 2): item for item in extracted}
        metadata = json.loads(run.metadata_json or "{}")
        source_fps = float(metadata.get("frame_rate") or 0)
        for row in missing:
            item = by_request[round(row.timestamp_seconds, 2)]
            frame = VideoFrame(source_id=source.id, timestamp_seconds=item["timestamp_seconds"],
                               frame_number=int(round(row.timestamp_seconds * source_fps)) if source_fps > 0 else 0,
                               frame_path=item["frame_path"], analysis_status="pending")
            db.add(frame)
            db.flush()
            row.frame_id = frame.id
            output[row.id] = frame
        db.commit()
    return output


def _run(source_id: int, force: bool = False):
    db = SessionLocal()
    started = time.monotonic()
    try:
        job = db.query(VideoAnalysis).filter_by(source_id=source_id).one()
        source = db.get(Source, source_id)
        if not source or source.source_type != "video":
            raise ValueError("Selected source is not a video.")
        content_hash = _sha256_file(source.file_path)
        transcript = _transcript_for_source(db, source_id)
        config_values = _configuration()
        transcript_hash = hashlib.sha256(json.dumps(transcript, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
        config_hash = hashlib.sha256(json.dumps({"pipeline": config_values, "transcript_sha256": transcript_hash}, sort_keys=True).encode()).hexdigest()
        run = None if force else (db.query(VideoPipelineRun).filter_by(source_id=source_id, input_sha256=content_hash,
                                                                        config_sha256=config_hash)
                                  .order_by(VideoPipelineRun.id.desc()).first())
        if not run:
            run = VideoPipelineRun(source_id=source_id, input_sha256=content_hash, config_sha256=config_hash,
                                   strategy_version=config.VIDEO_STRATEGY_VERSION, status="pending", stage="queued")
            db.add(run)
            db.commit()
            db.refresh(run)
        latest = db.query(VideoPipelineRun).filter_by(source_id=source_id).order_by(VideoPipelineRun.id.desc()).first()
        if run.status == "completed" and latest and run.id == latest.id and not force:
            job.status, job.progress, job.error, job.model_name = "completed", 100, "", VISION_MODEL
            db.commit()
            return
        run.status, run.error = "running", ""
        job.error = ""
        db.commit()
        run_dir = STORAGE_DIR / "frames" / str(source_id) / "runs" / str(run.id)
        frame_dir = run_dir / "frames"
        warnings = json.loads(run.warnings_json or "[]")
        metrics = json.loads(run.metrics_json or "{}")

        if not run.metadata_json or run.metadata_json == "{}":
            _set_stage(db, job, run, "video_inspection", 2, metrics)
            metadata = probe_video(source.file_path)
            run.metadata_json = json.dumps(metadata)
            db.commit()
        else:
            metadata = json.loads(run.metadata_json)
        duration = float(metadata["duration_seconds"])

        candidates = _candidate_objects(db, run.id)
        if not candidates:
            _set_stage(db, job, run, "motion_and_transcript_scan", 7, metrics)
            motion_fps = adaptive_motion_scan_fps(duration, float(metadata.get("frame_rate") or 0),
                config.VIDEO_MOTION_SCAN_FPS, config.VIDEO_MOTION_SCAN_MAX_SAMPLES)
            scan = scan_motion(source.file_path, duration, motion_fps,
                               config.VIDEO_MOTION_WIDTH, config.VIDEO_MOTION_HEIGHT,
                               config.VIDEO_MOTION_LOCAL_THRESHOLD)
            baseline_interval = adaptive_candidate_baseline_interval(
                duration, config.VIDEO_FRAME_INTERVAL, config.VIDEO_BASELINE_CANDIDATE_LIMIT)
            candidates_data = build_candidates(duration, scan["peaks"], transcript, baseline_interval)
            metrics.update({"motion_scan_samples": scan["sample_count"], "motion_scan_fps": motion_fps,
                            "candidate_baseline_interval_seconds": baseline_interval,
                            "motion_peaks": len(scan["peaks"]),
                            "baseline_candidates": sum("uniform_baseline" in row["score_components"] for row in candidates_data),
                            "candidate_count": len(candidates_data), "transcript_segments": len(transcript)})
            for item in candidates_data:
                db.add(VideoCandidate(run_id=run.id, source_id=source_id,
                    timestamp_seconds=item["timestamp_seconds"], score=item["score"],
                    score_components_json=json.dumps(item["score_components"]), reasons_json=json.dumps(item["reasons"])))
            run.metrics_json = json.dumps(metrics)
            run.stage = "candidate_window_selection"
            db.commit()
            candidates = _candidate_objects(db, run.id)

        windows = db.query(VideoActionWindow).filter_by(run_id=run.id).order_by(VideoActionWindow.timestamp_start).all()
        if not windows:
            _set_stage(db, job, run, "candidate_window_selection", 18, metrics)
            candidate_dicts = [{"timestamp_seconds": item.timestamp_seconds, "score": item.score,
                                "score_components": json.loads(item.score_components_json),
                                "reasons": json.loads(item.reasons_json)} for item in candidates]
            window_data = build_windows(candidate_dicts, duration, config.VIDEO_WINDOW_RADIUS_SECONDS,
                                        config.VIDEO_WINDOW_MERGE_GAP_SECONDS, config.VIDEO_WINDOW_MAX_SECONDS,
                                        config.VIDEO_DETAIL_COVERAGE_INTERVAL_SECONDS)
            for window in window_data:
                overlapping = [segment for segment in transcript
                               if segment["start_seconds"] <= window["end_seconds"] and segment["end_seconds"] >= window["start_seconds"]]
                rows = [candidate for candidate in candidates if candidate.timestamp_seconds in window["candidate_timestamps"]]
                db.add(VideoActionWindow(run_id=run.id, source_id=source_id,
                    timestamp_start=window["start_seconds"], timestamp_end=window["end_seconds"],
                    candidate_ids_json=json.dumps([row.id for row in rows]),
                    transcript_json=json.dumps(overlapping, ensure_ascii=False),
                    selection_reason="; ".join(window["reasons"]), status="selected"))
            metrics["windows_selected"] = len(window_data)
            run.metrics_json = json.dumps(metrics)
            db.commit()
            windows = db.query(VideoActionWindow).filter_by(run_id=run.id).order_by(VideoActionWindow.timestamp_start).all()

        contact_rows_raw = [
            {"id": row.id, "timestamp_seconds": row.timestamp_seconds, "score": row.score,
             "score_components": json.loads(row.score_components_json),
             "priority": any(key != "uniform_baseline" for key in json.loads(row.score_components_json))}
            for row in candidates
        ]
        contact_budget = adaptive_contact_budget(duration, contact_rows_raw,
            config.VIDEO_CONTACT_CANDIDATE_LIMIT, config.VIDEO_DETAIL_COVERAGE_INTERVAL_SECONDS)
        contact_rows = select_contact_candidates(contact_rows_raw, contact_budget)
        selected_contact_ids = {row["id"] for row in contact_rows}
        for row in candidates:
            row.is_contact_sample = row.id in selected_contact_ids
        db.commit()
        contact_candidates = [row for row in candidates if row.is_contact_sample]
        contact_frames = _ensure_frame_rows(db, source, run, contact_candidates, frame_dir)
        for row in contact_candidates:
            row.screening_status = "pending" if row.screening_status == "pending" else row.screening_status
        db.commit()
        _set_stage(db, job, run, "contact_sheet_screening", 28, metrics)
        contact_selected = []
        for sheet_index in range(0, len(contact_candidates), config.VIDEO_CONTACT_SHEET_SIZE):
            group = contact_candidates[sheet_index:sheet_index + config.VIDEO_CONTACT_SHEET_SIZE]
            if group and all(row.screening_status in {"completed", "fallback"} for row in group):
                contact_selected.extend(row for row in group if row.selected_for_detail)
                continue
            sheet_frames = [{"frame_id": row.frame_id, "timestamp_seconds": row.timestamp_seconds,
                             "frame_path": contact_frames[row.id].frame_path} for row in group]
            sheet_path = run_dir / "contact-sheets" / f"sheet-{sheet_index // config.VIDEO_CONTACT_SHEET_SIZE + 1:02d}.jpg"
            try:
                create_contact_sheet(sheet_frames, str(sheet_path), columns=3)
                selected = screen_contact_sheet(str(sheet_path), sheet_frames, source_id)
                selected_map = {item["frame_id"]: item for item in selected}
                for row in group:
                    answer = selected_map.get(str(row.frame_id))
                    row.screening_status = "completed"
                    row.selected_for_detail = answer is not None
                    if answer:
                        row.selection_reason = answer["reason"][:500]
                        if answer["uncertainty"]:
                            row.selection_reason += " · uncertainty: " + answer["uncertainty"][:300]
                        contact_selected.append(row)
                metrics["contact_sheets_processed"] = metrics.get("contact_sheets_processed", 0) + 1
            except Exception as exc:
                logger.exception("Contact-sheet screening failed; using deterministic fallback: source_id=%s sheet=%s", source_id, sheet_index // config.VIDEO_CONTACT_SHEET_SIZE + 1)
                fallback = sorted(group, key=lambda row: row.score, reverse=True)[:max(1, min(3, len(group)))]
                for row in group:
                    row.screening_status = "fallback"
                    row.selected_for_detail = row in fallback
                    row.selection_reason = "deterministic motion/transcript/baseline fallback; screening failed: " + str(exc)[:200]
                contact_selected.extend(fallback)
                warnings.append(f"Contact sheet {sheet_index // config.VIDEO_CONTACT_SHEET_SIZE + 1} used deterministic fallback.")
            run.metrics_json = json.dumps(metrics)
            run.warnings_json = json.dumps(warnings)
            job.progress = 28 + int(5 * (sheet_index // config.VIDEO_CONTACT_SHEET_SIZE + 1) /
                                    max(1, (len(contact_candidates) + config.VIDEO_CONTACT_SHEET_SIZE - 1) // config.VIDEO_CONTACT_SHEET_SIZE))
            db.commit()
        if not contact_selected and candidates:
            contact_selected = sorted(contact_candidates or candidates, key=lambda row: row.score, reverse=True)[:min(3, len(candidates))]
            for row in contact_selected:
                row.selected_for_detail = True
                row.screening_status = "fallback"
                row.selection_reason = "deterministic coverage fallback; contact sheet selected no candidates"
            warnings.append("No contact-sheet candidates were selected; deterministic coverage fallback retained the highest-scoring samples.")
            db.commit()

        selected_contact_times = {round(row.timestamp_seconds, 2) for row in contact_selected}
        window_dicts = []
        for window in windows:
            inside = [row for row in candidates
                      if window.timestamp_start - 0.02 <= row.timestamp_seconds <= window.timestamp_end + 0.02]
            reasons = {reason for reason in window.selection_reason.split("; ") if reason}
            screened_action = any(round(row.timestamp_seconds, 2) in selected_contact_times for row in inside)
            score = max((row.score for row in inside), default=0.15)
            window_dicts.append({"start_seconds": window.timestamp_start, "end_seconds": window.timestamp_end,
                "score": score, "baseline": "uniform_baseline" in reasons,
                "priority": screened_action or bool(reasons.intersection(
                    {"localized_motion", "scene_boundary", "procedural transcript cue"}))})
        detail_plan = plan_detail_frames(window_dicts, duration, config.VIDEO_DETAIL_FRAME_BUDGET,
                                         config.VIDEO_DETAIL_COVERAGE_INTERVAL_SECONDS)
        selected_indices = set(detail_plan["selected_window_indices"])
        selected_windows = [window for index, window in enumerate(windows) if index in selected_indices]
        dense_by_window = detail_plan["timestamps_by_window"]
        dense_times = {ts for values in dense_by_window.values() for ts in values}
        if not dense_times and candidates:
            dense_times = {candidates[0].timestamp_seconds}

        # Persist selected samples as candidate records so retries can reuse extracted files and outputs.
        timestamp_candidates = {round(row.timestamp_seconds, 2): row for row in candidates}
        detail_rows = []
        for ts in sorted(dense_times):
            row = timestamp_candidates.get(round(ts, 2))
            if row is None:
                window_index = next((index for index, window in enumerate(selected_windows)
                                     if window.timestamp_start <= ts <= window.timestamp_end), None)
                window_score = window_dicts[window_index]["score"] if window_index is not None else 0.15
                row = VideoCandidate(run_id=run.id, source_id=source_id, timestamp_seconds=round(ts, 2), score=window_score,
                    score_components_json=json.dumps({"dense_window_sample": window_score}),
                    reasons_json=json.dumps(["before/during/after dense window sample"]), selected_for_detail=True,
                    screening_status="not_screened", selection_reason="dense sample selected to preserve window chronology")
                db.add(row)
                db.flush()
                candidates.append(row)
                timestamp_candidates[round(ts, 2)] = row
            row.selected_for_detail = True
            detail_rows.append(row)
        db.commit()
        detail_frames = _ensure_frame_rows(db, source, run, detail_rows, frame_dir)
        detailed_window_ids = {window.id for window in selected_windows}
        unresolved_indices = set(detail_plan["unresolved_important_indices"])
        unresolved_windows = [windows[index] for index in sorted(unresolved_indices)]
        for index, window in enumerate(windows):
            if index in unresolved_indices:
                window.status = "unresolved_important"
            elif index in selected_indices:
                window.status = "selected_for_detail"
            else:
                window.status = "screened_out"
        if unresolved_windows:
            warning = (f"{len(unresolved_windows)} candidate window(s) were not included in detailed visual analysis "
                       "within the current frame budget or screening selection.")
            if warning not in warnings:
                warnings.append(warning)
        metrics.update({"contact_candidates": len(contact_candidates), "contact_selected": len(contact_selected),
                        "contact_budget_ceiling": config.VIDEO_CONTACT_CANDIDATE_LIMIT,
                        "contact_budget_used": contact_budget,
                        "detail_frame_budget_ceiling": config.VIDEO_DETAIL_FRAME_BUDGET,
                        "detail_frame_budget_used": len(detail_rows),
                        "detail_coverage_interval_seconds": config.VIDEO_DETAIL_COVERAGE_INTERVAL_SECONDS,
                        "detailed_frames_selected": len(detail_rows), "detailed_frames_completed": sum(detail_frames[row.id].analysis_status == "completed" for row in detail_rows),
                        "detailed_windows_covered": len(detailed_window_ids), "candidate_windows_not_detailed": len(unresolved_windows),
                        "candidate_windows_screened_out": len(detail_plan["screened_out_indices"]),
                        "vision_request_target": len(detail_rows)})
        _set_stage(db, job, run, "detailed_visual_analysis", 40, metrics)

        pending_frames = [detail_frames[row.id] for row in detail_rows if detail_frames[row.id].analysis_status != "completed"]
        groups, group_windows = [], []
        assigned_frame_ids = set()
        for window in selected_windows:
            window_frames = [frame for frame in pending_frames
                             if frame.id not in assigned_frame_ids and
                             window.timestamp_start - 0.02 <= frame.timestamp_seconds <= window.timestamp_end + 0.02]
            for start in range(0, len(window_frames), config.VIDEO_VISION_BATCH_SIZE):
                group = window_frames[start:start + config.VIDEO_VISION_BATCH_SIZE]
                groups.append(group)
                group_windows.append(window)
                assigned_frame_ids.update(frame.id for frame in group)
        unassigned = [frame for frame in pending_frames if frame.id not in assigned_frame_ids]
        for start in range(0, len(unassigned), config.VIDEO_VISION_BATCH_SIZE):
            groups.append(unassigned[start:start + config.VIDEO_VISION_BATCH_SIZE])
            group_windows.append(None)
        failed_frame_ids = set()
        for group_index, group in enumerate(groups):
            context = [{"frame_id": frame.id, "timestamp_seconds": frame.timestamp_seconds, "frame_path": frame.frame_path}
                       for frame in group]
            logger.info("Vision batch started: source_id=%s batch=%s/%s frame_ids=%s model=%s",
                        source_id, group_index + 1, len(groups), [frame.id for frame in group], VISION_MODEL)
            try:
                result = analyze_frame_batch(context, source_id)
                obs_by_id = {int(item["frame_id"]): item for item in result["observations"]}
                for frame in group:
                    observation = obs_by_id[frame.id]
                    frame.analysis_json = json.dumps({"description": observation["description"],
                        "uncertainties": observation["uncertainties"], "objects": [], "observable_actions": [],
                        "machine_states": [], "visible_text": [], "safety_observations": [],
                        "source_id": source_id, "timestamp_seconds": frame.timestamp_seconds,
                        "evidence_type": "direct_visual_observation", "model_name": VISION_MODEL}, ensure_ascii=False)
                    frame.analysis_status = "completed"
                    frame.model_name = VISION_MODEL
                window = group_windows[group_index]
                if window:
                    previous_summary = json.loads(window.summary or "{}") if window.summary else {}
                    previous_summary.setdefault("sequence_summaries", [])
                    if result["sequence_summary"]:
                        previous_summary["sequence_summaries"].append(result["sequence_summary"])
                    previous_summary["sequence_summary"] = " ".join(previous_summary["sequence_summaries"])
                    previous_summary["visible_changes"] = list(dict.fromkeys(
                        previous_summary.get("visible_changes", []) + result["visible_changes"]))
                    previous_summary["uncertainties"] = list(dict.fromkeys(
                        previous_summary.get("uncertainties", []) + result["uncertainties"]))
                    previous_summary["frame_ids"] = sorted(set(previous_summary.get("frame_ids", []) + [frame.id for frame in group]))
                    previous_summary["timestamps"] = sorted(set(previous_summary.get("timestamps", []) + [frame.timestamp_seconds for frame in group]))
                    window.summary = json.dumps(previous_summary, ensure_ascii=False)
                    window.status = "analyzed"
                metrics["detailed_frames_completed"] = metrics.get("detailed_frames_completed", 0) + len(group)
                metrics["vision_requests_completed"] = metrics.get("vision_requests_completed", 0) + 1
                logger.info("Vision batch completed: source_id=%s batch=%s/%s frames=%s",
                            source_id, group_index + 1, len(groups), len(group))
            except Exception as exc:
                logger.exception("Vision batch failed; retaining other batches: source_id=%s batch=%s frame_ids=%s",
                                 source_id, group_index + 1, [frame.id for frame in group])
                fallback_successes = 0
                for frame in group:
                    if isinstance(exc, VisionResponseError):
                        try:
                            observation = analyze_frame(frame.frame_path, frame.timestamp_seconds, source_id)
                            frame.analysis_json = json.dumps(observation | {"evidence_type": "direct_visual_observation",
                                "model_name": VISION_MODEL, "analysis_mode": "single_frame_fallback"}, ensure_ascii=False)
                            frame.analysis_status, frame.model_name = "completed", VISION_MODEL
                            fallback_successes += 1
                            continue
                        except Exception as frame_exc:
                            error_text = str(frame_exc)
                    else:
                        error_text = str(exc)
                    if frame.analysis_status != "completed":
                        frame.analysis_status = "failed"
                        frame.analysis_json = json.dumps({"error": error_text[:500], "evidence_type": "unresolved"})
                        failed_frame_ids.add(frame.id)
                window = group_windows[group_index]
                if window:
                    previous_summary = json.loads(window.summary or "{}") if window.summary else {}
                    previous_summary["sequence_summary"] = ""
                    previous_summary["visible_changes"] = []
                    previous_summary["uncertainties"] = list(dict.fromkeys(previous_summary.get("uncertainties", []) + [
                        "Batch sequence response failed; individual still frames were analyzed, so motion continuity is not established."]))
                    previous_summary["frame_ids"] = sorted(set(previous_summary.get("frame_ids", []) + [frame.id for frame in group if frame.analysis_status == "completed"]))
                    previous_summary["timestamps"] = sorted(set(previous_summary.get("timestamps", []) + [frame.timestamp_seconds for frame in group if frame.analysis_status == "completed"]))
                    window.summary = json.dumps(previous_summary, ensure_ascii=False)
                    window.status = "analyzed" if all(frame.analysis_status == "completed" for frame in group) else "partially_analyzed"
                metrics["vision_requests_failed"] = metrics.get("vision_requests_failed", 0) + 1
                metrics["individual_fallback_requests_completed"] = metrics.get("individual_fallback_requests_completed", 0) + fallback_successes
                metrics["vision_requests_completed"] = metrics.get("vision_requests_completed", 0) + fallback_successes
                warnings.append(f"Visual batch {group_index + 1} did not return valid structured output; {fallback_successes} frame(s) were retried individually. Sequence continuity is unavailable for this batch.")
            job.progress = 40 + int(40 * (group_index + 1) / max(len(groups), 1))
            run.metrics_json, run.warnings_json = json.dumps(metrics), json.dumps(warnings)
            db.commit()

        # Persist candidate visual excerpts for the existing source RAG index, idempotently per timestamp.
        _set_stage(db, job, run, "evidence_alignment", 84, metrics)
        for window in selected_windows:
            window_frames = [detail_frames[row.id] for row in detail_rows
                             if window.timestamp_start - 0.02 <= row.timestamp_seconds <= window.timestamp_end + 0.02]
            window.frame_ids_json = json.dumps([frame.id for frame in window_frames])
            if window_frames and any(frame.analysis_status != "completed" for frame in window_frames):
                window.status = "partially_analyzed"
            elif window_frames:
                window.status = "analyzed"
        failed_window_ids = {window.id for window in selected_windows if window.status == "partially_analyzed"}
        known_refs = {ref for (ref,) in db.query(KnowledgeChunk.source_ref).filter_by(source_id=source_id).all()}
        chunk_entries = []
        for row in detail_rows:
            frame = detail_frames[row.id]
            if frame.analysis_status != "completed":
                continue
            observation = json.loads(frame.analysis_json or "{}")
            text = observation.get("description", "")
            ref = f"visual frame · {frame.timestamp_seconds:.3f}s · F{frame.id} · run {run.id}"
            if text and ref not in known_refs:
                chunk_entries.append({"text": f"DIRECT VISUAL OBSERVATION (not a diagnosis or safety verification): {text}",
                    "source_ref": ref, "start_seconds": int(frame.timestamp_seconds), "end_seconds": int(frame.timestamp_seconds)})
        if chunk_entries:
            from .main import make_chunks
            make_chunks(db, source, chunk_entries)

        _set_stage(db, job, run, "sop_comparison", 90, metrics)
        comparisons = db.query(SOPComparison).filter_by(run_id=run.id).all()
        sop_fingerprint = _sop_fingerprint(db)
        comparison_windows = {item.window_id for item in comparisons}
        comparisons_valid = (len(comparisons) == len(selected_windows) and
            comparison_windows == {window.id for window in selected_windows} and
            all(json.loads(item.evidence_json or "{}").get("sop_fingerprint") == sop_fingerprint for item in comparisons))
        if comparisons and not comparisons_valid:
            db.query(SOPComparison).filter_by(run_id=run.id).delete(synchronize_session=False)
            for window in windows:
                window.comparison_json = "{}"
            db.commit()
            comparisons = []
        if not comparisons:
            from .knowledge import knowledge_index
            from .models import Procedure as ProcedureModel
            from .models import Source as SourceModel
            document_sources = db.query(SourceModel).filter_by(source_type="document").count()
            for window in selected_windows:
                summary_obj = json.loads(window.summary or "{}") if window.summary.startswith("{") else {}
                transcript_items = json.loads(window.transcript_json or "[]")
                query = " ".join([summary_obj.get("sequence_summary", ""),
                                   " ".join(summary_obj.get("visible_changes", [])),
                                   " ".join(item["text"] for item in transcript_items)])[:1000]
                hits = knowledge_index.search(db, query, top_k=3, source_type="document") if query.strip() else []
                refs = []
                for chunk in hits:
                    doc = db.get(SourceModel, chunk.source_id)
                    refs.append({"source_id": chunk.source_id, "filename": doc.filename if doc else "Unknown",
                                 "reference": chunk.source_ref, "chunk_id": chunk.id, "excerpt": chunk.text[:500]})
                if query.strip():
                    try:
                        refs.extend(_approved_procedure_references(db, query, top_k=3))
                    except Exception as exc:
                        logger.warning("Approved procedure comparison retrieval failed; marking comparison uncertain: source_id=%s window_id=%s error_type=%s",
                                       source_id, window.id, type(exc).__name__)
                        warnings.append("Approved procedure retrieval was unavailable for one or more comparisons; treat those comparisons as incomplete.")
                approved_count = db.query(ProcedureModel).filter_by(status="approved").count()
                if not document_sources and not approved_count:
                    relationship, explanation = "no_sop_available", "No indexed SOP/manual or approved procedure was available for comparison. No absence claim is made."
                elif refs:
                    relationship, explanation = "human_comparison_required", "Candidate manual/SOP passages were retrieved using the existing local RAG index and cached procedure-step embeddings. Similarity is not proof of equivalence; compare each passage with the timestamped evidence."
                else:
                    relationship, explanation = "ambiguous", "No relevant SOP passage was retrieved. This does not establish that the action is undocumented."
                comparison = SOPComparison(run_id=run.id, source_id=source_id, window_id=window.id,
                    relationship=relationship, explanation=explanation, references_json=json.dumps(refs, ensure_ascii=False),
                    evidence_json=json.dumps({"window_summary": summary_obj, "transcript": transcript_items,
                                              "sop_fingerprint": sop_fingerprint}, ensure_ascii=False))
                db.add(comparison)
                window.comparison_json = json.dumps({"relationship": relationship, "explanation": explanation, "references": refs}, ensure_ascii=False)
            db.commit()

        metrics.update({"failed_detailed_frames": len(failed_frame_ids),
                        "detailed_frames_completed": sum(detail_frames[row.id].analysis_status == "completed" for row in detail_rows),
                        "detailed_windows_covered": sum(window.status == "analyzed" for window in selected_windows),
                        "candidate_windows_not_detailed": len(unresolved_windows) + len(failed_window_ids),
                        "candidate_windows_failed": len(failed_window_ids),
                        "total_elapsed_seconds": round(time.monotonic() - started, 2),
                        "analysis_calls": metrics.get("vision_requests_completed", 0) + metrics.get("vision_requests_failed", 0)})
        run.metrics_json, run.warnings_json = json.dumps(metrics), json.dumps(warnings)
        has_gaps = bool(failed_frame_ids or unresolved_windows)
        run.status = "partially_completed" if has_gaps else "completed"
        run.stage = "completed_with_gaps" if has_gaps else "completed"
        gap_messages = []
        if failed_frame_ids:
            gap_messages.append("Some independent frame batches failed; completed evidence was preserved.")
        if unresolved_windows:
            gap_messages.append(f"{len(unresolved_windows)} high-priority candidate window(s) remain outside detailed visual analysis under the configured frame budget.")
        run.error = " ".join(gap_messages)
        job.status = run.status
        job.progress = 95 if has_gaps else 100
        job.error = run.error
        job.model_name = VISION_MODEL
        db.commit()
        logger.info("Video analysis pipeline finished: source_id=%s run_id=%s status=%s metrics=%s warnings=%s",
                    source_id, run.id, run.status, run.metrics_json, run.warnings_json)
    except Exception as exc:
        logger.exception("Video analysis pipeline failed: source_id=%s", source_id)
        try:
            job = db.query(VideoAnalysis).filter_by(source_id=source_id).one_or_none()
            if job:
                job.status, job.error = "failed", str(exc)[:1000]
            run = db.query(VideoPipelineRun).filter_by(source_id=source_id).order_by(VideoPipelineRun.id.desc()).first()
            if run:
                run.status, run.stage, run.error = "failed", run.stage, str(exc)[:1000]
                metrics = json.loads(run.metrics_json or "{}")
                metrics["total_elapsed_seconds"] = round(time.monotonic() - started, 2)
                run.metrics_json = json.dumps(metrics)
            db.commit()
        except Exception:
            db.rollback()
    finally:
        db.close()
        with _lock:
            _running.discard(source_id)


def start_video_analysis(source_id: int, force: bool = False):
    with _lock:
        if source_id in _running:
            return False
        _running.add(source_id)
    _pool.submit(_run, source_id, force)
    return True
