"""Source-scoped assembly and provenance for full-video SOP drafts."""
from __future__ import annotations

import hashlib
import json
import re

from .models import (
    DecisionKnowledge,
    DiscoveryQuestion,
    KnowledgeChunk,
    SOPComparison,
    Source,
    VideoActionWindow,
    VideoAnalysis,
    VideoCandidate,
    VideoExpertQuestion,
    VideoFrame,
    VideoPipelineRun,
    VideoSequence,
)

ACTIVE_STATUSES = {"pending", "running", "extracting_frames", "analysing_frames", "aligning_evidence", "indexing"}
VALID_RUN_STATUSES = {"completed", "partially_completed"}
PROMPT_VERSION = "source-video-sop-v8-reject-frame-label-steps"
TRANSCRIPT_ONLY_PROMPT_VERSION = "transcript-only-video-sop-v1"


class VideoSOPEvidenceError(ValueError):
    def __init__(self, message: str, status_code: int = 422):
        super().__init__(message)
        self.status_code = status_code


def assemble_transcript_sop_evidence(db, source: Source) -> dict:
    """Assemble only the selected video's persisted transcript for SOP drafting.

    Video observations, analysis status, expert records and retrieved references are
    intentionally outside this path. They remain available in the analysis UI.
    """
    if source.source_type != "video":
        raise VideoSOPEvidenceError("Transcript-based video SOP drafting requires a video source.", 422)

    chunks = (db.query(KnowledgeChunk).filter_by(source_id=source.id)
              .order_by(KnowledgeChunk.start_seconds, KnowledgeChunk.id).all())
    transcript = []
    manifest = []
    seen = set()
    for chunk in chunks:
        # Visual-frame chunks share the knowledge index but are not transcript.
        if (chunk.source_ref or "").strip().lower().startswith("visual frame ·"):
            continue
        text = (chunk.text or "").strip()
        if not text:
            continue
        key = (text, chunk.source_ref or "", chunk.start_seconds, chunk.end_seconds)
        if key in seen:
            continue
        seen.add(key)
        timestamp = _time_label(chunk.start_seconds)
        ref = chunk.source_ref or f"Transcript · {timestamp or 'untimed segment'}"
        item = {
            "text": text,
            "source_ref": ref,
            "timestamp": timestamp,
            "timestamp_start": chunk.start_seconds,
            "timestamp_end": chunk.end_seconds,
            "source_id": source.id,
            "evidence_type": "source_supported",
        }
        transcript.append(item)
        manifest.append(_manifest_item("transcript_chunk", chunk.id, source.id, ref, timestamp))

    if not transcript:
        raise VideoSOPEvidenceError(
            "This video has no usable stored transcript. Re-upload or re-transcribe it with the intended language; "
            "visual analysis is available for inspection but is not used to generate SOPs.", 422,
        )

    # Run identity is optional provenance only. It is deliberately excluded from
    # the digest so analysis reruns cannot change transcript-only draft identity.
    digest_source = {
        "prompt_version": TRANSCRIPT_ONLY_PROMPT_VERSION,
        "source_id": source.id,
        "transcript": transcript,
    }
    digest = hashlib.sha256(json.dumps(
        digest_source, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")).hexdigest()
    return {
        "source_id": source.id,
        "source_type": source.source_type,
        "source_name": source.filename,
        "run_id": None,
        "run_status": None,
        "transcript": transcript,
        "action_evidence": transcript,
        "contextual_answers": [],
        "contextual_evidence": [],
        "warnings": [],
        "counts": {"transcript_segments": len(transcript)},
        "metrics": {},
        "manifest": manifest,
        "snapshot": digest_source,
        "evidence_sha256": digest,
    }


def _json(value, fallback):
    try:
        return json.loads(value) if value else fallback
    except (TypeError, json.JSONDecodeError):
        return fallback


def _time_label(seconds: float | int | None) -> str | None:
    if seconds is None:
        return None
    whole = max(0, int(float(seconds)))
    return f"{whole // 60:02d}:{whole % 60:02d}"


_ACTION_WORDS = re.compile(
    r"\b(check|inspect|place|position|insert|remove|hold|secure|fix|attach|align|"
    r"turn|rotate|press|push|pull|connect|disconnect|tighten|loosen|adjust|set|"
    r"measure|verify|confirm|clean|apply|feed|drill|cut|solder|heat|stop|start|"
    r"switch|clamp|operate|open|close|move|lift|lower|replace|install|use|"
    r"確認|固定|回転|開け|差し|取り付け|外し|締め|緩め|押し|引き|測定|調整|"
    r"接続|切断|清掃|塗布|加熱|停止|開始|操作|動か|動作|置き|持ち|保持|使用|加工)\w*\b|"
    r"(?:します|する|して|した|ください|させます|させる|できる|行います|行う)"
    , re.IGNORECASE
)

def _is_nonprocedural_answer(text: str) -> bool:
    """Identify explicit scene/identity clarifications that are not work steps."""
    value = re.sub(r"\s+", " ", text or "").strip().lower()
    if not value:
        return True
    just_identification = re.search(
        r"\b(?:this|that|it)\s+(?:is|was)\s+(?:just|only|merely)\s+(?:an?\s+)?"
        r"(?:opening\s+)?(?:logo|title\s+card|intro|opening\s+screen|credits?)\b", value,
    )
    japanese_scene_label = re.search(r"(?:ただの|単なる).{0,8}(?:ロゴ|タイトル|画面|イントロ)|(?:オープニング|タイトル).{0,8}ロゴ", value)
    scene_label = re.search(r"\b(?:logo|title\s+card|opening\s+screen|intro(?:duction)?|credits?)\b", value)
    action_text = re.sub(r"\bopening\b", "", value)
    if just_identification:
        tail = re.sub(r"\bopening\b", "", value[just_identification.end():])
        return not bool(_ACTION_WORDS.search(tail))
    if japanese_scene_label:
        return not bool(_ACTION_WORDS.search(value[japanese_scene_label.end():]))
    if scene_label:
        return not bool(_ACTION_WORDS.search(action_text))
    return False


def _manifest_item(kind: str, item_id: int | None, source_id: int, source_ref: str, timestamp=None, **extra):
    return {"kind": kind, "id": item_id, "source_id": source_id, "source_ref": source_ref,
            "timestamp": timestamp, **extra}


def _seconds(value):
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _matching_windows(start, end, windows):
    """Return current-run windows whose timestamp range overlaps this evidence."""
    start, end = _seconds(start), _seconds(end)
    if start is None and end is None:
        return []
    if start is None:
        start = end
    if end is None:
        end = start
    lo, hi = min(start, end), max(start, end)
    matches = []
    for window in windows:
        if hi >= float(window.timestamp_start) and lo <= float(window.timestamp_end):
            matches.append({"window_id": window.id, "start": window.timestamp_start,
                            "end": window.timestamp_end})
    return matches


def _with_window_links(item, matches):
    item["window_ids"] = list(dict.fromkeys(link["window_id"] for link in matches))
    item["window_id"] = item["window_ids"][0] if item["window_ids"] else None
    item["action_window_refs"] = matches
    item["alignment_status"] = "timestamp_aligned_to_action_window" if matches else "timestamped_without_action_window"
    return item


def _sequence_observation_text(payload):
    """Read only persisted descriptive fields; do not synthesize an observation."""
    if not isinstance(payload, dict):
        return ""
    values = []
    for field in ("description", "sequence_summary", "summary"):
        value = payload.get(field)
        if isinstance(value, str) and value.strip():
            values.append(value.strip())
    for field in ("visible_changes", "observable_actions", "observations"):
        entries = payload.get(field)
        if isinstance(entries, list):
            for entry in entries:
                if isinstance(entry, str) and entry.strip():
                    values.append(entry.strip())
                elif isinstance(entry, dict):
                    description = entry.get("description") or entry.get("text")
                    if isinstance(description, str) and description.strip():
                        values.append(description.strip())
    return "; ".join(dict.fromkeys(values))


def _select_prompt_visuals(observations: list[dict]) -> list[dict]:
    """Use all aligned observations; relevance is assessed from evidence, not an English verb list."""
    return sorted((item for item in observations
                   if item.get("alignment_status") != "unaligned"),
                  key=lambda item: (_seconds(item.get("timestamp_start")) or 0.0,
                                    item.get("window_id") or 0,
                                    item.get("frame_id") or 0))


def assemble_video_sop_evidence(db, source: Source) -> dict:
    """Load the complete selected-source evidence set, bounded to one valid run."""
    if source.source_type != "video":
        raise VideoSOPEvidenceError("Full-video SOP drafting requires a video source.", 422)

    job = db.query(VideoAnalysis).filter_by(source_id=source.id).one_or_none()
    if job and job.status in ACTIVE_STATUSES:
        raise VideoSOPEvidenceError("Video analysis is still running. Wait for it to finish before generating a full-video SOP draft.", 409)

    runs = db.query(VideoPipelineRun).filter_by(source_id=source.id).order_by(VideoPipelineRun.id.desc()).all()
    if runs and runs[0].status in ACTIVE_STATUSES:
        raise VideoSOPEvidenceError("Video analysis is still running. Wait for it to finish before generating a full-video SOP draft.", 409)
    run = runs[0] if runs and runs[0].status in VALID_RUN_STATUSES else None
    if not run:
        latest_status = runs[0].status if runs else "not_started"
        raise VideoSOPEvidenceError(
            f"The latest video analysis run is {latest_status}; only a completed or partially completed current run can support this draft. "
            "Run or resume analysis, then retry.", 422,
        )

    transcript: list[dict] = []
    visual: list[dict] = []
    visual_sequence_context: list[dict] = []
    expert_answers: list[dict] = []
    contextual_answers: list[dict] = []
    document_references: list[dict] = []
    approved_sop_references: list[dict] = []
    seen_document_chunks: set[int] = set()
    seen_approved_references: set[tuple[int, str]] = set()
    manifest: list[dict] = []
    transcript_by_key: dict[tuple, dict] = {}

    windows = (db.query(VideoActionWindow).filter_by(run_id=run.id, source_id=source.id)
               .order_by(VideoActionWindow.timestamp_start, VideoActionWindow.id).all())
    comparisons = {record.window_id: record for record in
                   db.query(SOPComparison).filter_by(run_id=run.id, source_id=source.id).all()}

    chunks = (db.query(KnowledgeChunk).filter_by(source_id=source.id)
              .order_by(KnowledgeChunk.start_seconds, KnowledgeChunk.id).all())
    for chunk in chunks:
        if (chunk.source_ref or "").lower().startswith("visual frame ·"):
            continue
        text = (chunk.text or "").strip()
        if not text:
            continue
        timestamp = _time_label(chunk.start_seconds)
        key = (text, chunk.source_ref or "", timestamp)
        ref = chunk.source_ref or f"{source.filename} · {timestamp or 'transcript'}"
        matches = _matching_windows(chunk.start_seconds, chunk.end_seconds, windows)
        item = transcript_by_key.get(key)
        if item is None:
            item = _with_window_links({"text": text, "source_ref": ref, "timestamp": timestamp,
                "timestamp_start": chunk.start_seconds, "timestamp_end": chunk.end_seconds,
                "source_id": source.id, "evidence_type": "source_supported"}, matches)
            transcript_by_key[key] = item
            transcript.append(item)
        elif matches:
            merged = {link["window_id"]: link for link in item["action_window_refs"]}
            merged.update({link["window_id"]: link for link in matches})
            _with_window_links(item, list(merged.values()))
        manifest.append(_manifest_item("transcript_chunk", chunk.id, source.id, ref, timestamp,
                                       window_ids=item["window_ids"]))
    for window in windows:
        for segment in _json(window.transcript_json, []):
            if not isinstance(segment, dict) or not str(segment.get("text", "")).strip():
                continue
            text = str(segment["text"]).strip()
            start = segment.get("start_seconds", segment.get("start", window.timestamp_start))
            end = segment.get("end_seconds", segment.get("end", start))
            timestamp = _time_label(start)
            ref = segment.get("source_ref") or f"{source.filename} · {timestamp or _time_label(window.timestamp_start)}"
            key = (text, ref, timestamp)
            link = [{"window_id": window.id, "start": window.timestamp_start, "end": window.timestamp_end}]
            # A stored window relationship is authoritative only when its timestamp is
            # compatible with the window. Preserve mismatches as timestamped evidence.
            if _matching_windows(start, end, [window]):
                matches = link
            else:
                matches = []
            item = transcript_by_key.get(key)
            if item is None:
                item = _with_window_links({"text": text, "source_ref": ref, "timestamp": timestamp,
                    "timestamp_start": _seconds(start), "timestamp_end": _seconds(end),
                    "source_id": source.id, "evidence_type": "source_supported"}, matches)
                transcript_by_key[key] = item
                transcript.append(item)
            elif matches:
                merged = {entry["window_id"]: entry for entry in item["action_window_refs"]}
                merged.update({entry["window_id"]: entry for entry in matches})
                _with_window_links(item, list(merged.values()))
            manifest.append(_manifest_item("window_transcript", window.id, source.id, ref, timestamp,
                                           window_id=window.id if matches else None))

        summary = _json(window.summary, {})
        if isinstance(summary, dict) and str(summary.get("sequence_summary", "")).strip():
            timestamp = _time_label(window.timestamp_start)
            ref = f"{source.filename} · visual sequence {_time_label(window.timestamp_start)}–{_time_label(window.timestamp_end)}"
            visual_sequence_context.append({"text": "UNVERIFIED AI VISUAL SEQUENCE SUMMARY (context only; may omit or misorder actions): " + str(summary["sequence_summary"]).strip(),
                           "source_ref": ref, "timestamp": timestamp, "timestamp_start": window.timestamp_start,
                           "timestamp_end": window.timestamp_end, "source_id": source.id,
                           "evidence_type": "visual_sequence_summary", "window_id": window.id,
                           "window_ids": [window.id], "alignment_status": "action_window_record",
                           "context_only": True})
            manifest.append(_manifest_item("action_window_summary", window.id, source.id, ref, timestamp,
                                           window_id=window.id, verified=False, context_only=True,
                                           generation_selected=True))

        comparison = comparisons.get(window.id)
        if comparison:
            refs = _json(comparison.references_json, [])
            for reference in refs if isinstance(refs, list) else []:
                if not isinstance(reference, dict) or not reference.get("excerpt"):
                    continue
                if reference.get("chunk_id"):
                    chunk = db.get(KnowledgeChunk, reference["chunk_id"])
                    related_source = db.get(Source, chunk.source_id) if chunk else None
                    if not chunk or not related_source or related_source.source_type != "document" or chunk.id in seen_document_chunks:
                        continue
                    seen_document_chunks.add(chunk.id)
                    ref = chunk.source_ref or reference.get("reference") or f"{related_source.filename} · chunk {chunk.id}"
                    document_references.append({"text": str(reference["excerpt"]), "source_ref": ref,
                        "timestamp": None, "source_id": related_source.id, "evidence_type": "source_supported"})
                    manifest.append(_manifest_item("related_document_chunk", chunk.id, related_source.id, ref,
                                                   window_id=window.id))
                elif reference.get("procedure_id"):
                    from .models import Procedure
                    procedure = db.get(Procedure, reference["procedure_id"])
                    ref = str(reference.get("reference") or f"Approved procedure {reference['procedure_id']}")
                    identity = (procedure.id, ref) if procedure else None
                    if not procedure or procedure.status != "approved" or identity in seen_approved_references:
                        continue
                    seen_approved_references.add(identity)
                    approved_sop_references.append({"text": str(reference["excerpt"]), "source_ref": ref,
                        "timestamp": None, "source_id": procedure.source_id, "procedure_id": procedure.id,
                        "evidence_type": "approved_sop_reference", "similarity": reference.get("similarity")})
                    manifest.append(_manifest_item("approved_sop_reference", procedure.id, procedure.source_id or 0,
                                                   ref, window_id=window.id))

    candidates = (db.query(VideoCandidate).filter_by(run_id=run.id, source_id=source.id)
                  .filter(VideoCandidate.selected_for_detail.is_(True)).all())
    candidate_by_frame = {candidate.frame_id: candidate for candidate in candidates if candidate.frame_id is not None}
    frame_ids = set(candidate_by_frame)
    window_frame_ids = {}
    for window in windows:
        for frame_id in _json(window.frame_ids_json, []):
            if isinstance(frame_id, int):
                frame_ids.add(frame_id)
                window_frame_ids.setdefault(frame_id, []).append(window)
    frames = (db.query(VideoFrame).filter(VideoFrame.source_id == source.id,
                VideoFrame.id.in_(frame_ids), VideoFrame.analysis_status == "completed")
              .order_by(VideoFrame.timestamp_seconds, VideoFrame.id).all() if frame_ids else [])
    unaligned_visual = []
    for frame in frames:
        observation = _json(frame.analysis_json, {})
        if not isinstance(observation, dict):
            continue
        description = str(observation.get("description", "")).strip()
        actions = observation.get("observable_actions", [])
        if isinstance(actions, list):
            actions_text = "; ".join(str(item).strip() for item in actions if str(item).strip())
        else:
            actions_text = ""
        details = "; ".join(value for value in (description, actions_text) if value)
        if not details:
            continue
        timestamp = _time_label(frame.timestamp_seconds)
        ref = f"{source.filename} · frame {frame.id} · {timestamp}"
        matching = window_frame_ids.get(frame.id, [])
        if not matching:
            matching = [window for window in windows
                        if window.timestamp_start <= frame.timestamp_seconds <= window.timestamp_end]
        links = [{"window_id": window.id, "start": window.timestamp_start, "end": window.timestamp_end}
                 for window in matching]
        item = {"text": "UNVERIFIED AI VISUAL OBSERVATION (visible content only): " + details,
                "source_ref": ref, "timestamp": timestamp, "timestamp_start": frame.timestamp_seconds,
                "timestamp_end": frame.timestamp_seconds, "source_id": source.id,
                "evidence_type": "visual_observation", "frame_id": frame.id,
                "candidate_id": candidate_by_frame.get(frame.id).id if frame.id in candidate_by_frame else None,
                "window_ids": [link["window_id"] for link in links],
                "window_id": links[0]["window_id"] if links else None,
                "action_window_refs": links,
                "alignment_status": "timestamp_aligned_to_action_window" if links else "unaligned"}
        if links:
            visual.append(item)
        else:
            # Keep observations in the provenance but never let an unaligned still
            # substantiate a procedural action.
            unaligned_visual.append(item)
        manifest.append(_manifest_item("visual_frame", frame.id, source.id, ref, timestamp,
            verified=False, candidate_id=item["candidate_id"], window_ids=item["window_ids"],
            alignment_status=item["alignment_status"]))

    # Older persisted sequence observations have source/time but no run_id. Recover
    # their relationship only where their timestamp interval overlaps this run's windows.
    sequences = (db.query(VideoSequence).filter_by(source_id=source.id)
                 .order_by(VideoSequence.timestamp_start, VideoSequence.id).all())
    for sequence in sequences:
        details = _sequence_observation_text(_json(sequence.evidence_json, {}))
        if not details:
            continue
        matches = _matching_windows(sequence.timestamp_start, sequence.timestamp_end, windows)
        timestamp = _time_label(sequence.timestamp_start)
        ref = f"{source.filename} · saved visual sequence {sequence.id} · {timestamp}"
        item = {"text": "UNVERIFIED SAVED VISUAL SEQUENCE OBSERVATION: " + details,
                "source_ref": ref, "timestamp": timestamp, "timestamp_start": sequence.timestamp_start,
                "timestamp_end": sequence.timestamp_end, "source_id": source.id,
                "evidence_type": "visual_observation", "sequence_id": sequence.id,
                "window_ids": [link["window_id"] for link in matches],
                "window_id": matches[0]["window_id"] if matches else None,
                "action_window_refs": matches,
                "alignment_status": "timestamp_aligned_to_action_window" if matches else "unaligned"}
        (visual if matches else unaligned_visual).append(item)
        manifest.append(_manifest_item("video_sequence", sequence.id, source.id, ref, timestamp,
            verified=False, window_ids=item["window_ids"], alignment_status=item["alignment_status"]))

    questions = (db.query(VideoExpertQuestion).filter_by(source_id=source.id, run_id=run.id)
                 .order_by(VideoExpertQuestion.timestamp_seconds, VideoExpertQuestion.id).all())
    for question in questions:
        if question.answer_status != "answered" or not (question.answer or "").strip():
            continue
        timestamp = _time_label(question.timestamp_seconds)
        ref = f"{source.filename} · expert answer {question.id} · {timestamp}"
        record = {"text": f"Expert question: {question.question}\nExpert answer: {question.answer.strip()}",
                  "source_ref": ref, "timestamp": timestamp, "source_id": source.id,
                  "evidence_type": "expert_confirmed", "question_id": question.id,
                  "window_id": question.window_id}
        if _is_nonprocedural_answer(question.answer):
            contextual_answers.append({**record, "context_only": True})
        else:
            expert_answers.append(record)
        manifest.append(_manifest_item("video_expert_answer", question.id, source.id, ref, timestamp,
                                       window_id=question.window_id, context_only=_is_nonprocedural_answer(question.answer)))

    # Preserve expert knowledge entered through the earlier candidate-question workflow.
    legacy_answers = (db.query(DiscoveryQuestion).filter_by(source_id=source.id, answer_status="answered")
                      .order_by(DiscoveryQuestion.timestamp_seconds, DiscoveryQuestion.id).all())
    for answer in legacy_answers:
        text = " ".join(part for part in (answer.answer, answer.correction) if part and part.strip()).strip()
        if not text:
            continue
        timestamp = _time_label(answer.timestamp_seconds)
        ref = f"{source.filename} · legacy expert answer {answer.id} · {timestamp}"
        record = {"text": f"Expert question: {answer.question}\nExpert clarification: {text}",
                  "source_ref": ref, "timestamp": timestamp, "source_id": source.id,
                  "evidence_type": "expert_confirmed", "question_id": answer.id}
        if _is_nonprocedural_answer(text):
            contextual_answers.append({**record, "context_only": True})
        else:
            expert_answers.append(record)
        manifest.append(_manifest_item("legacy_expert_answer", answer.id, source.id, ref, timestamp,
                                       context_only=_is_nonprocedural_answer(text)))

    for record in (db.query(DecisionKnowledge).filter_by(source_id=source.id)
                   .filter(DecisionKnowledge.verification_status.in_(["expert_confirmed", "expert_reviewed"]))
                   .order_by(DecisionKnowledge.timestamp_seconds, DecisionKnowledge.id).all()):
        fields = ("context", "observed_cue", "interpretation", "decision", "action", "outcome", "exceptions", "limits")
        summary = "; ".join(f"{field.replace('_', ' ')}: {getattr(record, field)}" for field in fields if getattr(record, field, "").strip())
        if summary:
            timestamp = _time_label(record.timestamp_seconds)
            ref = f"{source.filename} · verified decision knowledge {record.id} · {timestamp}"
            expert_answers.append({"text": summary, "source_ref": ref, "timestamp": timestamp,
                                   "source_id": source.id, "evidence_type": "expert_confirmed"})
            manifest.append(_manifest_item("decision_knowledge", record.id, source.id, ref, timestamp))

    transcript.sort(key=lambda item: (_seconds(item.get("timestamp_start"))
        if _seconds(item.get("timestamp_start")) is not None else float("inf"), item["source_ref"]))
    prompt_visual = _select_prompt_visuals(visual)
    selected_visual_ids = {item.get("frame_id", item.get("sequence_id")) for item in prompt_visual}
    for item in manifest:
        if item["kind"] in {"visual_frame", "video_sequence"}:
            item["generation_selected"] = item["id"] in selected_visual_ids

    metrics = _json(run.metrics_json, {})
    run_warnings = _json(run.warnings_json, [])
    warnings = [str(item) for item in run_warnings if str(item).strip()] if isinstance(run_warnings, list) else []
    if run.status == "partially_completed":
        covered = metrics.get("detailed_windows_covered")
        unresolved_count = metrics.get("candidate_windows_not_detailed")
        total = covered + unresolved_count if isinstance(covered, int) and isinstance(unresolved_count, int) else "unknown"
        if "candidate_windows_screened_out" in metrics:
            warning = f"Analysis run {run.id} is partially complete. Detailed visual coverage: {covered if covered is not None else 'unknown'}/{total} selected or high-priority windows; lower-priority candidates were screened out."
        else:
            warning = f"Analysis run {run.id} is partially complete. Its legacy metrics do not distinguish screened-out candidates from unresolved windows; review the timeline coverage before approval."
        warnings.append(warning)
    known_selection_policy = "candidate_windows_screened_out" in metrics
    unresolved_statuses = {"unresolved_important", "partially_analyzed", "failed"}
    if not known_selection_policy:
        unresolved_statuses.add("not_selected_for_detail")
    unresolved = [window for window in windows if window.status in unresolved_statuses]
    if unresolved:
        labels = ", ".join(f"{_time_label(w.timestamp_start)}–{_time_label(w.timestamp_end)} ({w.status})" for w in unresolved)
        warnings.append("Unresolved candidate windows: " + labels)

    source_text = " ".join(item["text"] for item in transcript)
    query_text = source_text[:1600]
    if query_text.strip():
        try:
            from .knowledge import knowledge_index
            for chunk in knowledge_index.search(db, query_text, top_k=5, source_type="document"):
                related_source = db.get(Source, chunk.source_id)
                if not related_source or related_source.source_type != "document" or not chunk.text.strip():
                    continue
                if chunk.id in seen_document_chunks:
                    continue
                seen_document_chunks.add(chunk.id)
                ref = chunk.source_ref or f"{related_source.filename} · chunk {chunk.id}"
                document_references.append({"text": chunk.text, "source_ref": ref, "timestamp": None,
                                            "source_id": related_source.id, "evidence_type": "source_supported"})
                manifest.append(_manifest_item("related_document_chunk", chunk.id, related_source.id, ref))
        except Exception:
            warnings.append("Related written-document RAG references could not be retrieved for this draft.")
    if query_text.strip():
        try:
            from .video_jobs import _approved_procedure_references
            from .models import Procedure
            for reference in _approved_procedure_references(db, query_text, top_k=3):
                procedure = db.get(Procedure, reference["procedure_id"])
                if not procedure or procedure.status != "approved":
                    continue
                ref = reference["reference"]
                identity = (procedure.id, ref)
                if identity in seen_approved_references:
                    continue
                seen_approved_references.add(identity)
                approved_sop_references.append({"text": reference["excerpt"], "source_ref": ref, "timestamp": None,
                    "source_id": procedure.source_id, "procedure_id": procedure.id,
                    "evidence_type": "approved_sop_reference", "similarity": reference.get("similarity")})
                manifest.append(_manifest_item("approved_sop_reference", procedure.id, procedure.source_id or 0, ref))
        except Exception:
            warnings.append("Related approved SOP references could not be retrieved for this draft.")

    action_evidence = transcript + prompt_visual + expert_answers
    for item in action_evidence:
        if item.get("source_id") == source.id:
            continue
        if item.get("evidence_type") == "approved_sop_reference" and item.get("procedure_id"):
            continue
        # Related records are permitted only as clearly identified external reference context.
        if item.get("evidence_type") != "source_supported":
            raise VideoSOPEvidenceError("Evidence from another source was detected during assembly. No draft was created.", 500)

    # Related manuals/SOPs, scene clarifications, unaligned stills, and sequence
    # summaries are supplied as explicitly context-only material. The generator
    # may use them to surface uncertainty or comparison notes, never to substantiate
    # a step in place of evidence from this selected video.
    reference_context = [
        {**item, "context_only": True}
        for item in document_references + approved_sop_references
    ]
    for item in manifest:
        if item["kind"] in {"related_document_chunk", "approved_sop_reference"}:
            item["context_only"] = True
            item["generation_selected"] = True
    contextual_evidence = (
        contextual_answers + unaligned_visual + visual_sequence_context + reference_context
    )
    if unaligned_visual:
        warnings.append(f"{len(unaligned_visual)} visual observation(s) lack a confident link to an action window. They are retained for review but excluded as step evidence.")
    evidence_snapshot = {
        "prompt_version": PROMPT_VERSION,
        "source_id": source.id, "source_type": source.source_type, "source_name": source.filename,
        "run_id": run.id, "run_status": run.status, "transcript": transcript, "visual_observations": visual,
        "unaligned_visual_observations": unaligned_visual, "visual_sequence_context": visual_sequence_context,
        "generation_visual_observation_ids": [item.get("frame_id", item.get("sequence_id")) for item in prompt_visual],
        "expert_answers": expert_answers, "contextual_answers": contextual_answers,
        "document_references": document_references, "approved_sop_references": approved_sop_references,
        "action_windows": [{"id": window.id, "start": window.timestamp_start, "end": window.timestamp_end,
            "status": window.status, "summary": _json(window.summary, {}),
            "comparison": {"relationship": comparisons[window.id].relationship,
                           "explanation": comparisons[window.id].explanation} if window.id in comparisons else {}}
            for window in windows],
        "warnings": warnings, "metrics": metrics,
    }
    digest = hashlib.sha256(json.dumps(evidence_snapshot, ensure_ascii=False, sort_keys=True,
                                       separators=(",", ":")).encode("utf-8")).hexdigest()
    counts = {"transcript_segments": len(transcript), "visual_observations": len(visual) + len(unaligned_visual) + len(visual_sequence_context),
              "aligned_visual_observations": len(visual), "unaligned_visual_observations": len(unaligned_visual),
              "visual_sequence_summaries": len(visual_sequence_context),
              "visual_observations_used_for_generation": len(prompt_visual),
              "expert_answers": len(expert_answers), "contextual_answers": len(contextual_answers),
              "related_document_chunks": len(document_references), "approved_sop_references": len(approved_sop_references),
              "action_windows": len(windows)}
    if not transcript and not visual and not expert_answers:
        raise VideoSOPEvidenceError(
            "This video has no usable transcript, completed visual observation, or expert-confirmed procedural evidence. "
            f"Evidence found: {counts}. Generate or resume analysis and clarify an actionable work moment before retrying.", 422,
        )
    return {"source_id": source.id, "source_type": source.source_type, "source_name": source.filename,
            "run_id": run.id, "run_status": run.status, "action_evidence": action_evidence,
            "contextual_answers": contextual_answers, "contextual_evidence": contextual_evidence,
            "warnings": list(dict.fromkeys(warnings)),
            "counts": counts, "metrics": metrics, "manifest": manifest, "snapshot": evidence_snapshot,
            "evidence_sha256": digest}


