from importlib.resources import path
from pathlib import Path
from datetime import datetime, timezone
import json, shutil, re
import logging
from typing import Literal
from io import BytesIO
from xml.parsers.expat import model
from fastapi import Form

from numpy import info
from fastapi import FastAPI, Depends, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from .db import Base, engine, get_db, SessionLocal
from .models import (Source, KnowledgeChunk, Procedure, TrainingProgress, VideoAnalysis, VideoFrame,
                     VideoSequence, DiscoveryQuestion, DecisionKnowledge, VideoPipelineRun,
                     VideoCandidate, VideoActionWindow, VideoExpertQuestion, SOPComparison,
                     SOPImprovementProposal, VideoSOPDraftVersion)
from .schemas import (AskRequest, ProcedureUpdate, ReviewRequest, ProgressRequest,
                      DiscoveryAnswerRequest, DecisionKnowledgeRequest, DecisionKnowledgeUpdate,
                      VideoExpertAnswerRequest)
from .config import UPLOAD_DIR, CORS_ORIGINS, WHISPER_MODEL
from .knowledge import knowledge_index
from .ai import (grounded_answer, draft_procedure, ollama, LocalModelError,
                 PROCEDURE_RESPONSE_SCHEMA)
from .video_sop import (assemble_transcript_sop_evidence, VideoSOPEvidenceError,
                        TRANSCRIPT_ONLY_PROMPT_VERSION as VIDEO_SOP_PROMPT_VERSION)
from .transcription import transcribe_video
from .source_evidence import load_source_excerpts
from .video_jobs import start_video_analysis
from .config import STORAGE_DIR, VISION_MODEL, OLLAMA_MODEL
from . import config as app_config

logger = logging.getLogger(__name__)

# Create tables at launch. The Postgres service must be running first.
Base.metadata.create_all(bind=engine)
# Existing deployments created this provenance column as NOT NULL. A transcript
# draft is independent of video analysis, so make its run association optional
# without changing or deleting any existing version records.
if engine.dialect.name == "postgresql":
    with engine.begin() as connection:
        connection.execute(text(
            "ALTER TABLE video_sop_draft_versions ALTER COLUMN run_id DROP NOT NULL"
        ))
app = FastAPI(title="SkillBridge AR API", version="0.1.0", description="Hackathon MVP: document RAG, video RAG, approved SOPs and training progress.")
app.add_middleware(CORSMiddleware, allow_origins=CORS_ORIGINS, allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120, separators=["\n\n", "\n", ". ", " ", ""])

@app.on_event("startup")
def load_index():
    with SessionLocal() as db:
        knowledge_index.rebuild(db)
        # Jobs run in-process; mark interrupted jobs honestly after a restart.
        for job in db.query(VideoAnalysis).filter(VideoAnalysis.status.in_(
                ["pending", "running", "extracting_frames", "analysing_frames", "aligning_evidence", "indexing"])).all():
            job.status, job.error = "failed", "Analysis was interrupted when the API process stopped. Retry to resume using saved frames."
        for run in db.query(VideoPipelineRun).filter_by(status="running").all():
            run.status = "failed"
            run.error = "Analysis was interrupted when the API process stopped. Retry to resume using saved stages."
        db.commit()

def source_out(s): return {"id":s.id,"source_type":s.source_type,"filename":s.filename,"created_at":s.created_at.isoformat() if s.created_at else None}
def procedure_out(p):
    return {"id":p.id,"title":p.title,"description":p.description,"tools":json.loads(p.tools_json or "[]"),"warnings":json.loads(p.warnings_json or "[]"),"steps":json.loads(p.steps_json or "[]"),"source_id":p.source_id,"status":p.status,"rejection_note":p.rejection_note,"created_at":p.created_at.isoformat() if p.created_at else None,"updated_at":p.updated_at.isoformat() if p.updated_at else None}
def make_chunks(db, source, entries):
    rows=[]
    for item in entries:
        row=KnowledgeChunk(source_id=source.id,text=item["text"],source_ref=item["source_ref"],start_seconds=item.get("start_seconds"),end_seconds=item.get("end_seconds"),embedding_json="[]")
        db.add(row); rows.append(row)
    db.commit()
    for row in rows: db.refresh(row)
    knowledge_index.add_chunks(db, rows)
    return len(rows)

@app.get("/api/health")
def health(): return {"status":"ok","service":"SkillBridge AR API"}

@app.post("/api/documents/upload")
async def upload_document(file: UploadFile=File(...), db: Session=Depends(get_db)):
    filename = Path(file.filename or "manual.pdf").name
    if not filename.lower().endswith(".pdf"): raise HTTPException(400,"Please upload a PDF manual or SOP.")
    data = await file.read()
    if len(data) > 30*1024*1024: raise HTTPException(413,"MVP upload limit is 30 MB.")
    path=UPLOAD_DIR / f"{datetime.now().strftime('%Y%m%d%H%M%S%f')}_{filename}"; path.write_bytes(data)
    try: reader=PdfReader(BytesIO(data))
    except Exception as e: path.unlink(missing_ok=True); raise HTTPException(400,f"Unable to read PDF: {e}")
    entries=[]
    for page_no, page in enumerate(reader.pages, start=1):
        text = re.sub(
            r"\s+",
            " ",
            (page.extract_text() or "")
        ).strip()
        if not text: continue
        for idx, chunk in enumerate(splitter.split_text(text)):
            entries.append({"text":chunk,"source_ref":f"{filename} · page {page_no}"})
    if not entries: path.unlink(missing_ok=True); raise HTTPException(400,"No extractable text found. This PDF may be scanned; use an OCR/text-readable PDF for this demo.")
    source=Source(source_type="document",filename=filename,file_path=str(path)); db.add(source); db.commit(); db.refresh(source)
    count=make_chunks(db, source, entries)
    return {"source":source_out(source),"chunks_created":count,"message":"Document processed and indexed."}

@app.post("/api/videos/upload")
async def upload_video(file: UploadFile=File(...), db: Session=Depends(get_db),language: str = Form("auto")):
    filename=Path(file.filename or "expert-demo.mp4").name
    if Path(filename).suffix.lower() not in {".mp4",".mov",".m4v",".webm",".mkv",".mp3",".wav",".m4a"}: raise HTTPException(400,"Supported: mp4, mov, m4v, webm, mkv, mp3, wav, m4a.")
    data=await file.read()
    if len(data)>200*1024*1024: raise HTTPException(413,"MVP upload limit is 200 MB.")
    path=UPLOAD_DIR / f"{datetime.now().strftime('%Y%m%d%H%M%S%f')}_{filename}"; path.write_bytes(data)
    source=Source(source_type="video",filename=filename,file_path=str(path)); db.add(source); db.commit(); db.refresh(source)
    try:
        transcription, info = transcribe_video(str(path), language, filename)
        detected_language = info.language
        language_probability = float(info.language_probability)
    except Exception as e:
        logger.exception("Video transcription failed: source_id=%s filename=%s requested_language=%s model=%s", source.id, filename, language, WHISPER_MODEL)
        return {"source": source_out(source), "chunks_created": 0, "transcript": [],
                "transcription": {"status": "failed", "requested_language": language,
                    "detected_language": None, "language_probability": None, "segment_count": 0,
                    "transcript_characters": 0, "usable_text": False, "accuracy_verified": False,
                    "warning": "Transcription failed; the original video was retained. Re-upload with the intended language or continue with visual-only analysis.",
                    "error": str(e)[:500]},
                "message": "Video stored, but transcription failed. Visual-only analysis remains available."}
    count=make_chunks(db, source, transcription)
    transcript_length = sum(len(item["text"]) for item in transcription)
    language_warning = None
    if language != "auto" and detected_language != language:
        language_warning = f"Requested {language}, but Whisper detected {detected_language}. Review transcript language and accuracy before drafting."
    elif language_probability < 0.55:
        language_warning = "Whisper language confidence is low. Review the transcript before drafting."
    logger.info("Video transcription complete: source_id=%s filename=%s requested_language=%s detected_language=%s language_probability=%.3f segments=%s transcript_characters=%s usable_text=%s chunks=%s",
                source.id, filename, language, detected_language, language_probability, len(transcription), transcript_length, bool(transcript_length), count)
    return {"source":source_out(source),"chunks_created":count,"transcript":transcription,
            "transcription":{"status":"completed" if transcript_length else "empty","requested_language":language,"detected_language":detected_language,
                             "language_probability":language_probability,"segment_count":len(transcription),
                             "transcript_characters":transcript_length,"usable_text":bool(transcript_length),
                             "accuracy_verified":False,"warning":language_warning},
            "message":("Audio transcribed and timestamped." if not language_warning else "Audio transcribed and timestamped; transcript review recommended.") if transcript_length else "Video stored; Whisper returned no transcript text. Visual-only analysis remains available."}

@app.get("/api/sources")
def list_sources(db: Session=Depends(get_db)):
    sources = db.query(Source).order_by(Source.created_at.desc()).all()
    statuses = {job.source_id: job.status for job in db.query(VideoAnalysis).all()}
    return [{**source_out(s), "video_analysis_status": statuses.get(s.id, "not_started") if s.source_type == "video" else None} for s in sources]

@app.post("/api/sources/{source_id}/video-analysis")
def begin_video_analysis(source_id: int, force: bool = False, db: Session=Depends(get_db)):
    source = db.get(Source, source_id)
    if not source or source.source_type != "video": raise HTTPException(404, "Video source not found.")
    job = db.query(VideoAnalysis).filter_by(source_id=source_id).one_or_none()
    if job and job.status == "completed" and not force:
        return {"source_id": source_id, "status": job.status, "progress": job.progress, "message": "Analysis already completed. Set force=true to start a fresh run."}
    if job and job.status in {"pending", "running", "extracting_frames", "analysing_frames", "aligning_evidence", "indexing"}:
        return {"source_id": source_id, "status": job.status, "progress": job.progress, "message": "Analysis is already running."}
    if not job:
        job = VideoAnalysis(source_id=source_id, status="pending", model_name=VISION_MODEL)
        db.add(job)
    job.status, job.error = "pending", ""
    db.commit()
    if not start_video_analysis(source_id, force=force):
        return {"source_id": source_id, "status": "running", "progress": job.progress, "message": "Analysis is already running."}
    return {"source_id": source_id, "status": "running", "progress": job.progress, "message": "Local video analysis started; completed stages will be reused on retry."}

@app.get("/api/sources/{source_id}/video-analysis")
def get_video_analysis(source_id: int, db: Session=Depends(get_db)):
    job = db.query(VideoAnalysis).filter_by(source_id=source_id).one_or_none()
    if not job: return {"source_id": source_id, "status": "not_started", "progress": 0, "error": ""}
    run = db.query(VideoPipelineRun).filter_by(source_id=source_id).order_by(VideoPipelineRun.id.desc()).first()
    sheet_dir = STORAGE_DIR / "frames" / str(source_id) / "runs" / str(run.id) / "contact-sheets" if run else None
    contact_sheets = ([{"number": int(path.stem.rsplit("-", 1)[-1]),
                        "url": f"/api/sources/{source_id}/video-analysis/{run.id}/contact-sheets/{path.stem.rsplit('-', 1)[-1]}"}
                       for path in sorted(sheet_dir.glob("sheet-*.jpg"))] if sheet_dir and sheet_dir.is_dir() else [])
    return {"source_id": source_id, "status": job.status, "progress": job.progress, "error": job.error,
            "model": job.model_name, "run_id": run.id if run else None,
            "stage": run.stage if run else "not_started",
            "metrics": json.loads(run.metrics_json or "{}") if run else {},
            "warnings": json.loads(run.warnings_json or "[]") if run else [],
            "metadata": json.loads(run.metadata_json or "{}") if run else {},
            "run_error": run.error if run else "", "contact_sheets": contact_sheets}

@app.get("/api/sources/{source_id}/evidence-timeline")
def evidence_timeline(source_id: int, db: Session=Depends(get_db)):
    source = db.get(Source, source_id)
    if not source or source.source_type != "video": raise HTTPException(404, "Video source not found.")
    chunks = db.query(KnowledgeChunk).filter_by(source_id=source_id).order_by(KnowledgeChunk.start_seconds, KnowledgeChunk.id).all()
    chunks = [chunk for chunk in chunks if not (chunk.source_ref or "").startswith("visual frame ·")]
    run = db.query(VideoPipelineRun).filter_by(source_id=source_id).order_by(VideoPipelineRun.id.desc()).first()
    candidates = db.query(VideoCandidate).filter_by(run_id=run.id).order_by(VideoCandidate.timestamp_seconds).all() if run else []
    if run:
        frame_ids = {candidate.frame_id for candidate in candidates if candidate.frame_id is not None}
        frames = db.query(VideoFrame).filter(VideoFrame.id.in_(frame_ids), VideoFrame.source_id == source_id).order_by(VideoFrame.timestamp_seconds).all() if frame_ids else []
    else:
        frames = db.query(VideoFrame).filter_by(source_id=source_id).order_by(VideoFrame.timestamp_seconds).all()
    sequences = db.query(VideoSequence).filter_by(source_id=source_id).order_by(VideoSequence.timestamp_start).all()
    candidate_by_frame = {candidate.frame_id: candidate for candidate in candidates if candidate.frame_id is not None}
    entries = []
    for frame in frames:
        ts = frame.timestamp_seconds
        related_audio = [c for c in chunks if c.start_seconds is not None and c.end_seconds is not None and c.start_seconds <= ts + 2 and c.end_seconds >= ts]
        candidate = candidate_by_frame.get(frame.id)
        entries.append({"source_id": source_id, "timestamp_start": ts, "timestamp_end": ts,
                        "audio_text": " ".join(c.text for c in related_audio),
                        "visual": json.loads(frame.analysis_json or "{}") if frame.analysis_status == "completed" else None,
                        "frame_status": frame.analysis_status, "frame_id": frame.id,
                        "frame_url": f"/api/sources/{source_id}/frames/{frame.id}",
                        "candidate": ({"score": candidate.score, "score_components": json.loads(candidate.score_components_json or "{}"),
                                      "reasons": json.loads(candidate.reasons_json or "[]"), "selection_reason": candidate.selection_reason,
                                      "is_contact_sample": candidate.is_contact_sample, "selected_for_detail": candidate.selected_for_detail,
                                      "screening_status": candidate.screening_status} if candidate else None),
                        "evidence_types": ["visual"] + (["audio"] if related_audio else [])})
    if run:
        windows = db.query(VideoActionWindow).filter_by(run_id=run.id).order_by(VideoActionWindow.timestamp_start).all()
        for window in windows:
            summary = json.loads(window.summary or "{}") if window.summary else {}
            entries.append({"kind": "action_window", "source_id": source_id, "window_id": window.id,
                            "timestamp_start": window.timestamp_start, "timestamp_end": window.timestamp_end,
                            "transcript_segments": json.loads(window.transcript_json or "[]"),
                            "summary": summary, "comparison": json.loads(window.comparison_json or "{}"),
                            "candidate_ids": json.loads(window.candidate_ids_json or "[]"),
                            "selection_reason": window.selection_reason, "status": window.status,
                            "evidence_types": ["candidate_window"]})
    for sequence in sequences:
        entries.append({"source_id": source_id, "timestamp_start": sequence.timestamp_start, "timestamp_end": sequence.timestamp_end,
                        "sequence": json.loads(sequence.evidence_json), "evidence_types": ["visual_sequence"]})
    for chunk in chunks:
        if chunk.start_seconds is not None:
            entries.append({"source_id": source_id, "timestamp_start": chunk.start_seconds, "timestamp_end": chunk.end_seconds or chunk.start_seconds,
                            "audio_text": chunk.text, "reference": chunk.source_ref, "evidence_types": ["audio"]})
    return sorted(entries, key=lambda item: (item["timestamp_start"], item["timestamp_end"]))

@app.get("/api/sources/{source_id}/frames/{frame_id}")
def get_video_frame(source_id: int, frame_id: int, db: Session=Depends(get_db)):
    frame = db.query(VideoFrame).filter_by(id=frame_id, source_id=source_id).one_or_none()
    if not frame: raise HTTPException(404, "Frame not found.")
    path = Path(frame.frame_path).resolve()
    if not path.is_relative_to((STORAGE_DIR / "frames" / str(source_id)).resolve()) or not path.is_file(): raise HTTPException(404, "Frame file unavailable.")
    return FileResponse(path, media_type="image/jpeg")

@app.get("/api/sources/{source_id}/video-analysis/{run_id}/contact-sheets/{sheet_number}")
def get_contact_sheet(source_id: int, run_id: int, sheet_number: int, db: Session=Depends(get_db)):
    run = db.query(VideoPipelineRun).filter_by(id=run_id, source_id=source_id).one_or_none()
    if not run or sheet_number < 1: raise HTTPException(404, "Contact sheet not found.")
    path = (STORAGE_DIR / "frames" / str(source_id) / "runs" / str(run_id) / "contact-sheets" / f"sheet-{sheet_number:02d}.jpg").resolve()
    root = (STORAGE_DIR / "frames" / str(source_id) / "runs" / str(run_id) / "contact-sheets").resolve()
    if not path.is_relative_to(root) or not path.is_file(): raise HTTPException(404, "Contact sheet not found.")
    return FileResponse(path, media_type="image/jpeg")

@app.get("/api/sources/{source_id}/knowledge-discovery")
def knowledge_discovery(source_id: int, db: Session=Depends(get_db)):
    questions = db.query(DiscoveryQuestion).filter_by(source_id=source_id).order_by(DiscoveryQuestion.timestamp_seconds, DiscoveryQuestion.id).all()
    knowledge = db.query(DecisionKnowledge).filter_by(source_id=source_id).order_by(DecisionKnowledge.timestamp_seconds).all()
    run = db.query(VideoPipelineRun).filter_by(source_id=source_id).order_by(VideoPipelineRun.id.desc()).first()
    windows = db.query(VideoActionWindow).filter_by(run_id=run.id).order_by(VideoActionWindow.timestamp_start).all() if run else []
    comparisons = {item.window_id: item for item in db.query(SOPComparison).filter_by(run_id=run.id).all()} if run else {}
    video_questions = db.query(VideoExpertQuestion).filter_by(run_id=run.id).order_by(VideoExpertQuestion.timestamp_seconds, VideoExpertQuestion.id).all() if run else []
    proposals = db.query(SOPImprovementProposal).filter_by(source_id=source_id).order_by(SOPImprovementProposal.created_at.desc()).all()
    video_sop_versions = (db.query(VideoSOPDraftVersion).filter_by(source_id=source_id)
        .order_by(VideoSOPDraftVersion.version.desc(), VideoSOPDraftVersion.id.desc()).all())
    video_sop_drafts = []
    for version in video_sop_versions:
        procedure = db.get(Procedure, version.procedure_id)
        if not procedure:
            continue
        video_sop_drafts.append({"version": version.version, "run_id": version.run_id,
            "procedure": procedure_out(procedure), "evidence_sha256": version.evidence_sha256,
            "evidence_manifest": json.loads(version.evidence_manifest_json or "[]"),
            "warnings": json.loads(version.warnings_json or "[]"), "review_status": version.review_status,
            "created_at": version.created_at.isoformat() if version.created_at else None})
    return {"questions": [{"id": q.id, "frame_id": q.frame_id, "timestamp_seconds": q.timestamp_seconds, "dimension": q.dimension, "question": q.question, "answer": q.answer, "status": q.answer_status, "correction": q.correction} for q in questions],
            "video_questions": [{"id": q.id, "window_id": q.window_id, "frame_id": q.frame_id,
                                 "timestamp_seconds": q.timestamp_seconds, "category": q.category, "question": q.question,
                                 "rationale": q.rationale, "evidence": json.loads(q.evidence_json or "[]"),
                                 "answer": q.answer, "status": q.answer_status, "dismissal_reason": q.dismissal_reason,
                                 "proposal_id": next((p.id for p in proposals if json.loads(p.evidence_json or "{}").get("question_id") == q.id), None)} for q in video_questions],
            "action_windows": [{"id": w.id, "timestamp_start": w.timestamp_start, "timestamp_end": w.timestamp_end,
                                "summary": json.loads(w.summary or "{}") if w.summary else {},
                                "transcript_segments": json.loads(w.transcript_json or "[]"),
                                "selection_reason": w.selection_reason, "status": w.status,
                                "comparison": ({"relationship": comparisons[w.id].relationship,
                                                "explanation": comparisons[w.id].explanation,
                                                "references": json.loads(comparisons[w.id].references_json or "[]")} if w.id in comparisons else {})} for w in windows],
            "proposals": [{"id": p.id, "window_id": p.window_id, "procedure_id": p.procedure_id,
                           "title": p.title, "proposed_text": p.proposed_text, "rationale": p.rationale,
                           "evidence": json.loads(p.evidence_json or "[]"), "verification_notes": p.verification_notes,
                           "status": p.status, "reviewer_note": p.reviewer_note,
                           "legacy_question_level": bool(json.loads(p.evidence_json or "{}").get("question_id"))} for p in proposals],
            "video_sop_drafts": video_sop_drafts,
            "decision_knowledge": [{"id": k.id, "timestamp_seconds": k.timestamp_seconds, "context": k.context, "observed_cue": k.observed_cue, "interpretation": k.interpretation, "decision": k.decision, "action": k.action, "outcome": k.outcome, "exceptions": k.exceptions, "limits": k.limits, "verification_status": k.verification_status} for k in knowledge]}


@app.get("/api/sources/{source_id}/video-sop-drafts")
def video_sop_drafts(source_id: int, db: Session=Depends(get_db)):
    """Return source-level draft versions without loading interview/question records."""
    source = db.get(Source, source_id)
    if not source or source.source_type != "video":
        raise HTTPException(404, "Video source not found.")
    versions = (db.query(VideoSOPDraftVersion).filter_by(source_id=source_id)
                .order_by(VideoSOPDraftVersion.version.desc(), VideoSOPDraftVersion.id.desc()).all())
    drafts = []
    for version in versions:
        procedure = db.get(Procedure, version.procedure_id)
        if not procedure:
            continue
        drafts.append({"version": version.version, "run_id": version.run_id,
            "procedure": procedure_out(procedure), "evidence_sha256": version.evidence_sha256,
            "evidence_manifest": json.loads(version.evidence_manifest_json or "[]"),
            "warnings": json.loads(version.warnings_json or "[]"), "review_status": version.review_status,
            "created_at": version.created_at.isoformat() if version.created_at else None})
    return {"video_sop_drafts": drafts}


def _time_label(seconds: float) -> str:
    whole = max(0, int(seconds))
    return f"{whole // 60:02d}:{whole % 60:02d}"


def _video_sop_response(procedure, version, evidence, reused: bool):
    result = procedure_out(procedure)
    result["generation"] = {
        "kind": "full_video_sop", "version": version.version, "run_id": version.run_id,
        "evidence_sha256": version.evidence_sha256, "review_status": version.review_status,
        "reused_existing_draft": reused, "evidence_counts": evidence["counts"],
        "coverage_warnings": json.loads(version.warnings_json or "[]"),
        "evidence_manifest": json.loads(version.evidence_manifest_json or "[]"),
        "generation_mode": "quoted_transcript_fallback" if any(
            item.get("kind") == "draft_generation_fallback" for item in json.loads(version.evidence_manifest_json or "[]")
        ) else "local_model",
        "prompt_version": version.prompt_version,
    }
    return result


def _create_video_sop_draft(source: Source, language: str, db: Session):
    try:
        evidence = assemble_transcript_sop_evidence(db, source)
    except VideoSOPEvidenceError as exc:
        logger.info("Full-video SOP refused: source_id=%s status=%s reason_category=%s",
                    source.id, exc.status_code, "transcript_evidence_unavailable")
        raise HTTPException(exc.status_code, str(exc)) from exc

    # Keep the persisted run association null: analysis records are not inputs or
    # prerequisites for this transcript-only generation workflow.
    evidence["run_id"] = None
    logger.info("Transcript-only video SOP evidence assembled: source_id=%s counts=%s",
                source.id, evidence["counts"])
    existing = (db.query(VideoSOPDraftVersion).filter_by(
        source_id=source.id, evidence_sha256=evidence["evidence_sha256"])
        .order_by(VideoSOPDraftVersion.version.desc()).first())
    if existing:
        procedure = db.get(Procedure, existing.procedure_id)
        if procedure:
            logger.info("Full-video SOP idempotent reuse: source_id=%s run_id=%s procedure_id=%s version=%s status=%s",
                        source.id, evidence["run_id"], procedure.id, existing.version, procedure.status)
            return _video_sop_response(procedure, existing, evidence, reused=True)

    version = (db.query(VideoSOPDraftVersion).filter_by(source_id=source.id)
               .order_by(VideoSOPDraftVersion.version.desc()).first())
    next_version = (version.version + 1) if version else 1
    try:
        data = draft_procedure(
            "uploaded video transcript", evidence["action_evidence"], output_language=language,
            context_excerpts=[], transcript_only=True,
            force_consistency_check=True, output_schema=PROCEDURE_RESPONSE_SCHEMA,
            require_full_video_coverage=True, fallback_title=source.filename, diagnostic_source_id=source.id,
        )
    except LocalModelError as exc:
        logger.warning("Full-video SOP model output failed validation: source_id=%s run_id=%s model=%s error_type=%s",
                       source.id, evidence["run_id"], OLLAMA_MODEL, type(exc).__name__)
        raise HTTPException(502, str(exc)) from exc
    except ValueError as exc:
        inventory = ", ".join(
            f"{key.replace('_', ' ')}={value}" for key, value in evidence["counts"].items())
        unresolved = [warning for warning in evidence["warnings"]
                      if "unresolved" in warning.lower() or "partially complete" in warning.lower()]
        detail = (f"{exc} Evidence inventory: {inventory}. No draft was saved. "
                  "This is a source-evidence/generation refusal; the stored video analysis was retained.")
        if unresolved:
            detail += " Review the identified coverage gaps before retrying: " + " ".join(unresolved)
        else:
            detail += " Review or correct the stored transcript, then retry. Video analysis is not used by SOP generation and does not need to be rerun."
        logger.info("Full-video SOP generation refused: source_id=%s run_id=%s run_status=%s error_type=%s counts=%s unresolved_warnings=%s",
                    source.id, evidence["run_id"], evidence["run_status"], type(exc).__name__,
                    evidence["counts"], len(unresolved))
        raise HTTPException(422, detail) from exc
    except Exception as exc:
        logger.exception("Full-video SOP model error: source_id=%s run_id=%s model=%s",
                         source.id, evidence["run_id"], OLLAMA_MODEL)
        raise HTTPException(502, "The local procedure model could not complete the full-video draft. Check Ollama/model availability and retry; no draft was saved.") from exc

    if not isinstance(data.get("title"), str) or not data["title"].strip():
        logger.error("Full-video SOP title contract failed after model validation: source_id=%s run_id=%s", source.id, evidence["run_id"])
        raise HTTPException(502, "The local procedure model did not provide a usable title. No draft was saved; retry after checking Ollama output.")
    if data.get("title_fallback_used"):
        logger.info("Full-video SOP using normalized source-name title fallback: source_id=%s run_id=%s",
                    source.id, evidence["run_id"])

    warnings = list(data.get("warnings", []))
    if data.get("evidence_quality") == "limited":
        limited_note = "LIMITED DRAFT: " + (data.get("validation_explanation") or "Only some actions are supported; fill the evidence gaps during expert review.")
        if not any(str(warning).startswith("LIMITED DRAFT:") for warning in warnings):
            warnings.append(limited_note)
    try:
        procedure = Procedure(
            title=data["title"],
            description=data.get("description") or "Full-video draft generated from the assembled timestamped evidence; expert verification required.",
            tools_json=json.dumps(data.get("tools", []), ensure_ascii=False),
            warnings_json=json.dumps(warnings, ensure_ascii=False),
            steps_json=json.dumps(data.get("steps", []), ensure_ascii=False),
            source_id=source.id, status="draft",
        )
        db.add(procedure)
        db.flush()
        record = VideoSOPDraftVersion(
            source_id=source.id, run_id=evidence["run_id"], procedure_id=procedure.id,
            version=next_version, evidence_sha256=evidence["evidence_sha256"],
            evidence_manifest_json=json.dumps(evidence["manifest"], ensure_ascii=False),
            warnings_json=json.dumps(evidence["warnings"], ensure_ascii=False),
            model_name=OLLAMA_MODEL, prompt_version=VIDEO_SOP_PROMPT_VERSION, review_status="draft",
        )
        db.add(record)
        db.commit()
        db.refresh(procedure)
        db.refresh(record)
    except IntegrityError as exc:
        db.rollback()
        concurrent = (db.query(VideoSOPDraftVersion).filter_by(
            source_id=source.id, evidence_sha256=evidence["evidence_sha256"])
            .order_by(VideoSOPDraftVersion.version.desc()).first())
        if concurrent:
            existing_procedure = db.get(Procedure, concurrent.procedure_id)
            if existing_procedure:
                logger.info("Full-video SOP concurrent idempotency reuse: source_id=%s run_id=%s procedure_id=%s version=%s",
                            source.id, evidence["run_id"], existing_procedure.id, concurrent.version)
                return _video_sop_response(existing_procedure, concurrent, evidence, reused=True)
        logger.exception("Full-video SOP version persistence conflict: source_id=%s run_id=%s version=%s",
                         source.id, evidence["run_id"], next_version)
        raise HTTPException(409, "Video evidence changed while the draft was being saved. Refresh the source workspace and retry.") from exc
    except Exception as exc:
        db.rollback()
        logger.exception("Full-video SOP persistence failed: source_id=%s run_id=%s version=%s",
                         source.id, evidence["run_id"], next_version)
        raise HTTPException(500, "The full-video draft was generated but could not be saved. No success was reported; retry after checking the database.") from exc
    logger.info("Full-video SOP draft persisted: source_id=%s run_id=%s procedure_id=%s version=%s model=%s prompt_version=%s evidence_sha256=%s",
                source.id, evidence["run_id"], procedure.id, record.version, OLLAMA_MODEL,
                VIDEO_SOP_PROMPT_VERSION, record.evidence_sha256)
    return _video_sop_response(procedure, record, evidence, reused=False)


@app.post("/api/sources/{source_id}/video-expert-questions")
def generate_video_expert_questions(source_id: int, db: Session=Depends(get_db)):
    source = db.get(Source, source_id)
    if not source or source.source_type != "video": raise HTTPException(404, "Video source not found.")
    run = db.query(VideoPipelineRun).filter_by(source_id=source_id).order_by(VideoPipelineRun.id.desc()).first()
    if not run or run.status not in {"completed", "partially_completed"}:
        raise HTTPException(422, "Complete video analysis before generating timestamp-linked expert questions.")
    existing = db.query(VideoExpertQuestion).filter_by(run_id=run.id).count()
    if existing:
        return knowledge_discovery(source_id, db)
    windows = db.query(VideoActionWindow).filter_by(run_id=run.id).order_by(VideoActionWindow.timestamp_start).all()
    def question_priority(window):
        summary = json.loads(window.summary or "{}") if window.summary else {}
        comparison = json.loads(window.comparison_json or "{}")
        transcript = json.loads(window.transcript_json or "[]")
        return (len(summary.get("uncertainties", [])) +
                (2 if comparison.get("relationship") in {"ambiguous", "human_comparison_required"} else 0) +
                (1 if any(str(item.get("text", "")).strip() for item in transcript) else 0),
                -window.timestamp_start)
    created = 0
    for window in sorted(windows, key=question_priority, reverse=True):
        transcript = json.loads(window.transcript_json or "[]")
        transcript_text = " ".join(str(item.get("text", "")).strip() for item in transcript if item.get("text", "").strip())[:360]
        summary = json.loads(window.summary or "{}") if window.summary else {}
        frame_ids_in_window = json.loads(window.frame_ids_json or "[]")
        window_frames = (db.query(VideoFrame).filter(VideoFrame.source_id == source_id,
            VideoFrame.id.in_(frame_ids_in_window), VideoFrame.analysis_status == "completed")
            .order_by(VideoFrame.timestamp_seconds).all() if frame_ids_in_window else [])
        frame_observations = [{"frame_id": frame.id, "timestamp_seconds": frame.timestamp_seconds,
            "description": json.loads(frame.analysis_json or "{}").get("description", "")} for frame in window_frames
            if json.loads(frame.analysis_json or "{}").get("description", "").strip()]
        visual_text = " ".join([str(summary.get("sequence_summary", "")),
                                 " ".join(str(item) for item in summary.get("visible_changes", [])),
                                 " ".join(item["description"] for item in frame_observations)]).strip()[:360]
        if not transcript_text and not visual_text:
            continue
        timestamp = (window.timestamp_start + window.timestamp_end) / 2
        japanese = bool(re.search(r"[\u3040-\u30ff\u3400-\u9fff]", transcript_text))
        if transcript_text:
            if japanese:
                question = f"{_time_label(timestamp)}付近の文字起こしは「{transcript_text}」です。この説明は実作業のどの判断や動作を指していますか。次に進む前に学習者が確認すべき手がかりは何ですか。"
            else:
                question = f"The transcript near {_time_label(timestamp)} says “{transcript_text}”. What work action or decision does this refer to, and what cue should a learner verify before proceeding?"
            rationale = "Generated from the timestamp-aligned transcript. The transcript may contain recognition errors and must be checked against the recording."
        else:
            if japanese:
                question = f"{_time_label(timestamp)}付近には未検証の視覚観察があります。「{visual_text}」。この場面が作業上の意味を持つ場合、何を示していますか。学習者が確認すべき手がかりは何ですか。"
            else:
                question = f"Unverified visual observations near {_time_label(timestamp)} say “{visual_text}”. If this moment is relevant to the work, what does it show, and what cue should a learner verify?"
            rationale = "Generated from unverified AI visual observations, which may be incomplete or describe irrelevant footage."
        candidate_ids = json.loads(window.candidate_ids_json or "[]")
        candidates = db.query(VideoCandidate).filter(VideoCandidate.id.in_(candidate_ids)).all() if candidate_ids else []
        frame_ids = sorted({candidate.frame_id for candidate in candidates if candidate.frame_id is not None})
        evidence = {"source_id": source_id, "window_id": window.id,
                    "timestamp_start": window.timestamp_start, "timestamp_end": window.timestamp_end,
                    "transcript_segments": transcript, "visual_summary": summary,
                    "visual_frame_observations": frame_observations,
                    "visual_summary_is_unverified": bool(visual_text), "frame_ids": frame_ids,
                    "selection_reason": window.selection_reason}
        db.add(VideoExpertQuestion(run_id=run.id, source_id=source_id, window_id=window.id,
            frame_id=frame_ids[0] if frame_ids else None, timestamp_seconds=timestamp,
            category="interpretation", question=question, rationale=rationale,
            evidence_json=json.dumps(evidence, ensure_ascii=False), answer_status="pending"))
        created += 1
        if created >= app_config.VIDEO_EXPERT_QUESTION_LIMIT:
            break
    db.commit()
    if not created:
        raise HTTPException(422, "The analysis has no transcript or visual summaries suitable for grounded expert questions.")
    return knowledge_discovery(source_id, db)


@app.put("/api/video-expert-questions/{question_id}")
def save_video_expert_answer(question_id: int, body: VideoExpertAnswerRequest, db: Session=Depends(get_db)):
    question = db.get(VideoExpertQuestion, question_id)
    if not question: raise HTTPException(404, "Video expert question not found.")
    if body.status == "answered" and not body.answer.strip():
        raise HTTPException(422, "Enter the expert's answer before marking this question answered.")
    new_answer = body.answer.strip() if body.status == "answered" else ""
    question.answer = new_answer
    question.answer_status = body.status
    question.dismissal_reason = body.dismissal_reason.strip() if body.status == "dismissed" else ""
    question.answered_at = datetime.now(timezone.utc) if body.status == "answered" else None
    db.commit()
    return {"id": question.id, "status": question.answer_status, "answered_at": question.answered_at.isoformat() if question.answered_at else None}


@app.post("/api/video-expert-questions/{question_id}/proposal")
def create_sop_proposal(question_id: int, db: Session=Depends(get_db)):
    question = db.get(VideoExpertQuestion, question_id)
    if not question: raise HTTPException(404, "Video expert question not found.")
    raise HTTPException(410, "Expert answers are stored as timestamp-linked source knowledge. Generate one full SOP draft from the video source instead.")


@app.post("/api/sop-improvement-proposals/{proposal_id}/submit")
def submit_sop_proposal(proposal_id: int, db: Session=Depends(get_db)):
    proposal = db.get(SOPImprovementProposal, proposal_id)
    if not proposal: raise HTTPException(404, "SOP proposal not found.")
    if proposal.procedure_id:
        procedure = db.get(Procedure, proposal.procedure_id)
        if procedure:
            return {"proposal_id": proposal.id, "status": proposal.status, "procedure": procedure_out(procedure)}
    if proposal.status != "draft":
        raise HTTPException(409, "This proposal is no longer ready to submit. Review its current status and expert evidence first.")
    if not proposal.proposed_text.strip(): raise HTTPException(422, "The proposal contains no text to review.")
    source = db.get(Source, proposal.source_id)
    if not source: raise HTTPException(404, "Proposal source was not found.")
    evidence = json.loads(proposal.evidence_json or "{}")
    if evidence.get("question_id"):
        raise HTTPException(410, "Legacy question-level answer proposals can no longer create individual procedures. Use the full-video SOP draft for this source.")
    timestamp = _time_label(float(evidence.get("timestamp_seconds", 0)))
    source_ref = f"{source.filename} · expert answer at {timestamp}"
    step = {"instruction": proposal.proposed_text, "timestamp": timestamp,
            "source_ref": source_ref, "evidence_type": "expert_confirmed"}
    procedure = Procedure(title=proposal.title, description=proposal.rationale,
        tools_json="[]", warnings_json=json.dumps([proposal.verification_notes]),
        steps_json=json.dumps([step], ensure_ascii=False), source_id=source.id, status="draft")
    db.add(procedure); db.flush()
    proposal.procedure_id, proposal.status = procedure.id, "submitted_for_review"
    db.commit(); db.refresh(procedure)
    return {"proposal_id": proposal.id, "status": proposal.status, "procedure": procedure_out(procedure)}

@app.post("/api/sources/{source_id}/knowledge-discovery/questions")
def generate_discovery_questions(source_id: int, db: Session=Depends(get_db)):
    source = db.get(Source, source_id)
    if not source or source.source_type != "video": raise HTTPException(404, "Video source not found.")
    existing = db.query(DiscoveryQuestion).filter_by(source_id=source_id).count()
    if existing: return knowledge_discovery(source_id, db)
    frames = db.query(VideoFrame).filter_by(source_id=source_id, analysis_status="completed").order_by(VideoFrame.timestamp_seconds).all()
    if not frames: raise HTTPException(422, "No completed visual observations are available yet.")
    transcript = [chunk for chunk in db.query(KnowledgeChunk).filter(KnowledgeChunk.source_id == source_id, KnowledgeChunk.start_seconds.is_not(None)).all() if not (chunk.source_ref or "").startswith("visual frame ·")]
    candidates = []
    for frame in frames:
        evidence = json.loads(frame.analysis_json or "{}")
        uncertainties = evidence.get("uncertainties", [])
        actions = evidence.get("observable_actions", [])
        if uncertainties or actions:
            ts = frame.timestamp_seconds
            audio = [chunk.text for chunk in transcript if chunk.start_seconds <= ts + 2 and (chunk.end_seconds or chunk.start_seconds) >= ts]
            candidates.append({"frame_id": frame.id, "timestamp": ts, "description": evidence.get("description", ""), "actions": actions, "uncertainties": uncertainties, "transcript": audio})
    candidates = candidates[:8]
    if not candidates: raise HTTPException(422, "No candidate moments with actions or stated uncertainties were found.")
    prompt = ("Generate concise, contextual interview questions for a qualified technician using only these observed video notes. "
              "Do not assume a fault, hazard, diagnosis, intent, or hidden mental state. Ask about what they noticed, interpretation, decision, next action, outcome, exceptions, limits, or coaching. "
              "Return JSON array of objects with timestamp, dimension, question. Use English.\n" + json.dumps(candidates, ensure_ascii=False))
    raw = ollama(prompt, json_mode=True)
    try: questions = json.loads(raw or "")
    except (ValueError, TypeError): raise HTTPException(503, "Local text model could not generate structured expert questions.")
    if not isinstance(questions, list): raise HTTPException(502, "Local model returned an invalid question list.")
    valid_dimensions = {"perception", "interpretation", "decision", "action", "outcome", "exceptions", "limits", "coaching"}
    for item in questions[:24]:
        if not isinstance(item, dict) or not isinstance(item.get("question"), str): continue
        stamp = float(item.get("timestamp", 0))
        matched = min(candidates, key=lambda c: abs(c["timestamp"] - stamp))
        db.add(DiscoveryQuestion(source_id=source_id, frame_id=matched["frame_id"], timestamp_seconds=matched["timestamp"],
                                 dimension=item.get("dimension") if item.get("dimension") in valid_dimensions else "interpretation", question=item["question"][:500]))
    db.commit()
    return knowledge_discovery(source_id, db)

@app.put("/api/knowledge-discovery/questions/{question_id}")
def save_discovery_answer(question_id: int, body: DiscoveryAnswerRequest, db: Session=Depends(get_db)):
    question = db.get(DiscoveryQuestion, question_id)
    if not question: raise HTTPException(404, "Expert question not found.")
    question.answer, question.answer_status, question.correction = body.answer, body.status, body.correction
    db.commit()
    return {"id": question.id, "status": question.answer_status, "answer": question.answer, "correction": question.correction}

@app.post("/api/sources/{source_id}/knowledge-discovery/synthesize")
def synthesize_decision_knowledge(source_id: int, body: DecisionKnowledgeRequest, db: Session=Depends(get_db)):
    questions = db.query(DiscoveryQuestion).filter(DiscoveryQuestion.source_id == source_id, DiscoveryQuestion.id.in_(body.question_ids), DiscoveryQuestion.answer_status == "answered").all()
    if not questions: raise HTTPException(422, "Save at least one answered expert question before synthesis.")
    answers = {q.dimension: q.answer for q in questions if q.answer.strip()}
    frame = db.get(VideoFrame, questions[0].frame_id) if questions[0].frame_id else None
    provenance = [{"question_id": q.id, "timestamp_seconds": q.timestamp_seconds, "frame_id": q.frame_id, "answer_status": q.answer_status} for q in questions]
    item = DecisionKnowledge(source_id=source_id, timestamp_seconds=questions[0].timestamp_seconds,
        context=body.context, observed_cue=body.observed_cue or (json.loads(frame.analysis_json).get("description", "") if frame else ""),
        interpretation=body.interpretation or answers.get("interpretation", ""), decision=body.decision or answers.get("decision", ""),
        action=body.action or answers.get("action", ""), outcome=body.outcome or answers.get("outcome", ""),
        exceptions=body.exceptions or answers.get("exceptions", ""), limits=body.limits or answers.get("limits", ""),
        provenance_json=json.dumps(provenance), verification_status="expert_confirmed")
    db.add(item); db.commit(); db.refresh(item)
    return {"id": item.id, "source_id": source_id, "timestamp_seconds": item.timestamp_seconds, "context": item.context,
            "observed_cue": item.observed_cue, "interpretation": item.interpretation, "decision": item.decision,
            "action": item.action, "outcome": item.outcome, "exceptions": item.exceptions, "limits": item.limits,
            "verification_status": item.verification_status, "provenance": json.loads(item.provenance_json)}

@app.put("/api/decision-knowledge/{knowledge_id}")
def update_decision_knowledge(knowledge_id: int, body: DecisionKnowledgeUpdate, db: Session=Depends(get_db)):
    item = db.get(DecisionKnowledge, knowledge_id)
    if not item: raise HTTPException(404, "Decision knowledge record not found.")
    for field in ("context", "observed_cue", "interpretation", "decision", "action", "outcome", "exceptions", "limits"):
        setattr(item, field, getattr(body, field))
    item.verification_status = "expert_reviewed"
    db.commit()
    return {"id": item.id, "source_id": item.source_id, "verification_status": item.verification_status,
            "provenance": json.loads(item.provenance_json)}

@app.post("/api/ask")
def ask(body: AskRequest, db: Session=Depends(get_db)):
    found=knowledge_index.search(db,body.question,top_k=5,source_type=body.source_type)
    sources=[]
    from .models import Source as SourceModel
    for ch in found:
        src=db.get(SourceModel,ch.source_id)
        sources.append({"chunk_id":ch.id,"source_id":ch.source_id,"filename":src.filename if src else "Unknown source","source_type":src.source_type if src else "unknown","reference":ch.source_ref,"timestamp_start":ch.start_seconds,"timestamp_end":ch.end_seconds,"excerpt":ch.text[:700]})
    # Protect source-specific filters and citation provenance at the API boundary.
    if body.source_type:
        found = [chunk for chunk in found if db.get(KnowledgeChunk, chunk.id).source_id in {
            source.id for source in db.query(Source).filter(Source.source_type == body.source_type).all()
        }]
        allowed_ids = {chunk.id for chunk in found}
        sources = [item for item in sources if item["chunk_id"] in allowed_ids]
    result = grounded_answer(
    body.question,
    found,
    sources,
    response_language=body.response_language,
    )
    return result

@app.post("/api/procedures/draft/{source_id}")
def create_draft(
    source_id: int,
    language: Literal["en", "ja"] = "en",
    db: Session = Depends(get_db),
):
    source=db.get(Source,source_id)
    if not source: raise HTTPException(404,"Source not found.")
    logger.info("SOP draft requested: source_id=%s source_type=%s language=%s", source.id, source.source_type, language)
    if source.source_type == "video":
        return _create_video_sop_draft(source, language, db)
    excerpts = load_source_excerpts(db, source_id)
    excerpts = [excerpt for excerpt in excerpts if not (excerpt.get("source_ref") or "").startswith("visual frame ·")]
    if not excerpts:
        raise HTTPException(422, "This source has no usable indexed transcript, document, visual observation, or expert-confirmed evidence. No SOP was created.")
    if any(excerpt["source_id"] != source.id for excerpt in excerpts):
        logger.error("Source isolation failure while drafting: selected_source_id=%s", source.id)
        raise HTTPException(500,"Source evidence could not be isolated safely. No SOP was created.")
    try:
        data = draft_procedure(source.filename, excerpts, output_language=language)
    except ValueError as exc:
        logger.warning("SOP draft refused: source_id=%s reason=%s", source.id, str(exc))
        raise HTTPException(422, str(exc)) from exc
    if data.get("evidence_quality") == "limited":
        limited_note = "LIMITED DRAFT: " + (data.get("validation_explanation") or "Only some actions are supported; expert review must fill evidence gaps.")
        data["warnings"] = [*data.get("warnings", []), limited_note]
        logger.info("Limited SOP draft created: source_id=%s reason=%s", source.id, limited_note)
    p=Procedure(title=data.get("title",f"Draft — {source.filename}"),description=data.get("description","AI draft; expert review required."),tools_json=json.dumps(data.get("tools",[])),warnings_json=json.dumps(data.get("warnings",[])),steps_json=json.dumps(data.get("steps",[])),source_id=source.id,status="draft")
    db.add(p); db.commit(); db.refresh(p)
    return procedure_out(p)

@app.get("/api/procedures")
def list_procedures(status:str|None=None, db: Session=Depends(get_db)):
    q=db.query(Procedure)
    if status: q=q.filter(Procedure.status==status)
    return [procedure_out(p) for p in q.order_by(Procedure.updated_at.desc()).all()]

@app.get("/api/procedures/{procedure_id}")
def get_procedure(procedure_id:int, db: Session=Depends(get_db)):
    p=db.get(Procedure,procedure_id)
    if not p: raise HTTPException(404,"Procedure not found.")
    return procedure_out(p)

@app.put("/api/procedures/{procedure_id}")
def update_procedure(procedure_id:int, body:ProcedureUpdate, db: Session=Depends(get_db)):
    p=db.get(Procedure,procedure_id)
    if not p: raise HTTPException(404,"Procedure not found.")
    if p.status=="approved":
        p.status="draft" # changing an approved SOP forces a new review
        for draft_version in db.query(VideoSOPDraftVersion).filter_by(procedure_id=p.id).all():
            draft_version.review_status = "draft"
    p.title=body.title; p.description=body.description; p.tools_json=json.dumps(body.tools); p.warnings_json=json.dumps(body.warnings); p.steps_json=json.dumps([s.model_dump() for s in body.steps]); p.updated_at=datetime.now(timezone.utc)
    for proposal in db.query(SOPImprovementProposal).filter_by(procedure_id=p.id).all():
        proposal.status = "draft"
    db.commit(); db.refresh(p)
    return procedure_out(p)

@app.post("/api/procedures/{procedure_id}/review")
def review_procedure(procedure_id:int, body:ReviewRequest, db: Session=Depends(get_db)):
    p=db.get(Procedure,procedure_id)
    if not p: raise HTTPException(404,"Procedure not found.")
    action=body.action.lower().strip()
    if action not in {"approve","reject"}: raise HTTPException(400,"action must be approve or reject")
    p.status="approved" if action=="approve" else "rejected"; p.rejection_note=body.note if action=="reject" else ""; p.updated_at=datetime.now(timezone.utc)
    for proposal in db.query(SOPImprovementProposal).filter_by(procedure_id=p.id).all():
        proposal.status = "approved" if action == "approve" else "rejected"
        proposal.reviewer_note = body.note if action == "reject" else ""
    for draft_version in db.query(VideoSOPDraftVersion).filter_by(procedure_id=p.id).all():
        draft_version.review_status = "approved" if action == "approve" else "rejected"
    db.commit(); db.refresh(p)
    return procedure_out(p)

@app.get("/api/training/procedures")
def published_procedures(db: Session=Depends(get_db)):
    return [procedure_out(p) for p in db.query(Procedure).filter(Procedure.status=="approved").order_by(Procedure.updated_at.desc()).all()]

@app.post("/api/training/progress")
def save_progress(body:ProgressRequest, db:Session=Depends(get_db)):
    p=db.get(Procedure,body.procedure_id)
    if not p or p.status!="approved": raise HTTPException(404,"Approved procedure not found.")
    row=TrainingProgress(procedure_id=p.id,trainee_name=body.trainee_name,completed_steps=body.completed_steps,total_steps=body.total_steps,quiz_score=body.quiz_score,completed=body.completed)
    db.add(row); db.commit(); db.refresh(row)
    return {"id":row.id,"procedure_id":row.procedure_id,"trainee_name":row.trainee_name,"completed_steps":row.completed_steps,"total_steps":row.total_steps,"quiz_score":row.quiz_score,"completed":row.completed}

@app.get("/api/training/progress")
def get_progress(db:Session=Depends(get_db)):
    rows=db.query(TrainingProgress).order_by(TrainingProgress.created_at.desc()).limit(100).all()
    return [{"id":r.id,"procedure_id":r.procedure_id,"trainee_name":r.trainee_name,"completed_steps":r.completed_steps,"total_steps":r.total_steps,"quiz_score":r.quiz_score,"completed":r.completed,"created_at":r.created_at.isoformat() if r.created_at else None} for r in rows]
