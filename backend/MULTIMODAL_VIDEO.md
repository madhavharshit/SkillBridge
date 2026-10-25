# Local multimodal video analysis

SkillBridge processes a video in bounded stages: FFprobe inspection, a low-resolution FFmpeg motion scan, transcript and baseline candidate scoring, merged action windows, contact-sheet screening, dense frame selection, grouped local vision inference, RAG/SOP comparison, and persisted evidence. Motion and transcript cues nominate moments; they do not prove an action or safety condition.

## Storage and resumption

PostgreSQL changes are additive: SQLAlchemy `create_all` creates the pipeline run, candidate, action-window, comparison, expert-question, and proposal tables at backend startup. Existing tables and records are not reset. Analysis stays in the current single in-process worker. Successful frame outputs, candidates, windows, comparisons, input/config hashes, and metrics are persisted so a failed run can resume. On restart, interrupted jobs are marked failed and may be retried. Retries reuse completed frame batches; `force=true` starts a separate run.

The cache key includes the uploaded video content hash, selected-source transcript content, model/prompt settings, and processing configuration. Prompt version `visual-evidence-v3` requests concise frame descriptions and retains the first complete sentence to reduce trailing model speculation; changing prompts requires a version bump to invalidate prior results. SOP comparison has its own content fingerprint so changed indexed manuals or approved procedures invalidate comparisons without forcing vision re-analysis. Approved procedure step embeddings are computed with the existing local embedding model and cached in process by content version.

## Configuration

Defaults are listed in `.env.example` and may be overridden in the local environment:

- `VISION_MODEL=qwen2.5vl:3b`, `VISION_TIMEOUT_SECONDS=240`.
- `VIDEO_FRAME_INTERVAL=2` seconds for baseline coverage.
- `VIDEO_MOTION_SCAN_FPS=5`, `VIDEO_MOTION_WIDTH=160`, `VIDEO_MOTION_HEIGHT=90`, `VIDEO_MOTION_LOCAL_THRESHOLD=0.035`.
- `VIDEO_WINDOW_RADIUS_SECONDS=2`, `VIDEO_WINDOW_MERGE_GAP_SECONDS=1`, `VIDEO_WINDOW_MAX_SECONDS=20`.
- `VIDEO_DETAIL_FPS=3`, `VIDEO_DETAIL_FRAME_BUDGET=25`.
- `VIDEO_CONTACT_CANDIDATE_LIMIT=27`, `VIDEO_CONTACT_SHEET_SIZE=9`, `VIDEO_VISION_BATCH_SIZE=2` (bounded to 6).
- `VIDEO_FRAME_MAX_DIMENSION=960`, `VIDEO_EXPERT_QUESTION_LIMIT=8`.
- `VIDEO_STRATEGY_VERSION` and `VIDEO_PROMPT_VERSION` invalidate cached outputs after strategy or prompt changes.

The inexpensive scan streams small grayscale frames and keeps localized tile changes as well as global change and scene-cut signals. Its timestamps are taken from FFmpeg `showinfo` presentation timestamps. Contact screening can use a deterministic motion/transcript/baseline fallback if vision screening fails; fallback use is recorded in warnings. Windows outside the screen/detail budget remain visible as not selected for detailed analysis, and the run is marked partially complete rather than claiming full video coverage.

## Ollama vision contract

Vision requests use the configured Ollama host's `/api/chat` endpoint, `stream: false`, and images in the user message's `images` array. Detailed frame batches and contact sheets use strict JSON Schema output; a schema-format HTTP 400 gets one JSON-mode fallback. Invalid content gets at most one retry. The outer HTTP envelope is unwrapped from `message.content`, required fields and frame IDs are checked, and malformed-output logs are truncated without image data. A failed batch is recorded as unresolved and does not discard successful batches. Single-frame observation uses short plain text; the backend keeps its first complete sentence and attaches source/frame/timestamp metadata.

FFmpeg must be installed and on `PATH`. Original videos remain under `STORAGE_DIR/uploads`; derived frames and contact sheets are under `STORAGE_DIR/frames/{source_id}/runs/{run_id}`. No cloud vision or transcription service is used.

## API and UI workflow

- `POST /api/sources/{source_id}/video-analysis` starts or resumes analysis. Use `?force=true` for a new run.
- `GET /api/sources/{source_id}/video-analysis` returns job status, current stage, metrics, warnings, video metadata, run ID, and contact-sheet URLs.
- `GET /api/sources/{source_id}/evidence-timeline` returns the latest run's frames, transcript alignment, candidate reasons, action windows, and SOP comparisons. Legacy records are used when no new run exists.
- `GET /api/sources/{source_id}/frames/{frame_id}` serves only a frame belonging to that source.
- `GET /api/sources/{source_id}/video-analysis/{run_id}/contact-sheets/{sheet_number}` serves a sheet belonging to that source and run.
- `POST /api/sources/{source_id}/video-expert-questions` creates a bounded, idempotent set of questions from timestamped transcript or explicitly unverified visual evidence. `PUT /api/video-expert-questions/{question_id}` saves an answer or dismissal.
- `POST /api/video-expert-questions/{question_id}/proposal` copies a saved answer verbatim into a traceable improvement proposal. `POST /api/sop-improvement-proposals/{proposal_id}/submit` creates a normal draft procedure for the existing review studio. It is never approved automatically.
- Legacy knowledge-discovery question, answer, synthesis, and decision-record endpoints remain available.

The React Expert Knowledge page displays real stages, counts, warnings, thumbnails, action windows, comparisons, expert questions, and proposal state. Video upload retains the source if local transcription fails; the user can still request video-only analysis. A failed or empty transcript is reported as such and is never presented as successful transcription.

Existing document/video RAG, SOP drafting and human review, approved-only training, mobile progress, and AR integration remain on their existing endpoints. AI visual observations remain unverified evidence and are not instructions or safety approval.

## Local commands

From `backend/`, start the API using the project's existing environment:

```sh
. .venv/bin/activate
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

After analysis, inspect `GET /api/sources/{source_id}/video-analysis`, `GET /api/sources/{source_id}/evidence-timeline`, and `GET /api/sources/{source_id}/knowledge-discovery`. Backend logs include each stage and run metrics. Increase the relevant frame budget or screening limits in the local environment before requesting a new run with `force=true` if additional visual coverage is needed.

The system does not validate technical accuracy, confirm that an expert answer is safe, or prove that an entire video was understood. Reviewers must compare each proposal with the original recording and governing technical documentation before approval.
