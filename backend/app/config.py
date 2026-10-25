import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
STORAGE_DIR = Path(os.getenv("STORAGE_DIR", str(BASE_DIR / "storage"))).resolve()
UPLOAD_DIR = STORAGE_DIR / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg://skillbridge:skillbridge_dev_only@localhost:5432/skillbridge",
)
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")
VISION_MODEL = os.getenv("VISION_MODEL", "qwen2.5vl:3b")
VISION_TIMEOUT_SECONDS = max(10, int(os.getenv("VISION_TIMEOUT_SECONDS", "240")))
VISION_MAX_OUTPUT_TOKENS = max(128, int(os.getenv("VISION_MAX_OUTPUT_TOKENS", "1536")))
VIDEO_FRAME_INTERVAL = max(1.0, float(os.getenv("VIDEO_FRAME_INTERVAL", "2")))
VIDEO_MAX_FRAMES = max(1, int(os.getenv("VIDEO_MAX_FRAMES", "24")))
VIDEO_FRAME_MAX_DIMENSION = max(320, int(os.getenv("VIDEO_FRAME_MAX_DIMENSION", "960")))
VIDEO_MOTION_SCAN_FPS = max(1.0, float(os.getenv("VIDEO_MOTION_SCAN_FPS", "5")))
VIDEO_MOTION_SCAN_MAX_SAMPLES = max(300, int(os.getenv("VIDEO_MOTION_SCAN_MAX_SAMPLES", "3600")))
VIDEO_BASELINE_CANDIDATE_LIMIT = max(100, int(os.getenv("VIDEO_BASELINE_CANDIDATE_LIMIT", "1800")))
VIDEO_MOTION_WIDTH = max(64, int(os.getenv("VIDEO_MOTION_WIDTH", "160")))
VIDEO_MOTION_HEIGHT = max(36, int(os.getenv("VIDEO_MOTION_HEIGHT", "90")))
VIDEO_MOTION_LOCAL_THRESHOLD = min(1.0, max(0.0, float(os.getenv("VIDEO_MOTION_LOCAL_THRESHOLD", "0.035"))))
VIDEO_WINDOW_RADIUS_SECONDS = max(0.5, float(os.getenv("VIDEO_WINDOW_RADIUS_SECONDS", "2")))
VIDEO_WINDOW_MERGE_GAP_SECONDS = max(0.0, float(os.getenv("VIDEO_WINDOW_MERGE_GAP_SECONDS", "1")))
VIDEO_WINDOW_MAX_SECONDS = max(4.0, float(os.getenv("VIDEO_WINDOW_MAX_SECONDS", "20")))
VIDEO_DETAIL_FPS = max(1.0, float(os.getenv("VIDEO_DETAIL_FPS", "3")))
VIDEO_DETAIL_FRAME_BUDGET = max(3, int(os.getenv("VIDEO_DETAIL_FRAME_BUDGET", "25")))
VIDEO_CONTACT_CANDIDATE_LIMIT = max(3, int(os.getenv("VIDEO_CONTACT_CANDIDATE_LIMIT", "27")))
VIDEO_DETAIL_COVERAGE_INTERVAL_SECONDS = max(5.0, float(os.getenv("VIDEO_DETAIL_COVERAGE_INTERVAL_SECONDS", "30")))
VIDEO_CONTACT_SHEET_SIZE = min(9, max(3, int(os.getenv("VIDEO_CONTACT_SHEET_SIZE", "9"))))
VIDEO_VISION_BATCH_SIZE = min(6, max(1, int(os.getenv("VIDEO_VISION_BATCH_SIZE", "2"))))
VIDEO_STRATEGY_VERSION = os.getenv("VIDEO_STRATEGY_VERSION", "adaptive-motion-window-v2")
VIDEO_PROMPT_VERSION = os.getenv("VIDEO_PROMPT_VERSION", "visual-evidence-v3")
VIDEO_EXPERT_QUESTION_LIMIT = max(1, min(12, int(os.getenv("VIDEO_EXPERT_QUESTION_LIMIT", "8"))))
VIDEO_SOP_CONTEXT_CHAR_LIMIT = max(4000, int(os.getenv("VIDEO_SOP_CONTEXT_CHAR_LIMIT", "20000")))
VIDEO_SOP_EVIDENCE_PACKET_CHAR_LIMIT = max(1500, int(os.getenv("VIDEO_SOP_EVIDENCE_PACKET_CHAR_LIMIT", "7000")))
CORS_ORIGINS = [x.strip() for x in os.getenv("CORS_ORIGINS", "http://localhost:5173,http://localhost:8081").split(",") if x.strip()]
