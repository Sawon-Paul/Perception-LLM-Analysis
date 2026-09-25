"""Central config. Import from here. Never hardcode a path or a key elsewhere."""
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

# --- paths ---
DATA = ROOT / "data"
RAW = DATA / "raw"
FRAMES = RAW / "frames"
DETECTIONS = DATA / "detections"
EVENTS = DATA / "events"
CACHE = DATA / "cache"
WEIGHTS = ROOT / "weights"
for _p in (RAW, FRAMES, DETECTIONS, EVENTS, CACHE, WEIGHTS):
    _p.mkdir(parents=True, exist_ok=True)

DATASET_DIR = Path(os.getenv("DATASET_DIR", str(ROOT / "Dataset")))
DEFAULT_TRACE = os.getenv("DEFAULT_TRACE", "PVS 2")


def pvs_dir(trace: str) -> Path:
    """Folder for one PVS trace, e.g. trace='PVS 2'."""
    return DATASET_DIR / trace


def slug(trace: str) -> str:
    """'PVS 2' -> 'PVS2'. Used in every output filename so traces never collide."""
    return trace.replace(" ", "").replace("/", "_")

# --- YOLO ---
POTHOLE_WEIGHTS = WEIGHTS / "yolo11s_pothole.pt"
SIGN_WEIGHTS = WEIGHTS / "yolo11s_sign.pt"
CONF_THRESHOLD = 0.25      # keep low — the LLM is what filters false positives
IOU_THRESHOLD = 0.45
DEVICE = 0                 # CUDA index, or "cpu"

# --- video / sensor sync ---
EXTRACT_FPS = 4.0          # frames pulled from video_environment.mp4
SENSOR_HZ = 100            # MPU rate; used for rolling-window sizes
JOIN_TOLERANCE_S = 0.05

# --- Overpass ---
OVERPASS_URL = os.getenv("OVERPASS_URL", "https://overpass-api.de/api/interpreter")
OVERPASS_CONTACT = os.getenv("OVERPASS_CONTACT", "unset@example.com")
OVERPASS_RADIUS_M = 30
OVERPASS_TIMEOUT_S = 30
OVERPASS_MIN_INTERVAL_S = 1.2
CACHE_GRID_DP = 4          # lat/lon rounding for cache key (~11 m)

# --- LLM ---
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
PHASE1_MODEL = "qwen-plus"
PHASE2_MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"
PHASE1_GATE = 0.30         # below this prior, skip Phase 2 entirely
CROP_PAD_PX = 24           # legacy, unused
CROP_MIN_PX = 160          # smallest region taken from the frame, in source pixels
CROP_CONTEXT_SCALE = 1.6       # context for small boxes
CROP_CONTEXT_SCALE_LARGE = 1.05  # large boxes already contain their context
CROP_UPSCALE_TO = 336      # upscale short side to this before Phase 2

# Region of interest. The bonnet fills the bottom of every dashcam frame.
ROI_TOP = 0.0
ROI_BOTTOM = 1.0           # set from src.detect.roi_check
