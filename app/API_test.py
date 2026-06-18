import timm
import io, os, time
import numpy as np
import torch, torch.nn as nn
import matplotlib.pyplot as plt

from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
from PIL import Image
from scipy import io as sio, signal as sci_signal
from torchvision import transforms
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from pydantic import BaseModel

from app.csdl import get_db
from app.model_db import Prediction, stft_image, job_status

# ── Cấu hình ──────────────────────────────────────────────────────────────
drone_names    = ["Phantom 4", "Mavic Zoom", "Mavic Enterprise"]
upload_dir     = os.getenv("upload_dir", "upload")
model_path     = os.path.join(upload_dir, os.getenv("model_file", "model_.pth"))#model_.pth là model overlap =128, model.pth overlap =1024
device         = torch.device("cuda" if torch.cuda.is_available() else "cpu")

bin_threshold  = float(os.getenv("BIN_THRESHOLD",  "0.5"))  # có drone hay không
type_threshold = float(os.getenv("TYPE_THRESHOLD", "0.5"))  # ngưỡng từng loại
fq_threshold   = float(os.getenv("FQ_THRESHOLD",   "0.1"))  # tỉ lệ frame tối thiểu
min_type_freq  = float(os.getenv("MIN_TYPE_FREQ",  "0.075")) # tần suất tối thiểu để báo 1 loại
embedding_size = int(os.getenv("embedding_size",   "128"))

Fs             = 150e6
segment_length = 699032
num_segments   = 1440
M              = 2048
overlap        = 128
batch_size     = 20
mat_key        = os.getenv("MAT_KEY", "Y")

# ── FastAPI ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="SDrone API",
    description="Upload file .mat -> Process -> save MySQL",
    version="5.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

# ── Model ──────────────────────────────────────────────────────────────────
class DroneClass(nn.Module):
    def __init__(self, num_drone_types: int = 3, embedding_size: int = 128):
        super().__init__()
        self.backbone = timm.create_model("efficientnet_b0", pretrained=False)
        n_features = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Identity()

        self.binary_head = nn.Sequential(
            nn.Linear(n_features, 1),
            nn.Sigmoid(),
        )
        self.embedding = nn.Sequential(
            nn.Linear(n_features, embedding_size),
            nn.BatchNorm1d(embedding_size),
        )
        self.drone_type_head = nn.Sequential(
            nn.Linear(embedding_size, num_drone_types),
            nn.Sigmoid(),
        )

    def forward(self, x):
        features   = self.backbone(x)
        pre_bin    = self.binary_head(features)
        embeddings = self.embedding(features)
        pre_type   = self.drone_type_head(embeddings)
        return pre_bin, pre_type, embeddings


_model: Optional[DroneClass] = None

def get_model() -> DroneClass:
    global _model
    if _model is None:
        m = DroneClass(embedding_size=embedding_size)
        if os.path.exists(model_path):
            state = torch.load(model_path, map_location=device)
            m.load_state_dict(state.get("model_state_dict", state))
        m.to(device).eval()
        _model = m
    return _model

transform = transforms.Compose([
    transforms.Resize((288, 288)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# ── Pydantic schemas ───────────────────────────────────────────────────────
class resultPredict(BaseModel):
    id:                int
    drone_detected:    bool
    drone_types:       List[str]
    has_unknown:       bool
    avg_binary_score:  float
    drone_frame_ratio: float
    drone_segments:    int
    total_segments:    int
    type_detail:       dict
    process_time_ms:   int

class listHistory(BaseModel):
    id:           int
    filename:     str
    status:       str
    drone_bin:    Optional[bool]
    confidence:   Optional[float]
    drone_type:   Optional[list]
    total_images: Optional[int]
    created_at:   str

# ── Tiền xử lý ────────────────────────────────────────────────────────────
def preprocessing(mat_bytes: bytes) -> List[Image.Image]:
    """
    Đọc file .mat, lấy 1 tín hiệu 1D, cắt thành các segment,
    STFT từng segment → trả về list ảnh PIL.
    """
    buf = io.BytesIO(mat_bytes)
    try:
        mat_data = sio.loadmat(buf)
    except Exception as e:
        raise ValueError(f"Không đọc được file .mat: {e}")

    if mat_key not in mat_data:
        available = [k for k in mat_data if not k.startswith("_")]
        raise ValueError(f"Không tìm thấy key '{mat_key}'. Có: {available}")

    signal = mat_data[mat_key].squeeze()
    if signal.ndim != 1:
        raise ValueError(
            f"Kỳ vọng tín hiệu 1D, nhận được shape {signal.shape}. "
            "File này có nhiều signal — không hỗ trợ."
        )

    colormap = plt.get_cmap("jet")
    images: List[Image.Image] = []

    for i in range(num_segments):
        start, end = i * segment_length, (i + 1) * segment_length
        if end > len(signal):
            break
        segment = signal[start:end]

        _, _, Zxx = sci_signal.stft(
            segment, fs=Fs,
            window="hann", nperseg=M, noverlap=overlap,
            nfft=M, return_onesided=False,
        )
        Zxx_shifted = np.fft.fftshift(Zxx, axes=0)
        bien_do     = np.abs(Zxx_shifted)
        phoNL       = 20 * np.log10(bien_do + 1e-12)
        phoNL_chuan = (phoNL - phoNL.min()) / (phoNL.max() - phoNL.min() + 1e-12)

        color_img = colormap(phoNL_chuan)[:, :, :3]
        img = Image.fromarray((color_img * 255).astype(np.uint8)).resize((288, 288))
        images.append(img)

    return images

# ── Inference ──────────────────────────────────────────────────────────────
def thuc_thi(images: List[Image.Image]) -> List[dict]:
    """
    Chạy model trên từng frame:
      - bin_score < bin_threshold  → không có drone
      - bin_score >= bin_threshold → loại nào > type_threshold thì dương tính
                                     không loại nào vượt ngưỡng → "Unknown"
    """
    model = get_model()
    all_bin, all_type = [], []

    for start in range(0, len(images), batch_size):
        batch = torch.stack(
            [transform(img) for img in images[start:start + batch_size]]
        ).to(device)
        with torch.no_grad():
            bin_out, type_out, _ = model(batch)

        all_bin.extend(bin_out.squeeze(1).cpu().numpy().tolist())
        all_type.extend(type_out.cpu().numpy().tolist())

    results = []
    for bin_score, type_scores in zip(all_bin, all_type):
        if bin_score < bin_threshold:
            results.append({
                "is_drone":     False,
                "binary_score": float(bin_score),
                "drone_types":  [],
                "type_scores":  [float(s) for s in type_scores],
            })
            continue

        detected_types = [
            drone_names[i]
            for i, s in enumerate(type_scores)
            if s > type_threshold
        ]
        results.append({
            "is_drone":     True,
            "binary_score": float(bin_score),
            "drone_types":  detected_types if detected_types else ["Unknown"],
            "type_scores":  [float(s) for s in type_scores],
        })

    return results

# ── Aggregate ──────────────────────────────────────────────────────────────
def aggregate(results: List[dict]) -> dict:
    from collections import Counter

    total   = len(results)
    avg_bin = float(np.mean([r["binary_score"] for r in results])) if total else 0.0

    if total == 0:
        return {
            "drone_detected":    False,
            "avg_binary_score":  0.0,
            "drone_frame_ratio": 0.0,
            "drone_segments":    0,
            "total_segments":    0,
            "drone_types":       [],
            "type_detail":       {},
            "has_unknown":       False,
        }

    drone_segs = [r for r in results if r["is_drone"]]
    n_drone    = len(drone_segs)
    drone_pct  = n_drone / total

    if drone_pct < fq_threshold:
        return {
            "drone_detected":    False,
            "avg_binary_score":  round(avg_bin, 4),
            "drone_frame_ratio": round(drone_pct, 4),
            "drone_segments":    n_drone,
            "total_segments":    total,
            "drone_types":       [],
            "type_detail":       {},
            "has_unknown":       False,
        }

    # 1 frame có thể có nhiều loại cùng lúc
    type_counts: Counter = Counter()
    for r in drone_segs:
        for t in r["drone_types"]:
            type_counts[t] += 1

    type_detail: dict = {
        t: {"count": cnt, "frequency": round(cnt / total, 4)}
        for t, cnt in type_counts.items()
    }

    has_unknown = "Unknown" in type_detail
    val_types   = sorted(
        [t for t, d in type_detail.items() if t != "Unknown" and d["frequency"] >= min_type_freq],
        key=lambda t: type_detail[t]["frequency"],
        reverse=True,
    )
    final_types = val_types if val_types else (["Unknown"] if has_unknown else [])

    return {
        "drone_detected":    True,
        "avg_binary_score":  round(avg_bin, 4),
        "drone_frame_ratio": round(drone_pct, 4),
        "drone_segments":    n_drone,
        "total_segments":    total,
        "drone_types":       final_types,
        "type_detail":       type_detail,
        "has_unknown":       has_unknown,
    }

# ── Endpoints ──────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status":       "he thong chay on",
        "device":       str(device),
        "model_loaded": _model is not None,
        "thresholds": {
            "bin":           bin_threshold,
            "type":          type_threshold,
            "fq":            fq_threshold,
            "min_type_freq": min_type_freq,
        },
    }


@app.post("/predict/mat", response_model=resultPredict)
async def predict_from_mat(
    file: UploadFile = File(...),
    db:   AsyncSession = Depends(get_db),
):
    if not file.filename.lower().endswith(".mat"):
        raise HTTPException(406, "System chỉ nhận file .mat")
    mat_bytes = await file.read()
    if not mat_bytes:
        raise HTTPException(400, "File rỗng")

    job = Prediction(filename=file.filename, status=job_status.processing)
    db.add(job)
    await db.flush()
    t0 = time.monotonic()

    try:
        images = preprocessing(mat_bytes)
        if not images:
            raise ValueError(f"Signal quá ngắn, cần >= {segment_length} samples")

        seg_results = thuc_thi(images)

        db.add_all([
            stft_image(
                predict_id=job.id,
                segment_index=idx,
                pre_bin=r["binary_score"],
                pre_phantom=r["type_scores"][0],
                pre_mavic_zoom=r["type_scores"][1],
                pre_mavic_enterprise=r["type_scores"][2],
                drone_bin=r["is_drone"],
            )
            for idx, r in enumerate(seg_results)
        ])

        agg        = aggregate(seg_results)
        elapsed_ms = int((time.monotonic() - t0) * 1000)

        job.status           = job_status.done
        job.drone_bin        = agg["drone_detected"]
        job.confidence       = agg["avg_binary_score"]
        job.drone_type       = agg["drone_types"]
        job.drone_type_score = agg["type_detail"]
        job.total_images     = agg["total_segments"]
        job.processing_time  = elapsed_ms

        return resultPredict(
            id=job.id,
            drone_detected=agg["drone_detected"],
            drone_types=agg["drone_types"],
            has_unknown=agg["has_unknown"],
            avg_binary_score=agg["avg_binary_score"],
            drone_frame_ratio=agg["drone_frame_ratio"],
            drone_segments=agg["drone_segments"],
            total_segments=agg["total_segments"],
            type_detail=agg["type_detail"],
            process_time_ms=elapsed_ms,
        )

    except Exception as e:
        job.status = job_status.error
        raise HTTPException(500, f"Lỗi xử lý nội bộ: {e}")


@app.get("/jobs", response_model=List[listHistory])
async def list_jobs(
    limit:  int = 20,
    offset: int = 0,
    db: AsyncSession = Depends(get_db),
):
    rows = (await db.execute(
        select(Prediction)
        .order_by(Prediction.created_at.desc())
        .limit(limit).offset(offset)
    )).scalars().all()

    return [
        listHistory(
            id=r.id,
            filename=r.filename,
            status=r.status,
            drone_bin=r.drone_bin,
            confidence=r.confidence,
            drone_type=r.drone_type,
            total_images=r.total_images,
            created_at=r.created_at.isoformat(),
        )
        for r in rows
    ]


@app.get("/jobs/{job_id}", summary="Thông tin chi tiết 1 lần predict")
async def get_job(job_id: int, db: AsyncSession = Depends(get_db)):
    job = await db.get(Prediction, job_id)
    if not job:
        raise HTTPException(404, "Bản ghi không tồn tại")
    return {
        "id":               job.id,
        "filename":         job.filename,
        "status":           job.status.value,
        "drone_bin":        job.drone_bin,
        "confidence":       job.confidence,
        "drone_type":       job.drone_type,
        "drone_type_score": job.drone_type_score,
        "total_images":     job.total_images,
        "processing_time":  job.processing_time,
        "created_at":       job.created_at.isoformat(),
        "updated_at":       job.updated_at.isoformat() if job.updated_at else None,
    }


@app.get("/jobs/{job_id}/segments", summary="Raw scores từng đoạn")
async def get_segments(
    job_id: int,
    limit:  int = 200,
    offset: int = 0,
    db: AsyncSession = Depends(get_db),
):
    job = await db.get(Prediction, job_id)
    if not job:
        raise HTTPException(404, "Bản ghi không tồn tại")

    rows = (await db.execute(
        select(stft_image)
        .where(stft_image.predict_id == job_id)
        .order_by(stft_image.segment_index)
        .limit(limit).offset(offset)
    )).scalars().all()

    return {
        "job_id":         job_id,
        "total_segments": job.total_images,
        "segments": [
            {
                "segment_index":        r.segment_index,
                "pre_bin":              r.pre_bin,
                "pre_phantom":          r.pre_phantom,
                "pre_mavic_zoom":       r.pre_mavic_zoom,
                "pre_mavic_enterprise": r.pre_mavic_enterprise,
                "drone_bin":            r.drone_bin,
            }
            for r in rows
        ],
    }


@app.post("/admin/upload-model", summary="Upload model.pth mới")
async def upload_model(file: UploadFile = File(...)):
    global _model
    os.makedirs(upload_dir, exist_ok=True)
    data = await file.read()
    with open(model_path, "wb") as f:
        f.write(data)
    _model = None
    return {"message": f"Saved → {model_path}", "size_mb": round(len(data) / 1024**2, 2)}