import timm
import io, os, time
import numpy as np
import torch, torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
 
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional, Dict
from PIL import Image
from scipy import io as sio, signal as sci_signal
from torchvision import transforms, models
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from pydantic import BaseModel
 
from app.csdl import get_db, engine, Base
from app.model_db import Prediction, stft_image, job_status

drone_names = ["Phantom 4", "Mavic Zoom", "Mavic Enterprise"]
upload_dir = os.getenv("upload_dir", "upload")
model_path = os.path.join(upload_dir, os.getenv("model_file", "model.pth"))
centroids_path = os.path.join(upload_dir, os.getenv("centroid_file", "centroids.pth"))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bin_threshold = float(os.getenv("BIN_THRESHOLD", "0.5"))
sim_threshold = float(os.getenv("TYPE_THRESHOLD", "0.75"))
fq_threshold = float(os.getenv("FQ_THRESHOLD", "0.1"))
min_type_freq = float(os.getenv("MIN_TYPE_FREQ", "0.05"))
embedding_size = int(os.getenv("embedding_size", "128"))
min_drone_frames = float(os.getenv("min_drone_frames",  "0.05"))
Fs = 150e6
segment_length = 699032
num_segments = 1440

M = 2048
overlap = 1024
batch_size = 20
mat_key = os.getenv("MAT_KEY", "Y")

app = FastAPI(
    title = "SDrone API",
    description = "Upload file .mat -> Process -> save MySQL",
    version = "3.0.0",
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

class DroneClass(nn.Module):
    def __init__(self, num_drone_types=3, embedding_size=128):
        super().__init__()
        self.backbone = timm.create_model('efficientnet_b0', pretrained=False)
        n_features = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Identity()
        self.binary_head = nn.Sequential(
            nn.Linear(n_features, 1),
            nn.Sigmoid()
        )

        self.embedding = nn.Sequential(
            nn.Linear(n_features, embedding_size),
            nn.BatchNorm1d(embedding_size)
        )
        self.drone_type_head = nn.Sequential(
            nn.Linear(embedding_size, num_drone_types),
            nn.Sigmoid()
        )
    def forward(self, x):
        features = self.backbone(x)
        pre_bin = self.binary_head(features)
        embeddings = self.embedding(features)
        pre_type = self.drone_type_head(embeddings)
        return pre_bin, pre_type,embeddings
_model: Optional[DroneClass] = None
_centroids: Optional[Dict[int, torch.Tensor]] = None

def get_model()->DroneClass:
    global _model
    if _model is None:
        m = DroneClass()
        if os.path.exists(model_path):
            state = torch.load(model_path, map_location=device)
            m.load_state_dict(state.get("model_state_dict", state))
        m.to(device).eval()
        _model = m
    return _model

def get_centroids() -> Dict[int, torch.Tensor]:
    global _centroids
    if _centroids is None:
        if os.path.exists(centroids_path):
            _centroids = torch.load(centroids_path, map_location=device)
            print(f"load centroids success")
        else:
            print(f"not found centroids file")
            _centroids = {}
    return _centroids

transform = transforms.Compose([
    transforms.Resize((288,288)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def preprocessing(mat_bytes: bytes) -> List[Image.Image]:
    buf = io.BytesIO(mat_bytes)
    try:
        mat_data = sio.loadmat(buf)
    except Exception as e:
        raise ValueError(f"không đọc được file .mat {e}")
    
    if mat_key not in mat_data:
        available = [k for k in mat_data if not k.startswith("_")]
        raise ValueError(f"không tìm thấy key '{mat_key}' trong file {available}")
    
    signal_data = mat_data[mat_key].squeeze()
    colormap = plt.get_cmap('jet')
    images = []

    segments = []
    for i in range(num_segments):
        start = i * segment_length
        end = start + segment_length
        if end <= len(signal_data):
            segment = signal_data[start:end]
            segments.append(segment)
        else:
            break
    for idx, segment in enumerate(segments):
        f, t, Zxx = sci_signal.stft(
            segment,
            fs = Fs,
            window = 'hann',
            nperseg = M,
            noverlap = overlap,
            nfft = M,
            return_onesided = False
        )

        Zxx_shifted = np.fft.fftshift(Zxx, axes=0)
        f_shiffted = np.fft.fftshift(f)

        bien_do = np.abs(Zxx_shifted)
        phoNL = 20 * np.log10(bien_do + 1e-12)

        phoNL_min = phoNL.min()
        phoNL_max = phoNL.max()
        phoNL_chuan = (phoNL - phoNL_min) / (phoNL_max - phoNL_min + 1e-12)
        
        color_img = colormap(phoNL_chuan)[:, :, :3]
        color_img_uint8 = (color_img * 255).astype(np.uint8)

        img = Image.fromarray(color_img_uint8)
        img_resized = img.resize((288, 288))
        
        images.append(img_resized)
    return images

def thuc_thi(images: List[Image.Image]):
    model = get_model()
    centroids = get_centroids()

    all_bin = []
    all_type = []
    all_emb = []

    for start in range(0, len(images), batch_size):
        batch = torch.stack(
            [transform(img) for img in images[start:start + batch_size]]
        ).to(device)

        with torch.no_grad():
            bin_out, type_out, emb_out = model(batch)
        
        all_bin.extend(bin_out.squeeze(1).cpu().numpy().tolist())
        all_type.extend(type_out.cpu().numpy().tolist())
        all_emb.extend(emb_out.cpu())
    results = []
    for bin_score, type_score, emb in zip(all_bin, all_type, all_emb):
        if bin_score < bin_threshold:
            results.append({
                "is_drone": False,
                "binary_score": float(bin_score),
                "type": None,
                "similarity": None,
                "type_scores": [float(s) for s in type_score],
            })
            continue
        best_class = None
        best_sim = -1.0

        if centroids:
            for cls_id, center in centroids.items():
                sim = F.cosine_similarity(
                    emb.unsqueeze(0).to(device),
                    center.unsqueeze(0).to(device),
                ).item()
                if sim > best_sim:
                    best_sim = sim
                    best_class = cls_id
            drone_type = "Unknown" if best_sim < sim_threshold else drone_names[best_class]
            results.append({
                "is_drone": True,
                "binary_score": float(bin_score),
                "type": drone_type,
                "similarity": float(best_sim),
                "type_scores": [float(s) for s in type_score],
            })
        else:
            best_idx = int(np.argmax(type_score))
            results.append({
                "is_drone": True,
                "binary_score": float(bin_score),
                "type": drone_names[best_idx],
                "similarity": float(max(type_score)),
                "type_scores": [float(s) for s in type_score],
            })
    return results
def aggregate(results: List[dict]) ->dict:
    from collections import defaultdict, Counter
    total = len(results)
    if total == 0:
        return {
            "drone_detected": False,
            "avg_binary_score": 0.0,
            "drone_frame_ratio": 0.0,
            "drone_segments": 0,
            "total_segments": 0,
            "drone_types": [],
            "type_detail": {},
            "has_unknown": False,
        }
    avg_bin = float(np.mean([r["binary_score"]for r in results]))
    drone_segs = [r for r in results if r["is_drone"]]
    n_drone = len(drone_segs)
    drone_pct = n_drone/total
    drone_detected = drone_pct >= fq_threshold
    if not drone_detected:
        return{
            "drone_detected": False,
            "avg_binary_score": round(avg_bin,4),
            "drone_frame_ratio": round(drone_pct, 4),
            "drone_segments": n_drone,
            "total_segments": total,
            "drone_types": [],
            "type_detail": {},
            "has_unknown": False,
        }
    type_sims: dict = defaultdict(list)
    type_counts: Counter = Counter()
    for r in drone_segs:
        t = r["type"] if r["type"] is not None else "Unknown"
        sim = r["similarity"]
        type_counts[t] += 1
        if sim is not None:
            type_sims[t].append(sim)
    type_detail = {}
    for t,cnt in type_counts.items():
        fq = cnt/total
        avg_sim = float(np.mean(type_sims[t])) if type_sims[t] else 0.0
        score = avg_sim * fq
        type_detail[t] = {
            "count": cnt,
            "frequency": round(fq, 4),
            "avg_similarity": round(avg_sim, 4),
            "score": round(score,4),
        }
    val_types = [
        t for t,d in type_detail.items()
        if t != "Unknown" and d["frequency"] >= min_drone_frames
    ]
    val_types.sort(key=lambda t: type_detail[t]["score"], reverse=True)
    has_unknown = type_counts.get("Unknown",0)>0
    final_types = val_types if val_types else (["Unknown"] if has_unknown else [])
    return {
        "drone_detected": True,
        "avg_binary_score": round(avg_bin, 4),
        "drone_frame_ratio": round(drone_pct, 4),
        "drone_segments": n_drone,
        "total_segments": total,
        "drone_types": final_types,
        "type_detail": type_detail,
        "has_unknown": has_unknown,
    }

class resultPredict(BaseModel):
    id: int
    drone_detected: bool #co/ko drone
    drone_types: List[str] #tra ve loai drone ma he thong nhan dang dua ra kqua
    has_unknown: bool #co nam trong ds drone co trong db ko?
    avg_binary_score: float #dua ra xs ty le co drone/ko drone
    drone_frame_ratio: float #ty le anh duoc nhan la drone
    drone_segments: int #so luong anh co drone
    total_segments: int # tong so luong anh cua 1 fle .mat
    type_detail: dict 
    process_time_ms: int #thoi gian xu ly 1 file .mat trong khoang x ms

class listHistory(BaseModel):
    id: int
    filename: str
    status: str
    drone_bin: Optional[bool]
    confidence: Optional[float]
    drone_type: Optional[list]
    total_images: Optional[int]
    created_at: str

@app.get("/health") #check system co hoat dong khong
def health():
    return {
        "status": "he thong chay on",
        "device": str(device),
        "model_loaded": _model is not None,
        "centroids_loaded": len(_centroids) if _centroids else 0,
        "thresholds": {
            "bin": bin_threshold,
            "sim": sim_threshold,
            "fq": fq_threshold,
            "min_type_freq": min_type_freq,
        },
    }

@app.post("/predict/mat", response_model= resultPredict)
async def predict_from_mat (
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
):
    if not file.filename.lower().endswith(".mat"):
        raise HTTPException(406, "System just accept file .mat")
    mat_bytes = await file.read()
    if not mat_bytes:
        raise HTTPException(400, "System can't found in4")
    job = Prediction(filename = file.filename, status = job_status.processing)
    db.add(job)
    await db.flush()
    t0 = time.monotonic()
    
    try:
        images = preprocessing(mat_bytes)
        if not images:
            raise ValueError(f"Signal too short, need >= {segment_length} samples")
        
        seg_results = thuc_thi(images)
        db.add_all([
            stft_image(
                predict_id = job.id,
                segment_index = idx,
                pre_bin = r["binary_score"],
                pre_phantom = r["type_scores"][0],
                pre_mavic_zoom = r["type_scores"][1],
                pre_mavic_enterprise = r["type_scores"][2],
                drone_bin = r["is_drone"],
            )
            for idx, r in enumerate(seg_results)
        ])

        agg = aggregate(seg_results)
        elapsed_ms = int((time.monotonic()- t0)*1000)

        job.status = job_status.done 
        job.drone_bin = agg["drone_detected"] #bool #float
        job.confidence = agg["avg_binary_score"] #float
        job.drone_type = agg["drone_types"] #JSON array
        job.drone_type_score = agg["type_detail"] #JSON object
        job.total_images = agg["total_segments"] #int
        job.processing_time = elapsed_ms #int

        return resultPredict (
            id = job.id,
            drone_detected= agg ["drone_detected"],
            drone_types= agg["drone_types"],
            has_unknown= agg["has_unknown"],
            avg_binary_score= agg["avg_binary_score"],
            drone_frame_ratio= agg["drone_frame_ratio"],
            drone_segments= agg["drone_segments"],
            total_segments= agg["total_segments"],
            type_detail= agg["type_detail"],
            process_time_ms= elapsed_ms,
        )
    except Exception as e:
        job.status = job_status.error
        raise HTTPException(500, f"Lỗi xử lý phía nội bộ của server {e}")

@app.get("/jobs", response_model = List[listHistory])
async def list_jobs (
    limit: int = 20,
    offset: int = 0,
    db: AsyncSession = Depends(get_db),
):
    rows = (await db.execute(
        select(Prediction).order_by(Prediction.created_at.desc()).limit(limit).offset(offset)
    )).scalars().all()

    return [
        listHistory(
            id = r.id,
            filename= r.filename,
            status= r.status,
            drone_bin= r.drone_bin,
            confidence= r.confidence,
            drone_type= r.drone_type,
            total_images= r.total_images,
            created_at= r.created_at.isoformat(),
        )
        for r in rows
    ]

@app.get("/jobs/{job_id}", summary= "Thong tin chi tiet cua 1 lan predict")
async def get_job(job_id: int, db: AsyncSession = Depends(get_db)):
    job = await db.get(Prediction, job_id)
    if not job:
        raise HTTPException(404, "Ban ghi khong ton tai")
    return {
        "id": job.id,
        "filename": job.filename,
        "status": job.status.value,
        "drone_bin": job.drone_bin,
        "confidence":       job.confidence,
        "drone_type":       job.drone_type,
        "drone_type_score": job.drone_type_score,
        "total_images":     job.total_images,
        "processing_time":  job.processing_time,
        "created_at":       job.created_at.isoformat(),
        "updated_at":       job.updated_at.isoformat() if job.updated_at else None,
    }

@app.get("/jobs/{job_id}/segments", summary= "Raw scores tung doan")
async def get_segments(
    job_id: int,
    limit: int = 200,
    offset: int = 0,
    db: AsyncSession = Depends(get_db),
):
    job = await db.get(Prediction, job_id)
    if not job:
        raise HTTPException(404, "Ban ghi khong ton tai")
    rows = (await db.execute(
        select(stft_image).where(stft_image.predict_id == job_id).order_by(stft_image.segment_index).limit(limit).offset(offset)
    )).scalars().all()
    return {
        "job_id":         job_id,
        "total_segments": job.total_images,
        "segments": [
            {
                "segment_index" : r.segment_index,
                "pre_bin" : r.pre_bin,
                "pre_phantom" : r.pre_phantom,
                "pre_mavic_zoom" : r.pre_mavic_zoom,
                "pre_mavic_enterprise" : r.pre_mavic_enterprise,
                "drone_bin" : r.drone_bin,
            }
            for r in rows
        ],
    }

#Admin tai weight
@app.post("/admin/upload-model", summary= "upload model.pth moi")
async def upload_model(file: UploadFile= File(...)):
    global _model
    os.makedirs(upload_dir, exist_ok= True)
    data = await file.read()
    with open(model_path, "wb")as f:
        f.write(data)
    _model = None
    return {"message": f"Saved -> {model_path}", "size_mb": round(len(data) / 1024**2, 2)}

@app.post("/admin/upload-centroids", summary="Upload centroids.pth mới")
async def upload_centroids(file: UploadFile = File(...)):
    global _centroids
    os.makedirs(upload_dir, exist_ok=True)
    data = await file.read()
    with open(centroids_path, "wb") as f:
        f.write(data)
    _centroids = None   # lazy reload
    return {"message": f"Saved → {centroids_path}", "size_mb": round(len(data) / 1024**2, 2)}