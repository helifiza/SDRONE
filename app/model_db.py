import enum
from datetime import datetime
from sqlalchemy import text
from sqlalchemy import(
    BigInteger, Boolean, DateTime, Enum, Float,
    ForeignKey, Integer, String, JSON,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.csdl import Base

class job_status(str, enum.Enum):
    pending = "pending"
    processing = "processing"
    done = "done"
    error = "error"

class Prediction(Base):
    __tablename__ = "prediction"
    id : Mapped[int] = mapped_column(BigInteger, primary_key= True, autoincrement= True)
    filename: Mapped[str] = mapped_column(String(255), nullable= False)
    status: Mapped[job_status] = mapped_column(Enum(job_status), default= job_status.pending, server_default= "pending",)
    drone_bin : Mapped[bool | None] = mapped_column(Boolean, nullable= True)
    confidence: Mapped[float | None] = mapped_column(Float, nullable= True)
    drone_type: Mapped[list | None] = mapped_column(JSON, nullable= True)
    drone_type_score: Mapped[dict | None] = mapped_column(JSON, nullable= True)
    total_images: Mapped[int | None] = mapped_column(Integer, nullable= True)
    processing_time: Mapped[int | None] = mapped_column(Integer, nullable= True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default= text("CURRENT_TIMESTAMP"))
    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=text("CURRENT_TIMESTAMP"),server_onupdate=text("CURRENT_TIMESTAMP"),)
    images: Mapped[list["stft_image"]] = relationship(back_populates= "prediction", cascade= "all, delete-orphan")

class stft_image(Base):
    __tablename__ = "stft_image"
    id: Mapped[int] = mapped_column(BigInteger, primary_key= True, autoincrement= True) 
    predict_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("prediction.id", ondelete= "CASCADE"), nullable= False, index= True)
    segment_index: Mapped[int] = mapped_column(Integer, nullable=False)
    pre_bin: Mapped[float] = mapped_column(Float, nullable=True)
    pre_phantom: Mapped[float] = mapped_column(Float, nullable=True)
    pre_mavic_zoom: Mapped[float] = mapped_column(Float, nullable= True)
    pre_mavic_enterprise: Mapped[float] = mapped_column(Float, nullable= True)
    drone_bin: Mapped[bool] = mapped_column(Boolean, nullable=True)
    __table_args__ = (
        UniqueConstraint("predict_id", "segment_index", name= "uq_predict_segment"),
    )

    prediction: Mapped["Prediction"] = relationship(back_populates="images")