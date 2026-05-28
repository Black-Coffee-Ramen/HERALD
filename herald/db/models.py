from sqlalchemy import create_engine, Column, Integer, String, DateTime, Float, Boolean
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime
import os

Base = declarative_base()

class DomainScan(Base):
    __tablename__ = 'domain_scans'
    
    id = Column(Integer, primary_key=True)
    domain = Column(String, index=True)
    target_cse = Column(String)
    source = Column(String)
    scan_date = Column(DateTime, default=datetime.utcnow)
    label = Column(String)  # 'Suspected', 'Phishing', 'Legitimate'
    confidence = Column(Float)
    is_live = Column(Boolean, default=False)
    analyst_verdict = Column(String, nullable=True)
    
    # Real Investigation Fields
    lifecycle_state = Column(String, default="QUEUED") # QUEUED, PROCESSING, SCREENSHOT_COMPLETE, OCR_COMPLETE, VERDICT_READY, FAILED, DEGRADED, DLQ
    screenshot_path = Column(String, nullable=True)
    ocr_text = Column(String, nullable=True)
    dns_records = Column(String, nullable=True) # JSON stored as string for simplicity

class User(Base):
    __tablename__ = 'users'
    
    id = Column(Integer, primary_key=True)
    username = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    role = Column(String, default="analyst") # 'admin', 'analyst'
    is_active = Column(Boolean, default=True)

class AuditLog(Base):
    __tablename__ = 'audit_logs'
    
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    user_id = Column(String, index=True) # can be 'anonymous' or 'system'
    action = Column(String)
    domain = Column(String, nullable=True)
    result = Column(String)
    ip_address = Column(String, nullable=True)

class Whitelist(Base):
    __tablename__ = 'whitelist'
    
    id = Column(Integer, primary_key=True)
    domain = Column(String, unique=True, index=True)
    added_by = Column(String, default="system")
    added_on = Column(DateTime, default=datetime.utcnow)
    reason = Column(String, nullable=True)

# By default, use sqlite for local testing if DATABASE_URL is not provided
# In production (Docker), this will be overwritten to use postgresql://
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///domain_history.db")

engine_kwargs = {}
if DATABASE_URL.startswith("postgresql"):
    engine_kwargs["pool_size"] = int(os.getenv("DB_POOL_SIZE", "10"))
    engine_kwargs["max_overflow"] = int(os.getenv("DB_MAX_OVERFLOW", "20"))
    engine_kwargs["pool_timeout"] = int(os.getenv("DB_POOL_TIMEOUT", "10"))
    engine_kwargs["pool_recycle"] = int(os.getenv("DB_POOL_RECYCLE_SECONDS", "1800"))
    engine_kwargs["pool_pre_ping"] = True
elif DATABASE_URL.startswith("sqlite"):
    engine_kwargs["connect_args"] = {"timeout": int(os.getenv("SQLITE_BUSY_TIMEOUT_SECONDS", "30"))}

engine = create_engine(DATABASE_URL, **engine_kwargs)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def init_db():
    Base.metadata.create_all(bind=engine)

if __name__ == "__main__":
    init_db()
    print("Database initialized.")
