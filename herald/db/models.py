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
    engine_kwargs["pool_size"] = 10
    engine_kwargs["max_overflow"] = 20

engine = create_engine(DATABASE_URL, **engine_kwargs)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def init_db():
    Base.metadata.create_all(bind=engine)

if __name__ == "__main__":
    init_db()
    print("Database initialized.")
