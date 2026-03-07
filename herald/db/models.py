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
    scan_date = Column(DateTime, default=datetime.utcnow)
    label = Column(String)  # 'Suspected', 'Phishing', 'Legitimate'
    confidence = Column(Float)
    is_live = Column(Boolean, default=False)

# By default, use sqlite for local testing if DATABASE_URL is not provided
# In production (Docker), this will be overwritten to use postgresql://
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///domain_history.db")

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def init_db():
    Base.metadata.create_all(bind=engine)

if __name__ == "__main__":
    init_db()
    print("Database initialized.")
