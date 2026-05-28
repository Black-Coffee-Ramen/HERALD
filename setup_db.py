import os
import sqlite3
from herald.db.models import Base, engine

def migrate_sqlite(db_url):
    db_path = db_url.replace("sqlite:///", "")
    if not os.path.exists(db_path):
        return
        
    print(f"Migrating {db_path}...")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Check if domain_scans exists
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='domain_scans'")
    if not cursor.fetchone():
        return
        
    cursor.execute("PRAGMA table_info(domain_scans)")
    columns = [row[1] for row in cursor.fetchall()]
    
    if "lifecycle_state" not in columns:
        print("Adding lifecycle_state column...")
        cursor.execute("ALTER TABLE domain_scans ADD COLUMN lifecycle_state VARCHAR DEFAULT 'QUEUED'")
    
    if "screenshot_path" not in columns:
        print("Adding screenshot_path column...")
        cursor.execute("ALTER TABLE domain_scans ADD COLUMN screenshot_path VARCHAR")
        
    if "ocr_text" not in columns:
        print("Adding ocr_text column...")
        cursor.execute("ALTER TABLE domain_scans ADD COLUMN ocr_text VARCHAR")
        
    if "dns_records" not in columns:
        print("Adding dns_records column...")
        cursor.execute("ALTER TABLE domain_scans ADD COLUMN dns_records VARCHAR")
        
    conn.commit()
    conn.close()

def main():
    db_url = os.getenv("DATABASE_URL", "sqlite:///domain_history.db")
    
    # Simple migration for SQLite if it's the backend
    if db_url.startswith("sqlite"):
        try:
            migrate_sqlite(db_url)
        except Exception as e:
            print(f"Warning during migration: {e}")
    
    print("Ensuring all tables and schema exist...")
    Base.metadata.create_all(bind=engine)
    print("Database initialization complete.")

if __name__ == "__main__":
    main()
