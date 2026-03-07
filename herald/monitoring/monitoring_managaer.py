# src/monitoring/monitor_manager.py
import schedule
import time
from datetime import datetime, timedelta
import pandas as pd
import sqlite3

class MonitoringManager:
    def __init__(self, db_path='monitoring.db'):
        self.db_path = db_path
        self.init_database()
        self.monitoring_period = 90  # days
    
    def init_database(self):
        """Initialize monitoring database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS monitored_domains (
                domain TEXT PRIMARY KEY,
                initial_label TEXT,
                lexical_score REAL,
                whois_age_days INTEGER,
                registrar TEXT,
                content_state TEXT,
                visual_distance REAL,
                evidence_path TEXT,
                decision_timestamp TEXT,
                next_check_date TEXT,
                monitoring_end_date TEXT,
                current_label TEXT,
                check_count INTEGER DEFAULT 0
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS monitoring_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                domain TEXT,
                check_timestamp TEXT,
                content_state TEXT,
                visual_distance REAL,
                has_credentials BOOLEAN,
                screenshot_path TEXT,
                html_path TEXT,
                label_change TEXT,
                FOREIGN KEY (domain) REFERENCES monitored_domains (domain)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def add_to_monitoring(self, domain_data):
        """Add a domain to continuous monitoring"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        monitoring_end = (datetime.now() + 
                         timedelta(days=self.monitoring_period)).isoformat()
        next_check = (datetime.now() + timedelta(days=1)).isoformat()
        
        cursor.execute('''
            INSERT OR REPLACE INTO monitored_domains 
            (domain, initial_label, lexical_score, whois_age_days, registrar, 
             content_state, visual_distance, evidence_path, decision_timestamp,
             next_check_date, monitoring_end_date, current_label)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            domain_data['domain'],
            domain_data['initial_label'],
            domain_data['lexical_score'],
            domain_data['whois_age_days'],
            domain_data['registrar'],
            domain_data['content_state'],
            domain_data['visual_distance'],
            domain_data['evidence_path'],
            domain_data['decision_timestamp'],
            next_check,
            monitoring_end,
            domain_data['initial_label']  # current_label starts as initial_label
        ))
        
        conn.commit()
        conn.close()
        
        print(f"ðŸ” Added {domain_data['domain']} to monitoring until {monitoring_end}")
    
    def get_domains_due_for_check(self):
        """Get domains due for their daily check"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        current_time = datetime.now().isoformat()
        cursor.execute('''
            SELECT * FROM monitored_domains 
            WHERE next_check_date <= ? AND monitoring_end_date > ?
        ''', (current_time, current_time))
        
        domains = cursor.fetchall()
        conn.close()
        
        return domains
    
    def perform_daily_checks(self):
        """Perform daily checks on all due domains"""
        domains_due = self.get_domains_due_for_check()
        print(f"ðŸ“‹ Performing daily checks on {len(domains_due)} domains...")
        
        for domain_row in domains_due:
            domain = domain_row[0]  # domain is first column
            self.check_domain(domain)
    
    def check_domain(self, domain):
        """Check a single domain for changes"""
        from herald.core.domain_analyzer import DomainAnalyzer
        
        analyzer = DomainAnalyzer()
        
        # Re-analyze the domain
        content_analysis = analyzer.capture_and_analyze_content(domain)
        whois_analysis = analyzer.analyze_whois(domain)
        
        # Check for escalation conditions
        previous_state = self.get_previous_state(domain)
        current_state = content_analysis['content_state']
        
        # Escalation rule: parked/unrelated -> lookalike with credentials
        if (previous_state in ['parked', 'unrelated'] and 
            current_state == 'lookalike' and 
            content_analysis['has_credentials']):
            
            self.escalate_to_phishing(domain, content_analysis)
        
        # Update monitoring record
        self.update_domain_check(domain, content_analysis)
    
    def escalate_to_phishing(self, domain, content_analysis):
        """Escalate domain from Suspected to Phishing"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Update current label
        cursor.execute('''
            UPDATE monitored_domains 
            SET current_label = 'Phishing'
            WHERE domain = ?
        ''', (domain,))
        
        # Log the escalation
        cursor.execute('''
            INSERT INTO monitoring_log 
            (domain, check_timestamp, content_state, visual_distance, 
             has_credentials, label_change)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            domain,
            datetime.now().isoformat(),
            content_analysis['content_state'],
            content_analysis['visual_distance'],
            content_analysis['has_credentials'],
            'Suspected->Phishing'
        ))
        
        conn.commit()
        conn.close()
        
        # Trigger alert
        self.send_phishing_alert(domain, content_analysis)
        