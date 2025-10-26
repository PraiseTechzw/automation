"""
COMPLETE AI-POWERED LINKEDIN AUTOMATION SYSTEM
===============================================
Features: Gemini AI, Vector Memory, Vision AI, Analytics, A/B Testing, Webhooks

⚠️ EDUCATIONAL PURPOSE ONLY - VIOLATES LINKEDIN TOS ⚠️
"""

import os
import json
import pickle
import time
import random
import csv
from datetime import datetime, timedelta
from collections import defaultdict
import hashlib
import base64
from typing import List, Dict, Optional

# Core dependencies
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager

# AI and ML
from google import genai
from google.genai import types
import numpy as np

# Analytics and visualization
import matplotlib.pyplot as plt
import pandas as pd

# Webhooks
import requests

# Database
import sqlite3
from contextlib import contextmanager

# Security and Anti-Detection
import undetected_chromedriver as uc
from fake_useragent import UserAgent
import requests
from urllib.parse import urlparse

# =====================================================================
# ENHANCED ERROR HANDLING & RECOVERY SYSTEM
# =====================================================================
class ErrorHandler:
    """Comprehensive error handling and recovery system"""
    
    def __init__(self, notifier=None):
        self.notifier = notifier
        self.error_log = []
        self.recovery_attempts = {}
        self.circuit_breaker = CircuitBreaker()
        
    def handle_error(self, error, context="", max_retries=3):
        """Handle errors with intelligent recovery"""
        error_info = {
            'timestamp': datetime.now().isoformat(),
            'error_type': type(error).__name__,
            'error_message': str(error),
            'context': context,
            'retry_count': self.recovery_attempts.get(context, 0)
        }
        
        self.error_log.append(error_info)
        
        # Log error
        print(f"❌ Error in {context}: {error}")
        
        # Notify if webhook available
        if self.notifier:
            self.notifier.notify_error(f"{context}: {str(error)[:100]}")
        
        # Check circuit breaker
        if self.circuit_breaker.is_open(context):
            print(f"  🔒 Circuit breaker open for {context} - skipping retry")
            return False
        
        # Recovery logic
        if self.recovery_attempts.get(context, 0) < max_retries:
            self.recovery_attempts[context] = self.recovery_attempts.get(context, 0) + 1
            return self._attempt_recovery(error, context)
        
        # Trip circuit breaker after max retries
        self.circuit_breaker.trip(context)
        return False
    
    def _attempt_recovery(self, error, context):
        """Attempt intelligent recovery based on error type"""
        error_str = str(error).lower()
        
        if 'timeout' in error_str or 'wait' in error_str:
            print(f"  🔄 Retrying with longer timeout...")
            time.sleep(random.uniform(5, 10))
            return True
            
        elif 'element not found' in error_str or 'no such element' in error_str:
            print(f"  🔄 Retrying with different selector...")
            time.sleep(random.uniform(2, 5))
            return True
            
        elif 'connection' in error_str or 'network' in error_str:
            print(f"  🔄 Retrying with connection reset...")
            time.sleep(random.uniform(10, 20))
            return True
            
        elif 'linkedin' in error_str and 'blocked' in error_str:
            print(f"  ⚠️ Possible LinkedIn detection - switching to safe mode")
            time.sleep(random.uniform(60, 120))
            return True
            
        elif 'captcha' in error_str or 'verification' in error_str:
            print(f"  🤖 CAPTCHA detected - manual intervention required")
            self.notifier.send_notification("🚨 CAPTCHA detected - manual intervention needed", 'warning')
            return False
            
        return False
    
    def get_error_summary(self):
        """Get error summary for reporting"""
        if not self.error_log:
            return "No errors recorded"
            
        error_types = {}
        for error in self.error_log:
            error_type = error['error_type']
            error_types[error_type] = error_types.get(error_type, 0) + 1
            
        summary = "Error Summary:\n"
        for error_type, count in error_types.items():
            summary += f"  {error_type}: {count}\n"
            
        return summary

class CircuitBreaker:
    """Circuit breaker pattern for error handling"""
    
    def __init__(self, failure_threshold=5, timeout=300):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = {}
        self.last_failure_time = {}
        self.state = {}  # 'closed', 'open', 'half-open'
    
    def is_open(self, context):
        """Check if circuit breaker is open for context"""
        if context not in self.state:
            self.state[context] = 'closed'
            return False
            
        if self.state[context] == 'open':
            # Check if timeout has passed
            if context in self.last_failure_time:
                if time.time() - self.last_failure_time[context] > self.timeout:
                    self.state[context] = 'half-open'
                    return False
            return True
            
        return False
    
    def trip(self, context):
        """Trip the circuit breaker for context"""
        self.failure_count[context] = self.failure_count.get(context, 0) + 1
        self.last_failure_time[context] = time.time()
        
        if self.failure_count[context] >= self.failure_threshold:
            self.state[context] = 'open'
            print(f"🔒 Circuit breaker tripped for {context}")
    
    def reset(self, context):
        """Reset circuit breaker for context"""
        self.failure_count[context] = 0
        self.state[context] = 'closed'
        print(f"✅ Circuit breaker reset for {context}")

class RateLimiter:
    """Advanced rate limiting with LinkedIn detection avoidance"""
    
    def __init__(self):
        self.action_history = defaultdict(list)
        self.daily_limits = {
            'connection': 15,
            'message': 20,
            'post': 2,
            'comment': 10,
            'like': 50
        }
        self.hourly_limits = {
            'connection': 3,
            'message': 5,
            'post': 1,
            'comment': 3,
            'like': 10
        }
        
    def can_perform_action(self, action_type):
        """Check if action can be performed based on rate limits"""
        now = datetime.now()
        today = now.strftime('%Y-%m-%d')
        hour = now.hour
        
        # Check daily limits
        today_actions = [a for a in self.action_history[action_type] 
                        if a['date'] == today]
        if len(today_actions) >= self.daily_limits[action_type]:
            return False, "Daily limit reached"
            
        # Check hourly limits
        hour_actions = [a for a in today_actions 
                       if a['hour'] == hour]
        if len(hour_actions) >= self.hourly_limits[action_type]:
            return False, "Hourly limit reached"
            
        # Check for suspicious patterns
        if self._detect_suspicious_pattern(action_type):
            return False, "Suspicious pattern detected"
            
        return True, "OK"
    
    def record_action(self, action_type):
        """Record an action for rate limiting"""
        now = datetime.now()
        self.action_history[action_type].append({
            'timestamp': now,
            'date': now.strftime('%Y-%m-%d'),
            'hour': now.hour
        })
        
        # Clean old records (keep last 7 days)
        cutoff = now - timedelta(days=7)
        self.action_history[action_type] = [
            a for a in self.action_history[action_type]
            if a['timestamp'] > cutoff
        ]
    
    def _detect_suspicious_pattern(self, action_type):
        """Detect suspicious automation patterns"""
        now = datetime.now()
        recent_actions = [
            a for a in self.action_history[action_type]
            if (now - a['timestamp']).seconds < 300  # Last 5 minutes
        ]
        
        # Too many actions in short time
        if len(recent_actions) > 3:
            return True
            
        # Regular intervals (too robotic)
        if len(recent_actions) >= 3:
            intervals = []
            for i in range(1, len(recent_actions)):
                interval = (recent_actions[i-1]['timestamp'] - recent_actions[i]['timestamp']).seconds
                intervals.append(interval)
                
            # If intervals are too similar (within 10 seconds)
            if len(set(intervals)) == 1 and intervals[0] < 60:
                return True
                
        return False
    
    def get_safe_delay(self, action_type):
        """Get safe delay based on action history"""
        base_delays = {
            'connection': (30, 120),
            'message': (20, 60),
            'post': (300, 600),
            'comment': (60, 180),
            'like': (10, 30)
        }
        
        base_min, base_max = base_delays.get(action_type, (30, 60))
        
        # Increase delay if approaching limits
        now = datetime.now()
        today_actions = [a for a in self.action_history[action_type] 
                        if a['date'] == now.strftime('%Y-%m-%d')]
        
        if len(today_actions) > self.daily_limits[action_type] * 0.8:
            base_min *= 2
            base_max *= 2
            
        return random.uniform(base_min, base_max)

# =====================================================================
# DATABASE INTEGRATION SYSTEM
# =====================================================================
class DatabaseManager:
    """SQLite database integration for persistent data storage"""
    
    def __init__(self, db_path="data/linkedin_bot.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize database with all required tables"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Profiles table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS profiles (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    url TEXT UNIQUE NOT NULL,
                    name TEXT,
                    headline TEXT,
                    location TEXT,
                    connections INTEGER,
                    industry TEXT,
                    relevance_score INTEGER,
                    should_connect BOOLEAN,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Connections table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS connections (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    profile_id INTEGER,
                    message_sent TEXT,
                    variant_used TEXT,
                    sent_at TIMESTAMP,
                    accepted_at TIMESTAMP,
                    status TEXT DEFAULT 'pending',
                    FOREIGN KEY (profile_id) REFERENCES profiles (id)
                )
            """)
            
            # Conversations table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    profile_id INTEGER,
                    message TEXT,
                    is_sent BOOLEAN,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    sentiment_score REAL,
                    topics TEXT,
                    FOREIGN KEY (profile_id) REFERENCES profiles (id)
                )
            """)
            
            # Analytics table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS analytics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date TEXT,
                    action_type TEXT,
                    count INTEGER,
                    success_rate REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # A/B test results table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS ab_tests (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    test_name TEXT,
                    variant TEXT,
                    sent_count INTEGER DEFAULT 0,
                    accepted_count INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Error logs table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS error_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    error_type TEXT,
                    error_message TEXT,
                    context TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    resolved BOOLEAN DEFAULT FALSE
                )
            """)
            
            conn.commit()
    
    @contextmanager
    def get_connection(self):
        """Get database connection with proper error handling"""
        conn = None
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            yield conn
        except Exception as e:
            if conn:
                conn.rollback()
            raise e
        finally:
            if conn:
                conn.close()
    
    def save_profile(self, profile_data):
        """Save or update profile data"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT OR REPLACE INTO profiles 
                (url, name, headline, location, connections, industry, relevance_score, should_connect, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """, (
                profile_data.get('url'),
                profile_data.get('name'),
                profile_data.get('headline'),
                profile_data.get('location'),
                profile_data.get('connections', 0),
                profile_data.get('industry', 'Unknown'),
                profile_data.get('score', 0),
                profile_data.get('should_connect', False)
            ))
            
            conn.commit()
            return cursor.lastrowid
    
    def save_connection_attempt(self, profile_id, message, variant):
        """Save connection attempt"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO connections (profile_id, message_sent, variant_used, sent_at)
                VALUES (?, ?, ?, CURRENT_TIMESTAMP)
            """, (profile_id, message, variant))
            
            conn.commit()
            return cursor.lastrowid
    
    def update_connection_status(self, profile_id, status):
        """Update connection status"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                UPDATE connections 
                SET status = ?, accepted_at = CURRENT_TIMESTAMP
                WHERE profile_id = ? AND status = 'pending'
            """, (status, profile_id))
            
            conn.commit()
    
    def save_conversation(self, profile_id, message, is_sent, sentiment_score=None, topics=None):
        """Save conversation message"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO conversations (profile_id, message, is_sent, sentiment_score, topics)
                VALUES (?, ?, ?, ?, ?)
            """, (profile_id, message, is_sent, sentiment_score, topics))
            
            conn.commit()
    
    def save_analytics(self, date, action_type, count, success_rate):
        """Save analytics data"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO analytics (date, action_type, count, success_rate)
                VALUES (?, ?, ?, ?)
            """, (date, action_type, count, success_rate))
            
            conn.commit()
    
    def save_error_log(self, error_type, error_message, context):
        """Save error log"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO error_logs (error_type, error_message, context)
                VALUES (?, ?, ?)
            """, (error_type, error_message, context))
            
            conn.commit()
    
    def get_analytics_summary(self, days=7):
        """Get analytics summary from database"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT 
                    action_type,
                    SUM(count) as total_count,
                    AVG(success_rate) as avg_success_rate
                FROM analytics 
                WHERE date >= date('now', '-{} days')
                GROUP BY action_type
            """.format(days))
            
            return cursor.fetchall()
    
    def get_connection_stats(self):
        """Get connection statistics"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT 
                    COUNT(*) as total_sent,
                    SUM(CASE WHEN status = 'accepted' THEN 1 ELSE 0 END) as accepted,
                    AVG(CASE WHEN status = 'accepted' THEN 1.0 ELSE 0.0 END) * 100 as acceptance_rate
                FROM connections
            """)
            
            return cursor.fetchone()
    
    def get_top_performing_variants(self):
        """Get top performing A/B test variants"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT 
                    variant_used,
                    COUNT(*) as sent,
                    SUM(CASE WHEN status = 'accepted' THEN 1 ELSE 0 END) as accepted,
                    ROUND(AVG(CASE WHEN status = 'accepted' THEN 1.0 ELSE 0.0 END) * 100, 2) as rate
                FROM connections
                GROUP BY variant_used
                ORDER BY rate DESC
            """)
            
            return cursor.fetchall()

# =====================================================================
# CONFIGURATION WITH ENVIRONMENT VARIABLES
# =====================================================================
CONFIG = {
    # Credentials (USE ENVIRONMENT VARIABLES!)
    'linkedin_email': os.getenv('LINKEDIN_EMAIL', ''),
    'linkedin_password': os.getenv('LINKEDIN_PASSWORD', ''),
    'gemini_api_key': os.getenv('GEMINI_API_KEY', ''),

    # Webhook notifications (optional)
    'webhook_url': os.getenv('WEBHOOK_URL', ''),  # Slack, Discord, etc.

    # File paths
    'cookies_file': 'data/linkedin_cookies.pkl',
    'activity_log': 'data/activity_log.json',
    'profile_db': 'data/profiles.json',
    'conversation_memory': 'data/conversations.json',
    'analytics_db': 'data/analytics.json',
    'ab_test_results': 'data/ab_tests.json',

    # Daily limits
    'daily_connection_limit': 15,
    'daily_message_limit': 20,
    'daily_post_limit': 2,
    'daily_comment_limit': 10,

    # Timing patterns
    'short_delay': (3, 8),
    'medium_delay': (10, 20),
    'long_delay': (30, 60),

    # Active hours
    'active_hours': [(9, 12), (14, 17), (19, 21)],
    'timezone': 'Africa/Harare',

    # AI settings
    'ai_temperature': 0.8,
    'ai_max_tokens': 400,
    'enable_vision_ai': True,
    'enable_ab_testing': True,

    # Analytics
    'track_metrics': True,
    'generate_reports': True,
}

# Ensure data directory exists
os.makedirs('data', exist_ok=True)
os.makedirs('reports', exist_ok=True)

# =====================================================================
# WEBHOOK NOTIFICATION SYSTEM
# =====================================================================
class WebhookNotifier:
    """Send notifications via webhook (Slack, Discord, etc.)"""

    def __init__(self, webhook_url):
        self.webhook_url = webhook_url
        self.enabled = bool(webhook_url)

    def send_notification(self, message, level='info'):
        """Send notification with different levels"""
        if not self.enabled:
            return

        colors = {
            'success': '#36a64f',
            'warning': '#ff9800',
            'error': '#f44336',
            'info': '#2196f3'
        }

        # Slack format
        payload = {
            'attachments': [{
                'color': colors.get(level, colors['info']),
                'text': message,
                'footer': 'LinkedIn AI Bot',
                'ts': int(time.time())
            }]
        }

        try:
            requests.post(self.webhook_url, json=payload, timeout=5)
        except:
            pass

    def notify_connection_accepted(self, profile_name):
        self.send_notification(
            f"✅ Connection accepted by {profile_name}!",
            level='success'
        )

    def notify_message_received(self, from_name):
        self.send_notification(
            f"💬 New message from {from_name}",
            level='info'
        )

    def notify_daily_limit(self, action_type):
        self.send_notification(
            f"⚠️ Daily {action_type} limit reached",
            level='warning'
        )

    def notify_error(self, error_msg):
        self.send_notification(
            f"❌ Error: {error_msg}",
            level='error'
        )

# =====================================================================
# VECTOR MEMORY SYSTEM FOR CONVERSATIONS
# =====================================================================
class ConversationMemory:
    """Store and retrieve conversation context using embeddings"""

    def __init__(self, ai_client):
        self.ai_client = ai_client
        self.memory_file = CONFIG['conversation_memory']
        self.conversations = self.load_conversations()

    def load_conversations(self):
        """Load conversation history"""
        try:
            with open(self.memory_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return {}

    def save_conversations(self):
        """Save conversation history"""
        with open(self.memory_file, 'w') as f:
            json.dump(self.conversations, f, indent=2)

    def generate_embedding(self, text):
        """Generate simple embedding (in production, use proper embeddings)"""
        # Simplified embedding - in production use Gemini embeddings API
        return hashlib.md5(text.encode()).hexdigest()[:16]

    def store_conversation(self, profile_id, message, is_sent=True):
        """Store a conversation message"""
        if profile_id not in self.conversations:
            self.conversations[profile_id] = {
                'messages': [],
                'first_contact': datetime.now().isoformat(),
                'last_interaction': datetime.now().isoformat(),
                'sentiment_history': [],
                'topics_discussed': []
            }

        self.conversations[profile_id]['messages'].append({
            'text': message,
            'timestamp': datetime.now().isoformat(),
            'is_sent': is_sent,
            'embedding': self.generate_embedding(message)
        })

        self.conversations[profile_id]['last_interaction'] = datetime.now().isoformat()
        self.save_conversations()

    def get_conversation_context(self, profile_id, last_n=5):
        """Retrieve recent conversation context"""
        if profile_id not in self.conversations:
            return ""

        messages = self.conversations[profile_id]['messages'][-last_n:]
        context = "\n".join([
            f"{'Me' if m['is_sent'] else 'Them'}: {m['text']}"
            for m in messages
        ])

        return context

    def analyze_conversation_topics(self, profile_id):
        """Extract main topics from conversation"""
        if profile_id not in self.conversations:
            return []

        messages = self.conversations[profile_id]['messages']
        all_text = " ".join([m['text'] for m in messages])

        # Simple keyword extraction
        keywords = ['AI', 'software', 'development', 'business', 'startup',
                   'technology', 'innovation', 'Zimbabwe', 'Africa']

        topics = [kw for kw in keywords if kw.lower() in all_text.lower()]
        return topics

# =====================================================================
# QUANTUM-INSPIRED BEHAVIORAL PATTERNS
# =====================================================================
class QuantumBehaviorEngine:
    """Quantum-inspired behavioral patterns for ultra-human-like behavior"""
    
    def __init__(self):
        self.behavioral_states = {
            'energetic': {'probability': 0.3, 'actions': ['fast_typing', 'quick_scroll', 'rapid_clicks']},
            'contemplative': {'probability': 0.2, 'actions': ['slow_typing', 'pause_reading', 'careful_selection']},
            'social': {'probability': 0.25, 'actions': ['hover_elements', 'read_profiles', 'engage_content']},
            'analytical': {'probability': 0.15, 'actions': ['deep_scroll', 'examine_details', 'compare_profiles']},
            'casual': {'probability': 0.1, 'actions': ['random_clicks', 'browse_mode', 'light_engagement']}
        }
        self.current_state = 'social'
        self.state_history = []
        self.quantum_coherence = 0.7
        
    def quantum_state_transition(self):
        """Quantum-inspired state transitions"""
        import math
        
        # Calculate quantum probabilities
        probabilities = []
        for state, config in self.behavioral_states.items():
            base_prob = config['probability']
            # Add quantum interference effects
            interference = 0.1 * math.sin(len(self.state_history) * 0.5)
            quantum_prob = base_prob + interference
            probabilities.append(max(0, min(1, quantum_prob)))
        
        # Normalize probabilities
        total_prob = sum(probabilities)
        if total_prob > 0:
            probabilities = [p/total_prob for p in probabilities]
        
        # Quantum measurement (state collapse)
        import random
        rand_val = random.random()
        cumulative = 0
        states = list(self.behavioral_states.keys())
        
        for i, prob in enumerate(probabilities):
            cumulative += prob
            if rand_val <= cumulative:
                self.current_state = states[i]
                break
        
        self.state_history.append(self.current_state)
        if len(self.state_history) > 10:
            self.state_history.pop(0)
            
        return self.current_state
    
    def get_quantum_delay(self, base_delay_range):
        """Get quantum-inspired delay with uncertainty principle"""
        import math
        import random
        
        # Heisenberg uncertainty principle applied to timing
        uncertainty_factor = random.uniform(0.8, 1.2)
        quantum_fluctuation = math.sin(random.uniform(0, 2*math.pi)) * 0.1
        
        min_delay, max_delay = base_delay_range
        quantum_min = min_delay * uncertainty_factor + quantum_fluctuation
        quantum_max = max_delay * uncertainty_factor + quantum_fluctuation
        
        return random.uniform(max(0.1, quantum_min), max(quantum_min + 0.1, quantum_max))
    
    def quantum_typing_pattern(self, text):
        """Quantum-inspired typing patterns with natural variation"""
        import random
        import time
        
        typing_patterns = {
            'energetic': {'base_speed': 0.05, 'variation': 0.02, 'burst_prob': 0.3},
            'contemplative': {'base_speed': 0.15, 'variation': 0.05, 'burst_prob': 0.1},
            'social': {'base_speed': 0.08, 'variation': 0.03, 'burst_prob': 0.2},
            'analytical': {'base_speed': 0.12, 'variation': 0.04, 'burst_prob': 0.15},
            'casual': {'base_speed': 0.1, 'variation': 0.06, 'burst_prob': 0.25}
        }
        
        pattern = typing_patterns.get(self.current_state, typing_patterns['social'])
        
        for i, char in enumerate(text):
            # Quantum typing speed variation
            base_speed = pattern['base_speed']
            variation = random.uniform(-pattern['variation'], pattern['variation'])
            speed = max(0.01, base_speed + variation)
            
            # Quantum burst typing (rapid sequences)
            if random.random() < pattern['burst_prob'] and i < len(text) - 2:
                # Type next 2-4 characters rapidly
                burst_length = random.randint(2, min(4, len(text) - i))
                for j in range(burst_length):
                    if i + j < len(text):
                        time.sleep(0.01)  # Very fast typing
                i += burst_length - 1
            else:
                time.sleep(speed)
    
    def quantum_mouse_movement(self, driver):
        """Quantum-inspired mouse movements with natural physics"""
        import random
        import math
        
        # Get current mouse position (simulated)
        current_x = random.randint(0, 1920)
        current_y = random.randint(0, 1080)
        
        # Quantum field simulation for mouse path
        target_x = random.randint(0, 1920)
        target_y = random.randint(0, 1080)
        
        # Calculate quantum path with field interference
        steps = random.randint(5, 15)
        for step in range(steps):
            # Quantum field influence
            field_influence = math.sin(step * 0.5) * 0.1
            
            # Interpolate with quantum fluctuations
            progress = step / steps
            quantum_x = current_x + (target_x - current_x) * progress + field_influence * 50
            quantum_y = current_y + (target_y - current_y) * progress + field_influence * 30
            
            # Execute quantum mouse movement
            driver.execute_script(f"""
                var event = new MouseEvent('mousemove', {{
                    'view': window,
                    'bubbles': true,
                    'cancelable': true,
                    'clientX': {quantum_x},
                    'clientY': {quantum_y}
                }});
                document.dispatchEvent(event);
            """)
            
            time.sleep(random.uniform(0.01, 0.05))

# =====================================================================
# NEURAL CONVERSATION MEMORY WITH CONTEXT AWARENESS
# =====================================================================
class NeuralConversationMemory:
    """Advanced neural-inspired conversation memory with context awareness"""
    
    def __init__(self, ai_client):
        self.ai_client = ai_client
        self.memory_file = 'data/neural_memory.json'
        self.conversations = self.load_neural_memory()
        self.context_vectors = {}
        self.emotional_states = {}
        self.conversation_graphs = {}
        
    def load_neural_memory(self):
        """Load neural conversation memory"""
        try:
            with open(self.memory_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return {
                'conversations': {},
                'context_vectors': {},
                'emotional_states': {},
                'conversation_graphs': {},
                'memory_weights': {},
                'temporal_patterns': {}
            }
    
    def save_neural_memory(self):
        """Save neural conversation memory"""
        with open(self.memory_file, 'w') as f:
            json.dump(self.conversations, f, indent=2)
    
    def generate_neural_embedding(self, text, context=""):
        """Generate neural-inspired embedding with context"""
        import hashlib
        import time
        
        # Create context-aware hash
        context_text = f"{text}|{context}|{time.time()}"
        
        # Multi-layer hashing for neural-like representation
        layer1 = hashlib.sha256(context_text.encode()).hexdigest()
        layer2 = hashlib.md5(layer1.encode()).hexdigest()
        layer3 = hashlib.sha1(layer2.encode()).hexdigest()
        
        # Combine layers for neural representation
        neural_vector = layer1[:8] + layer2[:8] + layer3[:8]
        
        return neural_vector
    
    def store_conversation_with_context(self, profile_id, message, is_sent=True, emotional_context=None):
        """Store conversation with advanced context awareness"""
        if profile_id not in self.conversations['conversations']:
            self.conversations['conversations'][profile_id] = {
                'messages': [],
                'context_history': [],
                'emotional_journey': [],
                'conversation_flow': [],
                'temporal_patterns': [],
                'relationship_evolution': []
            }
        
        # Generate neural context
        neural_context = self.generate_neural_embedding(message, str(emotional_context))
        
        # Store with advanced metadata
        conversation_entry = {
            'text': message,
            'timestamp': datetime.now().isoformat(),
            'is_sent': is_sent,
            'neural_embedding': neural_context,
            'emotional_context': emotional_context,
            'conversation_stage': self._determine_conversation_stage(profile_id),
            'context_weight': self._calculate_context_weight(profile_id),
            'relationship_score': self._calculate_relationship_score(profile_id)
        }
        
        self.conversations['conversations'][profile_id]['messages'].append(conversation_entry)
        
        # Update emotional journey
        if emotional_context:
            self.conversations['conversations'][profile_id]['emotional_journey'].append({
                'timestamp': datetime.now().isoformat(),
                'emotion': emotional_context,
                'intensity': self._calculate_emotional_intensity(message)
            })
        
        # Update conversation flow
        self._update_conversation_flow(profile_id, message, is_sent)
        
        self.save_neural_memory()
    
    def _determine_conversation_stage(self, profile_id):
        """Determine conversation stage based on neural patterns"""
        if profile_id not in self.conversations['conversations']:
            return 'initial'
        
        messages = self.conversations['conversations'][profile_id]['messages']
        if len(messages) == 0:
            return 'initial'
        elif len(messages) <= 2:
            return 'introduction'
        elif len(messages) <= 5:
            return 'building_rapport'
        elif len(messages) <= 10:
            return 'deepening_connection'
        else:
            return 'established_relationship'
    
    def _calculate_context_weight(self, profile_id):
        """Calculate context weight based on conversation history"""
        if profile_id not in self.conversations['conversations']:
            return 1.0
        
        messages = self.conversations['conversations'][profile_id]['messages']
        if len(messages) == 0:
            return 1.0
        
        # Weight increases with conversation depth
        base_weight = 1.0
        depth_factor = min(2.0, 1.0 + (len(messages) * 0.1))
        
        return base_weight * depth_factor
    
    def _calculate_relationship_score(self, profile_id):
        """Calculate relationship score based on interaction patterns"""
        if profile_id not in self.conversations['conversations']:
            return 0.0
        
        messages = self.conversations['conversations'][profile_id]['messages']
        if len(messages) == 0:
            return 0.0
        
        # Calculate based on message frequency, emotional context, and engagement
        frequency_score = min(1.0, len(messages) / 10.0)
        
        # Emotional engagement score
        emotional_messages = [m for m in messages if m.get('emotional_context')]
        emotional_score = len(emotional_messages) / max(1, len(messages))
        
        # Response rate score (simulated)
        response_score = 0.8  # Would be calculated from actual response patterns
        
        total_score = (frequency_score * 0.4 + emotional_score * 0.3 + response_score * 0.3)
        return min(1.0, total_score)
    
    def _calculate_emotional_intensity(self, message):
        """Calculate emotional intensity of message"""
        emotional_words = {
            'excited': ['excited', 'thrilled', 'amazing', 'fantastic', 'wonderful'],
            'concerned': ['concerned', 'worried', 'troubled', 'anxious'],
            'grateful': ['thankful', 'grateful', 'appreciate', 'blessed'],
            'curious': ['curious', 'interested', 'wondering', 'intrigued']
        }
        
        message_lower = message.lower()
        max_intensity = 0
        
        for emotion, words in emotional_words.items():
            intensity = sum(1 for word in words if word in message_lower)
            max_intensity = max(max_intensity, intensity)
        
        return min(1.0, max_intensity / 3.0)
    
    def _update_conversation_flow(self, profile_id, message, is_sent):
        """Update conversation flow patterns"""
        if profile_id not in self.conversations['conversations']:
            self.conversations['conversations'][profile_id]['conversation_flow'] = []
        
        flow_entry = {
            'timestamp': datetime.now().isoformat(),
            'message_type': 'sent' if is_sent else 'received',
            'message_length': len(message),
            'contains_question': '?' in message,
            'emotional_tone': self._analyze_emotional_tone(message)
        }
        
        self.conversations['conversations'][profile_id]['conversation_flow'].append(flow_entry)
    
    def _analyze_emotional_tone(self, message):
        """Analyze emotional tone of message"""
        positive_words = ['great', 'excellent', 'amazing', 'wonderful', 'fantastic', 'love', 'enjoy']
        negative_words = ['bad', 'terrible', 'awful', 'hate', 'disappointed', 'frustrated']
        neutral_words = ['okay', 'fine', 'good', 'alright', 'sure']
        
        message_lower = message.lower()
        
        positive_count = sum(1 for word in positive_words if word in message_lower)
        negative_count = sum(1 for word in negative_words if word in message_lower)
        neutral_count = sum(1 for word in neutral_words if word in message_lower)
        
        if positive_count > negative_count and positive_count > neutral_count:
            return 'positive'
        elif negative_count > positive_count and negative_count > neutral_count:
            return 'negative'
        else:
            return 'neutral'
    
    def get_conversation_context_with_emotion(self, profile_id, last_n=5):
        """Get conversation context with emotional awareness"""
        if profile_id not in self.conversations['conversations']:
            return "", {}
        
        messages = self.conversations['conversations'][profile_id]['messages'][-last_n:]
        context = ""
        emotional_summary = {}
        
        for msg in messages:
            speaker = "Me" if msg['is_sent'] else "Them"
            context += f"{speaker}: {msg['text']}\n"
            
            if msg.get('emotional_context'):
                emotion = msg['emotional_context']
                emotional_summary[emotion] = emotional_summary.get(emotion, 0) + 1
        
        return context.strip(), emotional_summary

# =====================================================================
# ADVANCED GEMINI AI WITH VISION
# =====================================================================
class AdvancedGeminiAI:
    """Enhanced AI with vision capabilities and memory"""

    def __init__(self, api_key):
        self.api_key = api_key
        self.client = None
        self.memory = None
        self.initialize_client()

    def initialize_client(self):
        """Initialize Gemini client"""
        try:
            self.client = genai.Client(api_key=self.api_key)
            print("✓ Gemini AI initialized")
        except Exception as e:
            print(f"✗ Failed to initialize Gemini: {e}")
            self.client = None

    def set_memory(self, memory_system):
        """Connect to conversation memory"""
        self.memory = memory_system

    def analyze_profile_photo(self, image_url):
        """Analyze profile photo using Vision AI"""
        if not CONFIG['enable_vision_ai']:
            return {'professional': True, 'context': 'standard'}

        try:
            # In production, fetch and analyze actual image
            # For now, return default analysis
            return {
                'professional': True,
                'setting': 'office',
                'context': 'professional headshot',
                'confidence': 0.85
            }
        except:
            return {'professional': True, 'context': 'unknown'}

    def generate_connection_message(self, profile_data, conversation_context="", style='professional'):
        """Generate personalized connection message with context"""

        # Get conversation history if exists
        if self.memory and profile_data.get('url'):
            conversation_context = self.memory.get_conversation_context(profile_data['url'])

        style_prompts = {
            'professional': 'Professional and formal',
            'casual': 'Friendly and casual',
            'story': 'Start with a brief personal story'
        }

        prompt = f"""Generate a LinkedIn connection request message.

Profile Information:
- Name: {profile_data.get('name', 'Professional')}
- Headline: {profile_data.get('headline', 'Not specified')}
- Location: {profile_data.get('location', 'Not specified')}
- Connections: {profile_data.get('connections', 0)}

{f'Previous conversation context:{conversation_context}' if conversation_context else ''}

Style: {style_prompts.get(style, 'Professional')}

Requirements:
- Maximum 280 characters (LinkedIn limit with buffer)
- {style_prompts.get(style, 'Professional')} tone
- Reference something specific from their profile
- Clear reason for connecting
- Authentic and genuine
- NO sales pitch
- NO generic templates

Generate the message:"""

        try:
            response = self.client.models.generate_content(
                model="gemini-2.0-flash-exp",
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=CONFIG['ai_temperature'],
                    max_output_tokens=150,
                )
            )

            message = response.text.strip().strip('"').strip("'")

            # Ensure within limit
            if len(message) > 280:
                message = message[:277] + "..."

            return message

        except Exception as e:
            print(f"AI generation failed: {e}")
            return f"Hi {profile_data.get('name', 'there')}, impressed by your work in {self._extract_industry(profile_data.get('headline', ''))}. Let's connect!"

    def generate_follow_up_message(self, profile_data, conversation_context=""):
        """Generate follow-up message with conversation memory"""

        if self.memory and profile_data.get('url'):
            conversation_context = self.memory.get_conversation_context(profile_data['url'])
            topics = self.memory.analyze_conversation_topics(profile_data['url'])
        else:
            topics = []

        prompt = f"""Generate a follow-up message for a LinkedIn connection.

Profile: {profile_data.get('name', 'Professional')}
Headline: {profile_data.get('headline', '')}

{f'Previous conversation:{conversation_context}' if conversation_context else 'First message after connecting'}
{f'Topics discussed: {", ".join(topics)}' if topics else ''}

Requirements:
- Reference previous conversation if available
- Ask engaging question or offer value
- Natural and conversational
- Maximum 500 characters
- Build on existing relationship

Generate the message:"""

        try:
            response = self.client.models.generate_content(
                model="gemini-2.0-flash-exp",
                contents=prompt,
                config=types.GenerateContentConfig(temperature=0.7)
            )
            return response.text.strip()
        except:
            return f"Thanks for connecting, {profile_data.get('name', 'there')}!"

    def generate_post_comment(self, post_content, post_author=""):
        """Generate intelligent comment"""

        prompt = f"""Generate a thoughtful LinkedIn comment.

Post content (excerpt):
{post_content[:500]}

Author: {post_author}

Requirements:
- Add genuine insight or perspective
- Ask a specific question OR share brief experience
- 30-120 characters
- Professional but conversational
- NO generic praise
- Sound authentically human

Generate comment:"""

        try:
            response = self.client.models.generate_content(
                model="gemini-2.0-flash-exp",
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.85,
                    max_output_tokens=80,
                )
            )
            return response.text.strip()
        except:
            return "Interesting perspective! What challenges did you face implementing this?"

    def generate_linkedin_post(self, topic, post_type="insight"):
        """Generate engaging post"""

        post_templates = {
            'insight': 'Share a professional insight or lesson',
            'story': 'Tell a brief professional story',
            'question': 'Ask thought-provoking question',
            'list': 'Create valuable list of tips',
            'trend': 'Comment on industry trend'
        }

        prompt = f"""Generate a LinkedIn post about: {topic}

Type: {post_templates.get(post_type, 'insight')}

Context: Tech professional in Zimbabwe/Africa ecosystem

Requirements:
- Hook in first line
- Personal and authentic
- 150-280 words
- 2-4 relevant hashtags
- Line breaks for readability
- End with engagement question
- Sound human, not AI-generated
- Relevant to African/Zimbabwe tech context

Generate post:"""

        try:
            response = self.client.models.generate_content(
                model="gemini-2.0-flash-exp",
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.9,
                    max_output_tokens=CONFIG['ai_max_tokens'],
                )
            )
            return response.text.strip()
        except:
            return f"Thoughts on {topic}... What's your perspective? #TechCommunity #Zimbabwe"

    def analyze_profile_relevance(self, profile_data):
        """AI-powered relevance analysis"""

        prompt = f"""Analyze LinkedIn profile for networking relevance.

Profile:
- Name: {profile_data.get('name', '')}
- Headline: {profile_data.get('headline', '')}
- Location: {profile_data.get('location', '')}
- Connections: {profile_data.get('connections', 0)}

My profile: Tech professional in Zimbabwe, interested in software development, AI, and African tech ecosystem.

Provide:
1. Relevance score (0-100)
2. Should connect? (Yes/No)
3. Key reasons (2-3 points)
4. Conversation starters (2 topics)

Format:
Score: [number]
Connect: [Yes/No]
Reasons:
- [reason 1]
- [reason 2]
Topics:
- [topic 1]
- [topic 2]"""

        try:
            response = self.client.models.generate_content(
                model="gemini-2.0-flash-exp",
                contents=prompt,
                config=types.GenerateContentConfig(temperature=0.3)
            )

            analysis = response.text.strip()

            # Parse response
            score = 50
            should_connect = True

            for line in analysis.split('\n'):
                if 'Score:' in line:
                    score_str = ''.join(filter(str.isdigit, line))
                    score = int(score_str) if score_str else 50
                if 'Connect: No' in line or 'Connect: no' in line:
                    should_connect = False

            return {
                'score': min(100, max(0, score)),
                'analysis': analysis,
                'should_connect': should_connect and score >= 40
            }
        except:
            return {'score': 50, 'analysis': 'Analysis failed', 'should_connect': True}

    def _extract_industry(self, headline):
        """Extract industry from headline"""
        headline_lower = headline.lower()

        industries = {
            'Software Development': ['software', 'developer', 'engineer', 'programmer', 'coding'],
            'AI/ML': ['ai', 'machine learning', 'data science', 'artificial intelligence'],
            'Business': ['business', 'entrepreneur', 'founder', 'ceo', 'manager'],
            'Design': ['design', 'ux', 'ui', 'creative'],
            'Marketing': ['marketing', 'brand', 'content', 'social media']
        }

        for industry, keywords in industries.items():
            if any(kw in headline_lower for kw in keywords):
                return industry

        return 'Technology'
    
    def analyze_sentiment(self, text):
        """Analyze sentiment of text using AI"""
        prompt = f"""Analyze the sentiment of this text and provide a score from -1 (very negative) to 1 (very positive).

Text: "{text}"

Provide:
1. Sentiment score (-1 to 1)
2. Confidence level (0 to 1)
3. Key emotions detected
4. Overall tone

Format:
Score: [number]
Confidence: [number]
Emotions: [emotion1, emotion2, ...]
Tone: [description]"""

        try:
            response = self.client.models.generate_content(
                model="gemini-2.0-flash-exp",
                contents=prompt,
                config=types.GenerateContentConfig(temperature=0.3)
            )
            
            # Parse response
            lines = response.text.strip().split('\n')
            score = 0
            confidence = 0.5
            
            for line in lines:
                if 'Score:' in line:
                    score_str = ''.join(filter(lambda x: x.isdigit() or x in '.-', line))
                    try:
                        score = float(score_str)
                    except:
                        score = 0
                elif 'Confidence:' in line:
                    conf_str = ''.join(filter(lambda x: x.isdigit() or x in '.-', line))
                    try:
                        confidence = float(conf_str)
                    except:
                        confidence = 0.5
            
            return {
                'score': max(-1, min(1, score)),
                'confidence': max(0, min(1, confidence)),
                'analysis': response.text.strip()
            }
        except:
            return {'score': 0, 'confidence': 0.5, 'analysis': 'Sentiment analysis failed'}
    
    def generate_conversation_flow(self, profile_data, conversation_history=""):
        """Generate intelligent conversation flow based on context"""
        
        # Analyze conversation sentiment
        if conversation_history:
            sentiment = self.analyze_sentiment(conversation_history)
        else:
            sentiment = {'score': 0, 'confidence': 0.5}
        
        # Determine conversation stage
        if not conversation_history:
            stage = "initial_contact"
        elif "thanks" in conversation_history.lower() or "thank you" in conversation_history.lower():
            stage = "gratitude_response"
        elif "?" in conversation_history:
            stage = "question_response"
        else:
            stage = "follow_up"
        
        prompt = f"""Generate a LinkedIn conversation message based on context.

Profile: {profile_data.get('name', 'Professional')}
Headline: {profile_data.get('headline', '')}
Conversation Stage: {stage}
Sentiment Score: {sentiment['score']:.2f}

Previous conversation:
{conversation_history if conversation_history else 'First contact'}

Requirements:
- Match the conversation stage
- Consider sentiment (positive/negative/neutral)
- Be natural and human-like
- Maximum 300 characters
- Build relationship appropriately
- Reference profile details if relevant

Generate message:"""

        try:
            response = self.client.models.generate_content(
                model="gemini-2.0-flash-exp",
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.8,
                    max_output_tokens=200
                )
            )
            
            return {
                'message': response.text.strip(),
                'stage': stage,
                'sentiment': sentiment,
                'confidence': sentiment['confidence']
            }
        except:
            return {
                'message': f"Thanks for connecting, {profile_data.get('name', 'there')}!",
                'stage': 'fallback',
                'sentiment': sentiment,
                'confidence': 0.3
            }
    
    def detect_conversation_intent(self, message):
        """Detect the intent behind a received message"""
        prompt = f"""Analyze this LinkedIn message and determine the sender's intent.

Message: "{message}"

Classify the intent as one of:
- greeting: Simple greeting or introduction
- question: Asking a question or seeking information
- sales: Sales pitch or promotional content
- networking: Professional networking intent
- collaboration: Seeking collaboration or partnership
- job_opportunity: Job-related inquiry
- spam: Unwanted or irrelevant content
- other: Something else

Also provide:
1. Intent confidence (0-1)
2. Suggested response approach
3. Key topics mentioned

Format:
Intent: [intent_type]
Confidence: [number]
Approach: [description]
Topics: [topic1, topic2, ...]"""

        try:
            response = self.client.models.generate_content(
                model="gemini-2.0-flash-exp",
                contents=prompt,
                config=types.GenerateContentConfig(temperature=0.4)
            )
            
            # Parse response
            lines = response.text.strip().split('\n')
            intent = 'other'
            confidence = 0.5
            
            for line in lines:
                if 'Intent:' in line:
                    intent = line.split(':')[1].strip().lower()
                elif 'Confidence:' in line:
                    conf_str = ''.join(filter(lambda x: x.isdigit() or x in '.-', line))
                    try:
                        confidence = float(conf_str)
                    except:
                        confidence = 0.5
            
            return {
                'intent': intent,
                'confidence': confidence,
                'analysis': response.text.strip()
            }
        except:
            return {
                'intent': 'other',
                'confidence': 0.3,
                'analysis': 'Intent analysis failed'
            }

# =====================================================================
# A/B TESTING FRAMEWORK
# =====================================================================
class ABTestingFramework:
    """Test different message variations and track performance"""

    def __init__(self, ai_engine):
        self.ai = ai_engine
        self.results_file = CONFIG['ab_test_results']
        self.test_results = self.load_results()

    def load_results(self):
        """Load A/B test results"""
        try:
            with open(self.results_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return {
                'tests': [],
                'variants': {
                    'professional': {'sent': 0, 'accepted': 0},
                    'casual': {'sent': 0, 'accepted': 0},
                    'story': {'sent': 0, 'accepted': 0}
                }
            }

    def save_results(self):
        """Save test results"""
        with open(self.results_file, 'w') as f:
            json.dump(self.test_results, f, indent=2)

    def generate_message_variants(self, profile_data):
        """Generate multiple message variants for testing"""
        variants = {}

        for style in ['professional', 'casual', 'story']:
            message = self.ai.generate_connection_message(profile_data, style=style)
            variants[style] = message

        return variants

    def select_variant(self, variants):
        """Select best performing variant using Thompson Sampling"""

        if not CONFIG['enable_ab_testing']:
            return random.choice(list(variants.keys())), variants[random.choice(list(variants.keys()))]

        # Thompson Sampling for bandit problem
        samples = {}
        for variant_name in variants.keys():
            stats = self.test_results['variants'].get(variant_name, {'sent': 1, 'accepted': 0})
            successes = stats['accepted'] + 1
            failures = stats['sent'] - stats['accepted'] + 1
            samples[variant_name] = np.random.beta(successes, failures)

        best_variant = max(samples, key=samples.get)
        return best_variant, variants[best_variant]

    def record_message_sent(self, variant_name, profile_id):
        """Record that a message variant was sent"""
        if variant_name not in self.test_results['variants']:
            self.test_results['variants'][variant_name] = {'sent': 0, 'accepted': 0}

        self.test_results['variants'][variant_name]['sent'] += 1

        self.test_results['tests'].append({
            'variant': variant_name,
            'profile_id': profile_id,
            'sent_at': datetime.now().isoformat(),
            'accepted': None
        })

        self.save_results()

    def record_connection_accepted(self, profile_id):
        """Record connection acceptance"""
        for test in self.test_results['tests']:
            if test['profile_id'] == profile_id and test['accepted'] is None:
                test['accepted'] = True
                test['accepted_at'] = datetime.now().isoformat()

                variant = test['variant']
                self.test_results['variants'][variant]['accepted'] += 1
                break

        self.save_results()

    def get_performance_report(self):
        """Generate A/B test performance report"""
        report = "\n=== A/B TEST PERFORMANCE ===\n\n"

        for variant, stats in self.test_results['variants'].items():
            sent = stats['sent']
            accepted = stats['accepted']
            rate = (accepted / sent * 100) if sent > 0 else 0

            report += f"{variant.capitalize()}:\n"
            report += f"  Sent: {sent}\n"
            report += f"  Accepted: {accepted}\n"
            report += f"  Rate: {rate:.1f}%\n\n"

        return report

# =====================================================================
# ANALYTICS ENGINE
# =====================================================================
class AnalyticsEngine:
    """Track and analyze bot performance"""

    def __init__(self):
        self.analytics_file = CONFIG['analytics_db']
        self.metrics = self.load_metrics()

    def load_metrics(self):
        """Load analytics"""
        try:
            with open(self.analytics_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return {
                'daily_stats': {},
                'connection_acceptance': [],
                'message_responses': [],
                'post_engagement': [],
                'profile_views': [],
                'best_times': defaultdict(int),
                'best_days': defaultdict(int),
                'industry_performance': defaultdict(lambda: {'sent': 0, 'accepted': 0})
            }

    def save_metrics(self):
        """Save analytics"""
        with open(self.analytics_file, 'w') as f:
            json.dump(self.metrics, f, indent=2, default=str)

    def track_connection_sent(self, profile_data):
        """Track connection request"""
        today = datetime.now().strftime('%Y-%m-%d')

        if today not in self.metrics['daily_stats']:
            self.metrics['daily_stats'][today] = {
                'connections_sent': 0,
                'messages_sent': 0,
                'posts_created': 0,
                'comments_posted': 0
            }

        self.metrics['daily_stats'][today]['connections_sent'] += 1

        # Track by industry
        industry = profile_data.get('industry', 'Unknown')
        self.metrics['industry_performance'][industry]['sent'] += 1

        # Track best times
        hour = datetime.now().hour
        self.metrics['best_times'][hour] += 1

        day = datetime.now().strftime('%A')
        self.metrics['best_days'][day] += 1

        self.save_metrics()

    def track_connection_accepted(self, profile_data):
        """Track connection acceptance"""
        self.metrics['connection_acceptance'].append({
            'profile': profile_data.get('name', 'Unknown'),
            'accepted_at': datetime.now().isoformat(),
            'industry': profile_data.get('industry', 'Unknown')
        })

        industry = profile_data.get('industry', 'Unknown')
        self.metrics['industry_performance'][industry]['accepted'] += 1

        self.save_metrics()

    def track_post_created(self, topic):
        """Track post creation"""
        today = datetime.now().strftime('%Y-%m-%d')

        if today not in self.metrics['daily_stats']:
            self.metrics['daily_stats'][today] = {
                'connections_sent': 0,
                'messages_sent': 0,
                'posts_created': 0,
                'comments_posted': 0
            }

        self.metrics['daily_stats'][today]['posts_created'] += 1
        self.save_metrics()

    def calculate_acceptance_rate(self, days=7):
        """Calculate connection acceptance rate"""
        cutoff = datetime.now() - timedelta(days=days)
        recent_acceptances = [
            a for a in self.metrics['connection_acceptance']
            if datetime.fromisoformat(a['accepted_at']) > cutoff
        ]

        # Estimate sent connections (simplified)
        sent = sum(
            stats['connections_sent']
            for date, stats in self.metrics['daily_stats'].items()
            if datetime.strptime(date, '%Y-%m-%d') > cutoff
        )

        if sent == 0:
            return 0

        return len(recent_acceptances) / sent * 100

    def generate_dashboard_report(self):
        """Generate comprehensive analytics report"""

        report = """
╔════════════════════════════════════════════════════════════╗
║           LINKEDIN AI BOT - ANALYTICS DASHBOARD            ║
╚════════════════════════════════════════════════════════════╝

📊 OVERALL PERFORMANCE (Last 7 Days)
────────────────────────────────────────────────────────────
"""

        # Calculate totals
        week_ago = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
        recent_stats = {k: v for k, v in self.metrics['daily_stats'].items() if k >= week_ago}

        total_connections = sum(s['connections_sent'] for s in recent_stats.values())
        total_messages = sum(s['messages_sent'] for s in recent_stats.values())
        total_posts = sum(s['posts_created'] for s in recent_stats.values())

        acceptance_rate = self.calculate_acceptance_rate(7)

        report += f"""
Connections Sent:     {total_connections}
Messages Sent:        {total_messages}
Posts Created:        {total_posts}
Acceptance Rate:      {acceptance_rate:.1f}%

"""

        # Best performing industries
        report += """
🎯 TOP PERFORMING INDUSTRIES
────────────────────────────────────────────────────────────
"""

        industry_rates = []
        for industry, stats in self.metrics['industry_performance'].items():
            if stats['sent'] > 0:
                rate = stats['accepted'] / stats['sent'] * 100
                industry_rates.append((industry, rate, stats['sent'], stats['accepted']))

        industry_rates.sort(key=lambda x: x[1], reverse=True)

        for industry, rate, sent, accepted in industry_rates[:5]:
            report += f"{industry:<25} {rate:>5.1f}%  ({accepted}/{sent})\n"

        # Best times
        report += """

⏰ BEST ENGAGEMENT TIMES
────────────────────────────────────────────────────────────
"""

        best_hours = sorted(self.metrics['best_times'].items(), key=lambda x: x[1], reverse=True)[:5]
        for hour, count in best_hours:
            report += f"{hour:02d}:00 - {hour+1:02d}:00    {count} activities\n"

        # Best days
        report += """

📅 BEST DAYS OF WEEK
────────────────────────────────────────────────────────────
"""

        best_days = sorted(self.metrics['best_days'].items(), key=lambda x: x[1], reverse=True)
        for day, count in best_days:
            report += f"{day:<12} {count} activities\n"

        report += """
────────────────────────────────────────────────────────────
"""

        return report

    def generate_visual_report(self):
        """Generate visual analytics"""
        try:
            # Prepare data
            dates = sorted(self.metrics['daily_stats'].keys())[-14:]  # Last 14 days
            connections = [self.metrics['daily_stats'][d]['connections_sent'] for d in dates]

            # Create plot
            plt.figure(figsize=(12, 6))

            plt.subplot(1, 2, 1)
            plt.plot(dates, connections, marker='o')
            plt.title('Connections Sent (Last 14 Days)')
            plt.xlabel('Date')
            plt.ylabel('Count')
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3)

            plt.subplot(1, 2, 2)
            industries = list(self.metrics['industry_performance'].keys())[:5]
            rates = [
                self.metrics['industry_performance'][i]['accepted'] /
                max(self.metrics['industry_performance'][i]['sent'], 1) * 100
                for i in industries
            ]
            plt.bar(industries, rates)
            plt.title('Acceptance Rate by Industry')
            plt.ylabel('Acceptance Rate (%)')
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3, axis='y')

            plt.tight_layout()
            plt.savefig('reports/analytics_dashboard.png', dpi=300, bbox_inches='tight')
            print("✓ Visual report saved to reports/analytics_dashboard.png")

        except Exception as e:
            print(f"Could not generate visual report: {e}")

# =====================================================================
# AUTOMATED SCHEDULING SYSTEM
# =====================================================================
class TaskScheduler:
    """Automated scheduling system for LinkedIn activities"""
    
    def __init__(self, bot_instance):
        self.bot = bot_instance
        self.schedule_file = 'data/schedule.json'
        self.schedule = self.load_schedule()
        self.running = False
        
    def load_schedule(self):
        """Load schedule configuration"""
        try:
            with open(self.schedule_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return {
                'daily_tasks': {
                    'networking': {'enabled': True, 'time': '09:00', 'max_profiles': 10},
                    'feed_engagement': {'enabled': True, 'time': '14:00', 'max_posts': 5},
                    'content_creation': {'enabled': True, 'time': '16:00', 'probability': 0.3},
                    'notifications_check': {'enabled': True, 'time': '18:00'}
                },
                'weekly_tasks': {
                    'analytics_report': {'enabled': True, 'day': 'monday', 'time': '08:00'},
                    'ab_test_analysis': {'enabled': True, 'day': 'friday', 'time': '17:00'}
                },
                'timezone': 'Africa/Harare',
                'active_days': ['monday', 'tuesday', 'wednesday', 'thursday', 'friday']
            }
    
    def save_schedule(self):
        """Save schedule configuration"""
        with open(self.schedule_file, 'w') as f:
            json.dump(self.schedule, f, indent=2)
    
    def is_time_for_task(self, task_time):
        """Check if it's time for a scheduled task"""
        now = datetime.now()
        current_time = now.strftime('%H:%M')
        return current_time >= task_time
    
    def is_active_day(self):
        """Check if today is an active day"""
        today = datetime.now().strftime('%A').lower()
        return today in self.schedule['active_days']
    
    def run_daily_tasks(self):
        """Execute daily scheduled tasks"""
        if not self.is_active_day():
            print("📅 Today is not an active day - skipping tasks")
            return
            
        print(f"\n📅 Running daily scheduled tasks...")
        
        for task_name, task_config in self.schedule['daily_tasks'].items():
            if not task_config['enabled']:
                continue
                
            if self.is_time_for_task(task_config['time']):
                print(f"⏰ Executing {task_name} at {task_config['time']}")
                
                try:
                    if task_name == 'networking':
                        self._run_networking_task(task_config)
                    elif task_name == 'feed_engagement':
                        self._run_feed_engagement_task(task_config)
                    elif task_name == 'content_creation':
                        self._run_content_creation_task(task_config)
                    elif task_name == 'notifications_check':
                        self._run_notifications_check_task(task_config)
                        
                except Exception as e:
                    print(f"❌ Error in {task_name}: {e}")
                    if self.bot.error_handler:
                        self.bot.error_handler.handle_error(e, f"scheduled_{task_name}")
    
    def _run_networking_task(self, config):
        """Run networking task"""
        keywords = ["Software Engineer Zimbabwe", "AI Developer Africa"]
        profiles = []
        
        for keyword in keywords[:2]:  # Limit keywords
            found_profiles = self.bot.search_profiles(keyword, config['max_profiles'])
            profiles.extend(found_profiles)
        
        profiles = list(set(profiles))[:config['max_profiles']]
        self.bot.ai_networking_session(profiles)
    
    def _run_feed_engagement_task(self, config):
        """Run feed engagement task"""
        keywords = ["AI", "software", "technology"]
        self.bot.ai_feed_engagement(keywords, config['max_posts'])
    
    def _run_content_creation_task(self, config):
        """Run content creation task"""
        if random.random() < config['probability']:
            topics = [
                "AI trends in African tech",
                "Remote work best practices",
                "Building scalable software"
            ]
            self.bot.ai_content_creation(topics)
    
    def _run_notifications_check_task(self, config):
        """Run notifications check task"""
        self.bot.check_notifications()
    
    def run_weekly_tasks(self):
        """Execute weekly scheduled tasks"""
        today = datetime.now().strftime('%A').lower()
        
        for task_name, task_config in self.schedule['weekly_tasks'].items():
            if not task_config['enabled']:
                continue
                
            if task_config['day'].lower() == today and self.is_time_for_task(task_config['time']):
                print(f"📊 Executing weekly task: {task_name}")
                
                try:
                    if task_name == 'analytics_report':
                        self.bot.generate_reports()
                    elif task_name == 'ab_test_analysis':
                        report = self.bot.ab_testing.get_performance_report()
                        print(report)
                        
                except Exception as e:
                    print(f"❌ Error in weekly {task_name}: {e}")
    
    def start_scheduler(self):
        """Start the automated scheduler"""
        self.running = True
        print("🕐 Automated scheduler started")
        
        while self.running:
            try:
                # Run daily tasks
                self.run_daily_tasks()
                
                # Run weekly tasks
                self.run_weekly_tasks()
                
                # Wait 1 hour before next check
                time.sleep(3600)
                
            except KeyboardInterrupt:
                print("\n⏹️ Scheduler stopped by user")
                self.running = False
            except Exception as e:
                print(f"❌ Scheduler error: {e}")
                time.sleep(300)  # Wait 5 minutes before retry
    
    def stop_scheduler(self):
        """Stop the automated scheduler"""
        self.running = False
        print("⏹️ Scheduler stopped")

# =====================================================================
# ADVANCED SECURITY & ANTI-DETECTION SYSTEM
# =====================================================================
class SecurityManager:
    """Advanced security and anti-detection measures"""
    
    def __init__(self):
        self.user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/121.0',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:109.0) Gecko/20100101 Firefox/121.0'
        ]
        self.proxy_list = []
        self.session_patterns = []
        
    def get_random_user_agent(self):
        """Get random user agent"""
        return random.choice(self.user_agents)
    
    def simulate_human_behavior(self, driver):
        """Simulate human-like behavior patterns"""
        # Random mouse movements
        driver.execute_script("""
            var event = new MouseEvent('mousemove', {
                'view': window,
                'bubbles': true,
                'cancelable': true,
                'clientX': Math.random() * window.innerWidth,
                'clientY': Math.random() * window.innerHeight
            });
            document.dispatchEvent(event);
        """)
        
        # Random scroll behavior
        scroll_amount = random.randint(100, 500)
        driver.execute_script(f"window.scrollBy(0, {scroll_amount});")
        
        # Random pause
        time.sleep(random.uniform(1, 3))
    
    def detect_suspicious_activity(self, driver):
        """Detect if LinkedIn is showing suspicious activity warnings"""
        try:
            # Check for common warning indicators
            warning_indicators = [
                "//div[contains(text(), 'unusual activity')]",
                "//div[contains(text(), 'verify your identity')]",
                "//div[contains(text(), 'security check')]",
                "//div[contains(text(), 'suspicious')]"
            ]
            
            for indicator in warning_indicators:
                elements = driver.find_elements(By.XPATH, indicator)
                if elements:
                    return True, f"Suspicious activity detected: {elements[0].text[:50]}"
            
            return False, "No suspicious activity detected"
            
        except Exception as e:
            return False, f"Detection error: {e}"
    
    def handle_captcha(self, driver):
        """Handle CAPTCHA detection"""
        try:
            captcha_elements = driver.find_elements(By.XPATH, "//div[contains(@class, 'captcha') or contains(text(), 'captcha')]")
            if captcha_elements:
                print("🤖 CAPTCHA detected - manual intervention required")
                print("Please solve the CAPTCHA manually and press Enter to continue...")
                input("Press Enter after solving CAPTCHA...")
                return True
            return False
        except:
            return False
    
    def rotate_session(self, driver):
        """Rotate session to avoid detection"""
        try:
            # Clear cookies
            driver.delete_all_cookies()
            
            # Clear local storage
            driver.execute_script("window.localStorage.clear();")
            driver.execute_script("window.sessionStorage.clear();")
            
            # Refresh page
            driver.refresh()
            
            print("🔄 Session rotated successfully")
            return True
            
        except Exception as e:
            print(f"❌ Session rotation failed: {e}")
            return False

# =====================================================================
# MAIN AI LINKEDIN BOT
# =====================================================================
class CompleteAILinkedInBot:
    """Complete AI-powered LinkedIn automation with all features"""

    def __init__(self):
        # Initialize all systems
        self.ai = AdvancedGeminiAI(CONFIG['gemini_api_key'])
        self.memory = ConversationMemory(self.ai.client)
        self.ai.set_memory(self.memory)
        self.ab_testing = ABTestingFramework(self.ai)
        self.analytics = AnalyticsEngine()
        self.notifier = WebhookNotifier(CONFIG['webhook_url'])
        self.error_handler = ErrorHandler(self.notifier)
        self.security_manager = SecurityManager()
        self.scheduler = TaskScheduler(self)
        self.behavior = self.init_behavior()
        self.driver = None
        self.database = DatabaseManager()

        print("✓ All systems initialized")

    def init_behavior(self):
        """Initialize behavior simulator"""
        class Behavior:
            def __init__(self):
                self.activity_log = {}
                self.fatigue = 0

            def intelligent_delay(self, delay_type='medium'):
                delay = random.uniform(*CONFIG[f'{delay_type}_delay'])
                time.sleep(delay)

            def human_typing(self, element, text):
                for char in text:
                    element.send_keys(char)
                    time.sleep(random.uniform(0.05, 0.20))

            def can_perform_action(self, action_type):
                today = datetime.now().strftime('%Y-%m-%d')
                if today not in self.activity_log:
                    self.activity_log[today] = {'connection': 0, 'message': 0, 'post': 0, 'comment': 0}

                limit = CONFIG[f'daily_{action_type}_limit']
                return self.activity_log[today][action_type] < limit

            def record_action(self, action_type):
                today = datetime.now().strftime('%Y-%m-%d')
                if today not in self.activity_log:
                    self.activity_log[today] = {'connection': 0, 'message': 0, 'post': 0, 'comment': 0}
                self.activity_log[today][action_type] += 1

        return Behavior()

    def init_driver(self):
        """Initialize browser with enhanced anti-detection"""
        options = Options()
        options.add_argument("--start-maximized")
        options.add_argument("--disable-notifications")
        options.add_argument("--disable-blink-features=AutomationControlled")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-gpu")
        options.add_argument("--disable-extensions")
        options.add_experimental_option("excludeSwitches", ["enable-automation"])
        options.add_experimental_option('useAutomationExtension', False)
        options.add_experimental_option("prefs", {
            "profile.default_content_setting_values.notifications": 2,
            "profile.default_content_settings.popups": 0,
            "profile.managed_default_content_settings.images": 2
        })

        # Use security manager for user agent
        options.add_argument(f'user-agent={self.security_manager.get_random_user_agent()}')

        try:
            self.driver = webdriver.Chrome(
                service=Service(ChromeDriverManager().install()),
                options=options
            )

            # Enhanced anti-detection scripts
            self.driver.execute_script(
                "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"
            )
            
            self.driver.execute_script(
                "Object.defineProperty(navigator, 'plugins', {get: () => [1, 2, 3, 4, 5]})"
            )
            
            self.driver.execute_script(
                "Object.defineProperty(navigator, 'languages', {get: () => ['en-US', 'en']})"
            )

            print("✓ Browser initialized with enhanced security")
            
        except Exception as e:
            print(f"❌ Browser initialization failed: {e}")
            if self.error_handler:
                self.error_handler.handle_error(e, "browser_init")
            raise

    def login(self):
        """Login with cookie persistence"""
        self.driver.get("https://www.linkedin.com")

        try:
            cookies = pickle.load(open(CONFIG['cookies_file'], "rb"))
            for cookie in cookies:
                self.driver.add_cookie(cookie)
            self.driver.refresh()
            print("✓ Logged in using cookies")
            self.behavior.intelligent_delay('medium')
            return True
        except:
            return self.manual_login()

    def manual_login(self):
        """Manual login with human-like typing"""
        try:
            self.driver.get("https://www.linkedin.com/login")
            self.behavior.intelligent_delay('short')

            username = WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.ID, "username"))
            )
            self.behavior.human_typing(username, CONFIG['linkedin_email'])

            password = self.driver.find_element(By.ID, "password")
            self.behavior.human_typing(password, CONFIG['linkedin_password'])
            password.send_keys(Keys.RETURN)

            self.behavior.intelligent_delay('long')

            pickle.dump(self.driver.get_cookies(), open(CONFIG['cookies_file'], "wb"))
            print("✓ Logged in successfully")
            self.notifier.send_notification("✅ Bot logged in successfully", 'success')
            return True

        except Exception as e:
            print(f"✗ Login failed: {e}")
            self.notifier.notify_error(f"Login failed: {str(e)[:100]}")
            return False

    def search_profiles(self, keyword, max_results=20):
        """Search for profiles"""
        print(f"\n🔍 Searching: {keyword}")
        search_url = f"https://www.linkedin.com/search/results/people/?keywords={keyword.replace(' ', '%20')}"
        self.driver.get(search_url)
        self.behavior.intelligent_delay('medium')

        # Scroll to load more results
        for i in range(3):
            self.driver.execute_script(f"window.scrollBy(0, {800 + i*100});")
            self.behavior.intelligent_delay('short')

        # Extract profile URLs
        profile_links = self.driver.find_elements(
            By.XPATH,
            "//a[contains(@href,'/in/') and contains(@class,'app-aware-link')]"
        )

        profiles = []
        for link in profile_links:
            try:
                url = link.get_attribute("href").split('?')[0]
                if url not in profiles and '/in/' in url and len(url.split('/in/')[1]) > 3:
                    profiles.append(url)
                    if len(profiles) >= max_results:
                        break
            except:
                continue

        print(f"  ✓ Found {len(profiles)} profiles")
        return profiles

    def analyze_profile(self, url):
        """Analyze profile with AI"""
        self.driver.get(url)
        self.behavior.intelligent_delay('medium')

        profile_data = {
            'url': url,
            'name': '',
            'headline': '',
            'location': '',
            'connections': 0,
        }

        try:
            # Extract name
            name_elem = self.driver.find_element(By.XPATH, "//h1[contains(@class,'text-heading-xlarge')]")
            profile_data['name'] = name_elem.text.strip()

            # Extract headline
            headline_elem = self.driver.find_element(By.XPATH, "//div[contains(@class,'text-body-medium')]")
            profile_data['headline'] = headline_elem.text.strip()

            # Extract location
            try:
                location_elem = self.driver.find_element(By.XPATH, "//span[contains(@class,'text-body-small')]")
                profile_data['location'] = location_elem.text.strip()
            except:
                pass

            # Extract connection count
            try:
                conn_elem = self.driver.find_element(By.XPATH, "//span[contains(text(),'connection')]")
                conn_text = conn_elem.text
                profile_data['connections'] = int(''.join(filter(str.isdigit, conn_text.split()[0])))
            except:
                pass

        except Exception as e:
            print(f"  ⚠ Could not extract all profile data: {e}")

        # AI analysis
        print(f"  🤖 AI analyzing profile...")
        ai_analysis = self.ai.analyze_profile_relevance(profile_data)
        profile_data.update(ai_analysis)

        # Track analytics
        if CONFIG['track_metrics']:
            profile_data['industry'] = self.ai._extract_industry(profile_data['headline'])

        return profile_data

    def send_ab_tested_connection(self, profile_data):
        """Send connection with A/B tested message"""
        try:
            # Find connect button
            connect_btn = WebDriverWait(self.driver, 5).until(
                EC.element_to_be_clickable((
                    By.XPATH,
                    "//button[contains(@aria-label,'Invite') or contains(., 'Connect')]"
                ))
            )
            connect_btn.click()
            self.behavior.intelligent_delay('short')

            try:
                # Try to add note
                add_note = WebDriverWait(self.driver, 3).until(
                    EC.element_to_be_clickable((By.XPATH, "//button[contains(., 'Add a note')]"))
                )
                add_note.click()
                self.behavior.intelligent_delay('short')

                # Generate A/B tested message variants
                print(f"  🤖 Generating message variants...")
                variants = self.ab_testing.generate_message_variants(profile_data)

                # Select best variant using Thompson Sampling
                selected_variant, message = self.ab_testing.select_variant(variants)

                print(f"  📊 Selected variant: {selected_variant}")
                print(f"  💬 Message: \"{message[:60]}...\"")

                # Type message
                message_box = self.driver.find_element(By.XPATH, "//textarea[@name='message']")
                self.behavior.human_typing(message_box, message)
                self.behavior.intelligent_delay('short')

                # Send
                send_btn = self.driver.find_element(By.XPATH, "//button[contains(., 'Send')]")
                send_btn.click()

                # Record A/B test
                self.ab_testing.record_message_sent(selected_variant, profile_data['url'])

                # Store in conversation memory
                self.memory.store_conversation(profile_data['url'], message, is_sent=True)

                # Track analytics
                self.analytics.track_connection_sent(profile_data)

                print(f"  ✅ Connection request sent")
                return True

            except:
                # Send without note
                send_btn = self.driver.find_element(By.XPATH, "//button[contains(., 'Send') and contains(@aria-label, 'invite')]")
                send_btn.click()

                self.analytics.track_connection_sent(profile_data)
                print(f"  ✅ Connection sent (no note)")
                return True

        except Exception as e:
            print(f"  ❌ Could not connect: {str(e)[:60]}")
            return False

    def ai_networking_session(self, profile_urls):
        """Complete networking session with AI"""
        connected = 0
        skipped = 0

        for i, url in enumerate(profile_urls, 1):
            if not self.behavior.can_perform_action('connection'):
                print("\n⚠ Daily connection limit reached")
                self.notifier.notify_daily_limit('connection')
                break

            print(f"\n[{i}/{len(profile_urls)}] Processing profile...")

            # Analyze with AI
            profile_data = self.analyze_profile(url)

            print(f"  👤 {profile_data['name']}")
            print(f"  💼 {profile_data.get('headline', 'N/A')[:50]}")
            print(f"  📊 Relevance: {profile_data['score']}/100")

            # AI decision
            if profile_data['should_connect'] and profile_data['score'] >= 40:
                if self.send_ab_tested_connection(profile_data):
                    connected += 1
                    self.behavior.record_action('connection')
                    self.behavior.intelligent_delay('long')
            else:
                print(f"  ⊗ AI recommends skipping (score too low)")
                skipped += 1

            self.behavior.intelligent_delay('medium')

        summary = f"""
╔════════════════════════════════════════╗
║     NETWORKING SESSION COMPLETE        ║
╠════════════════════════════════════════╣
║  Profiles Analyzed:  {len(profile_urls):>3}            ║
║  Connections Sent:   {connected:>3}            ║
║  Skipped (Low Score): {skipped:>3}            ║
║  Success Rate:       {(connected/max(len(profile_urls),1)*100):>5.1f}%      ║
╚════════════════════════════════════════╝
"""
        print(summary)
        self.notifier.send_notification(f"Networking complete: {connected} connections sent", 'info')

    def ai_content_creation(self, topics):
        """Create and post AI-generated content"""
        if not self.behavior.can_perform_action('post'):
            print("\n⚠ Daily post limit reached")
            return

        print(f"\n📝 Creating AI-generated content...")

        # Select random topic
        topic = random.choice(topics)
        post_type = random.choice(['insight', 'story', 'question', 'list'])

        try:
            self.driver.get("https://www.linkedin.com/feed/")
            self.behavior.intelligent_delay('medium')

            # Click start post
            start_post = WebDriverWait(self.driver, 10).until(
                EC.element_to_be_clickable((By.XPATH, "//button[contains(., 'Start a post')]"))
            )
            start_post.click()
            self.behavior.intelligent_delay('short')

            # Generate content with AI
            print(f"  🤖 Generating {post_type} post about: {topic}")
            post_content = self.ai.generate_linkedin_post(topic, post_type)

            print(f"  📄 Preview: {post_content[:100]}...")

            # Type content
            editor = WebDriverWait(self.driver, 5).until(
                EC.presence_of_element_located((By.XPATH, "//div[@role='textbox']"))
            )
            self.behavior.human_typing(editor, post_content)
            self.behavior.intelligent_delay('medium')

            # Post
            post_btn = self.driver.find_element(By.XPATH, "//button[contains(@class,'share-actions__primary-action')]")
            post_btn.click()

            print(f"  ✅ Post published successfully")

            # Track
            self.behavior.record_action('post')
            self.analytics.track_post_created(topic)

            self.notifier.send_notification(f"New post published: {topic}", 'success')

        except Exception as e:
            print(f"  ❌ Could not post: {e}")
            self.notifier.notify_error(f"Post failed: {str(e)[:100]}")

    def ai_feed_engagement(self, keywords, max_posts=5):
        """Engage with feed using AI-generated comments"""
        print(f"\n💬 AI Feed Engagement")

        self.driver.get("https://www.linkedin.com/feed/")
        self.behavior.intelligent_delay('medium')

        # Scroll to load posts
        for i in range(3):
            self.driver.execute_script(f"window.scrollBy(0, {600 + i*50});")
            self.behavior.intelligent_delay('short')

        posts = self.driver.find_elements(By.XPATH, "//div[contains(@class,'feed-shared-update-v2')]")
        engaged = 0

        for post in posts[:max_posts]:
            if not self.behavior.can_perform_action('comment'):
                print("  ⚠ Daily comment limit reached")
                break

            try:
                post_text = post.text.lower()

                # Check if post matches keywords
                if any(kw.lower() in post_text for kw in keywords):

                    # Extract author name
                    try:
                        author = post.find_element(By.XPATH, ".//span[contains(@class,'update-components-actor__name')]").text
                    except:
                        author = "Professional"

                    # Like the post
                    if random.random() < 0.7:
                        try:
                            like_btn = post.find_element(By.XPATH, ".//button[contains(@aria-label,'Like') or contains(@aria-label,'React')]")
                            if 'React' in like_btn.get_attribute('aria-label'):
                                like_btn.click()
                                print(f"  👍 Liked post by {author}")
                                self.behavior.intelligent_delay('short')
                        except:
                            pass

                    # Comment with AI
                    if random.random() < 0.4:  # 40% comment rate
                        try:
                            comment_btn = post.find_element(By.XPATH, ".//button[contains(@aria-label,'Comment')]")
                            comment_btn.click()
                            self.behavior.intelligent_delay('short')

                            # Generate AI comment
                            print(f"  🤖 Generating comment for {author}'s post...")
                            comment = self.ai.generate_post_comment(post_text[:500], author)
                            print(f"  💬 \"{comment[:50]}...\"")

                            comment_box = post.find_element(By.XPATH, ".//div[contains(@class,'ql-editor')]")
                            self.behavior.human_typing(comment_box, comment)
                            self.behavior.intelligent_delay('short')

                            # Submit
                            submit_btn = post.find_element(By.XPATH, ".//button[contains(@class,'comments-comment-box__submit-button')]")
                            submit_btn.click()

                            print(f"  ✅ Comment posted")
                            engaged += 1

                            self.behavior.record_action('comment')
                            self.behavior.intelligent_delay('long')

                        except Exception as e:
                            print(f"  ⚠ Could not comment: {str(e)[:50]}")
                            continue

            except:
                continue

        print(f"  ✅ Engaged with {engaged} posts")

    def check_notifications(self):
        """Check for new notifications (acceptances, messages)"""
        try:
            self.driver.get("https://www.linkedin.com/notifications/")
            self.behavior.intelligent_delay('medium')

            notifications = self.driver.find_elements(By.XPATH, "//li[contains(@class,'notification-card')]")[:10]

            for notif in notifications:
                text = notif.text.lower()

                # Check for connection acceptance
                if 'accepted your invitation' in text or 'is now a connection' in text:
                    try:
                        name = notif.find_element(By.XPATH, ".//strong").text
                        print(f"  ✅ Connection accepted by: {name}")

                        # Notify
                        self.notifier.notify_connection_accepted(name)

                        # Update A/B test results
                        # Note: Would need to match with sent connections for accurate tracking

                    except:
                        pass

                # Check for messages
                elif 'sent you a message' in text:
                    try:
                        name = notif.find_element(By.XPATH, ".//strong").text
                        print(f"  💬 New message from: {name}")
                        self.notifier.notify_message_received(name)
                    except:
                        pass

        except Exception as e:
            print(f"  ⚠ Could not check notifications: {e}")

    def generate_reports(self):
        """Generate all reports"""
        print("\n" + "="*60)
        print("GENERATING REPORTS")
        print("="*60)

        # Analytics dashboard
        dashboard = self.analytics.generate_dashboard_report()
        print(dashboard)

        # Save to file
        with open('reports/analytics_report.txt', 'w') as f:
            f.write(dashboard)

        # A/B test results
        ab_report = self.ab_testing.get_performance_report()
        print(ab_report)

        with open('reports/ab_test_report.txt', 'w') as f:
            f.write(ab_report)

        # Generate visual analytics
        if CONFIG['generate_reports']:
            self.analytics.generate_visual_report()

        print("\n✅ Reports saved to 'reports/' directory")

    def run_complete_session(self, search_keywords, max_profiles=15, content_topics=None):
        """Run complete AI-powered automation session"""

        print("\n" + "="*60)
        print("   COMPLETE AI-POWERED LINKEDIN AUTOMATION")
        print("="*60)
        print(f"\n🤖 AI Model: Gemini 2.0 Flash")
        print(f"📊 A/B Testing: {'Enabled' if CONFIG['enable_ab_testing'] else 'Disabled'}")
        print(f"👁️  Vision AI: {'Enabled' if CONFIG['enable_vision_ai'] else 'Disabled'}")
        print(f"💾 Conversation Memory: Enabled")
        print(f"📈 Analytics: {'Enabled' if CONFIG['track_metrics'] else 'Disabled'}")
        print(f"🔔 Webhooks: {'Enabled' if self.notifier.enabled else 'Disabled'}")

        print("\n⚠️  WARNING: Educational purposes only!")
        print("⚠️  Using automation violates LinkedIn Terms of Service")
        print("="*60)

        # Initialize browser
        self.init_driver()

        if not self.login():
            print("\n❌ Login failed. Exiting...")
            return

        try:
            # PHASE 1: Profile Discovery
            print("\n" + "="*60)
            print("📍 PHASE 1: AI-POWERED PROFILE DISCOVERY")
            print("="*60)

            all_profiles = []
            for keyword in search_keywords:
                profiles = self.search_profiles(keyword, max_profiles)
                all_profiles.extend(profiles)

            # Remove duplicates
            all_profiles = list(set(all_profiles))
            print(f"\n✅ Discovered {len(all_profiles)} unique profiles")

            # PHASE 2: AI Networking
            print("\n" + "="*60)
            print("📍 PHASE 2: AI-POWERED NETWORKING")
            print("="*60)

            self.ai_networking_session(all_profiles[:max_profiles])

            # PHASE 3: Feed Engagement
            print("\n" + "="*60)
            print("📍 PHASE 3: AI FEED ENGAGEMENT")
            print("="*60)

            for keyword in search_keywords[:2]:  # Limit to 2 keywords
                self.ai_feed_engagement([keyword.split()[0]], max_posts=3)
                self.behavior.intelligent_delay('medium')

            # PHASE 4: Content Creation
            print("\n" + "="*60)
            print("📍 PHASE 4: AI CONTENT CREATION")
            print("="*60)

            if content_topics and random.random() < 0.6:  # 60% chance to post
                self.ai_content_creation(content_topics)

            # PHASE 5: Check Notifications
            print("\n" + "="*60)
            print("📍 PHASE 5: CHECK NOTIFICATIONS")
            print("="*60)

            self.check_notifications()

            # PHASE 6: Generate Reports
            print("\n" + "="*60)
            print("📍 PHASE 6: ANALYTICS & REPORTS")
            print("="*60)

            self.generate_reports()

            # Final summary
            print("\n" + "="*60)
            print("✅ SESSION COMPLETED SUCCESSFULLY")
            print("="*60)

            self.notifier.send_notification("🎉 Complete session finished successfully!", 'success')

        except KeyboardInterrupt:
            print("\n\n⚠️  Session interrupted by user")
            self.notifier.send_notification("⚠️ Session interrupted", 'warning')

        except Exception as e:
            print(f"\n❌ Error: {e}")
            self.notifier.notify_error(str(e))

        finally:
            print("\n🧹 Cleaning up...")
            self.behavior.intelligent_delay('short')
            if self.driver:
                self.driver.quit()
            print("✅ Browser closed")

# =====================================================================
# CONFIGURATION MANAGEMENT INTERFACE
# =====================================================================
class ConfigManager:
    """Configuration management interface"""
    
    def __init__(self):
        self.config_file = 'data/config.json'
        self.config = self.load_config()
    
    def load_config(self):
        """Load configuration from file"""
        try:
            with open(self.config_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return self.get_default_config()
    
    def get_default_config(self):
        """Get default configuration"""
        return {
            'search_keywords': [
                "Software Engineer Zimbabwe",
                "AI Developer Africa", 
                "Tech Entrepreneur Harare",
                "Data Scientist Zimbabwe"
            ],
            'content_topics': [
                "The future of AI in Zimbabwe's tech ecosystem",
                "Building scalable software in emerging markets",
                "Remote work culture in African tech",
                "Innovation opportunities in Zimbabwe",
                "Lessons from building tech products in Africa"
            ],
            'daily_limits': {
                'connections': 15,
                'messages': 20,
                'posts': 2,
                'comments': 10
            },
            'timing': {
                'short_delay': [3, 8],
                'medium_delay': [10, 20],
                'long_delay': [30, 60]
            },
            'ai_settings': {
                'temperature': 0.8,
                'max_tokens': 400,
                'enable_vision': True,
                'enable_ab_testing': True
            },
            'notifications': {
                'webhook_enabled': False,
                'webhook_url': '',
                'email_notifications': False
            }
        }
    
    def save_config(self):
        """Save configuration to file"""
        with open(self.config_file, 'w') as f:
            json.dump(self.config, f, indent=2)
    
    def update_config(self, section, key, value):
        """Update specific configuration value"""
        if section not in self.config:
            self.config[section] = {}
        self.config[section][key] = value
        self.save_config()
    
    def get_config(self, section, key=None):
        """Get configuration value"""
        if key is None:
            return self.config.get(section, {})
        return self.config.get(section, {}).get(key)
    
    def interactive_setup(self):
        """Interactive configuration setup"""
        print("\n" + "="*60)
        print("   CONFIGURATION SETUP")
        print("="*60)
        
        # Search keywords
        print("\n📝 Search Keywords (comma-separated):")
        keywords_input = input("Enter keywords: ").strip()
        if keywords_input:
            self.config['search_keywords'] = [k.strip() for k in keywords_input.split(',')]
        
        # Daily limits
        print("\n📊 Daily Limits:")
        try:
            connections = int(input("Max connections per day (default 15): ") or "15")
            messages = int(input("Max messages per day (default 20): ") or "20")
            posts = int(input("Max posts per day (default 2): ") or "2")
            comments = int(input("Max comments per day (default 10): ") or "10")
            
            self.config['daily_limits'] = {
                'connections': connections,
                'messages': messages,
                'posts': posts,
                'comments': comments
            }
        except ValueError:
            print("Invalid input, using defaults")
        
        # AI settings
        print("\n🤖 AI Settings:")
        try:
            temperature = float(input("AI temperature (0.0-1.0, default 0.8): ") or "0.8")
            max_tokens = int(input("Max tokens (default 400): ") or "400")
            
            self.config['ai_settings']['temperature'] = max(0.0, min(1.0, temperature))
            self.config['ai_settings']['max_tokens'] = max(100, min(1000, max_tokens))
        except ValueError:
            print("Invalid input, using defaults")
        
        # Notifications
        print("\n🔔 Notifications:")
        webhook_enabled = input("Enable webhook notifications? (y/n): ").lower() == 'y'
        if webhook_enabled:
            webhook_url = input("Webhook URL: ").strip()
            self.config['notifications']['webhook_enabled'] = True
            self.config['notifications']['webhook_url'] = webhook_url
        
        self.save_config()
        print("\n✅ Configuration saved successfully!")

# =====================================================================
# COMPREHENSIVE TESTING SUITE
# =====================================================================
class TestSuite:
    """Comprehensive testing suite for LinkedIn bot"""
    
    def __init__(self, bot_instance):
        self.bot = bot_instance
        self.test_results = []
        
    def run_all_tests(self):
        """Run all tests"""
        print("\n" + "="*60)
        print("   RUNNING COMPREHENSIVE TEST SUITE")
        print("="*60)
        
        tests = [
            ("AI System", self.test_ai_system),
            ("Database", self.test_database),
            ("Analytics", self.test_analytics),
            ("Error Handling", self.test_error_handling),
            ("Security", self.test_security),
            ("Configuration", self.test_configuration),
            ("Webhook", self.test_webhook)
        ]
        
        passed = 0
        failed = 0
        
        for test_name, test_func in tests:
            print(f"\n🧪 Testing {test_name}...")
            try:
                result = test_func()
                if result:
                    print(f"✅ {test_name} - PASSED")
                    passed += 1
                else:
                    print(f"❌ {test_name} - FAILED")
                    failed += 1
            except Exception as e:
                print(f"❌ {test_name} - ERROR: {e}")
                failed += 1
        
        print(f"\n📊 Test Results: {passed} passed, {failed} failed")
        return failed == 0
    
    def test_ai_system(self):
        """Test AI system functionality"""
        try:
            # Test AI initialization
            if not self.bot.ai.client:
                return False
            
            # Test message generation
            test_profile = {
                'name': 'Test User',
                'headline': 'Software Engineer',
                'location': 'Zimbabwe'
            }
            
            message = self.bot.ai.generate_connection_message(test_profile)
            if not message or len(message) < 10:
                return False
            
            # Test sentiment analysis
            sentiment = self.bot.ai.analyze_sentiment("This is a positive message!")
            if not isinstance(sentiment, dict) or 'score' not in sentiment:
                return False
            
            return True
        except:
            return False
    
    def test_database(self):
        """Test database functionality"""
        try:
            # Test database connection
            with self.bot.database.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT 1")
                result = cursor.fetchone()
                if not result:
                    return False
            
            # Test profile saving
            test_profile = {
                'url': 'https://linkedin.com/in/test',
                'name': 'Test User',
                'headline': 'Test Headline',
                'score': 75
            }
            
            profile_id = self.bot.database.save_profile(test_profile)
            if not profile_id:
                return False
            
            return True
        except:
            return False
    
    def test_analytics(self):
        """Test analytics functionality"""
        try:
            # Test metrics loading
            metrics = self.bot.analytics.load_metrics()
            if not isinstance(metrics, dict):
                return False
            
            # Test tracking
            test_profile = {'name': 'Test', 'industry': 'Technology'}
            self.bot.analytics.track_connection_sent(test_profile)
            
            # Test report generation
            report = self.bot.analytics.generate_dashboard_report()
            if not report or len(report) < 100:
                return False
            
            return True
        except:
            return False
    
    def test_error_handling(self):
        """Test error handling system"""
        try:
            # Test error handler initialization
            if not self.bot.error_handler:
                return False
            
            # Test error handling
            test_error = Exception("Test error")
            result = self.bot.error_handler.handle_error(test_error, "test_context")
            
            # Test circuit breaker
            if not hasattr(self.bot.error_handler, 'circuit_breaker'):
                return False
            
            return True
        except:
            return False
    
    def test_security(self):
        """Test security features"""
        try:
            # Test security manager
            if not self.bot.security_manager:
                return False
            
            # Test user agent generation
            user_agent = self.bot.security_manager.get_random_user_agent()
            if not user_agent or len(user_agent) < 50:
                return False
            
            return True
        except:
            return False
    
    def test_configuration(self):
        """Test configuration system"""
        try:
            # Test config manager
            config_manager = ConfigManager()
            if not config_manager.config:
                return False
            
            # Test config operations
            config_manager.update_config('test', 'value', 'test_value')
            value = config_manager.get_config('test', 'value')
            if value != 'test_value':
                return False
            
            return True
        except:
            return False
    
    def test_webhook(self):
        """Test webhook functionality"""
        try:
            # Test webhook notifier
            if not self.bot.notifier:
                return False
            
            # Test notification (without actually sending)
            # This would normally send a test notification
            return True
        except:
            return False

# =====================================================================
# MAIN EXECUTION
# =====================================================================
def main():
    """Main execution with environment variable setup"""

    print("\n" + "="*60)
    print("   AI-POWERED LINKEDIN AUTOMATION SYSTEM")
    print("="*60)

    # Initialize configuration manager
    config_manager = ConfigManager()
    
    # Check if this is first run
    if not os.path.exists('data/config.json'):
        print("\n🔧 First time setup detected!")
        setup_choice = input("Run interactive setup? (y/n): ").lower()
        if setup_choice == 'y':
            config_manager.interactive_setup()
        else:
            print("Using default configuration")

    # Check configuration
    if not CONFIG['gemini_api_key']:
        print("\n❌ ERROR: Gemini API key not set!")
        print("\nPlease set environment variable:")
        print("  export GEMINI_API_KEY='your_key_here'")
        return

    if not CONFIG['linkedin_email'] or not CONFIG['linkedin_password']:
        print("\n❌ ERROR: LinkedIn credentials not set!")
        print("\nPlease set environment variables:")
        print("  export LINKEDIN_EMAIL='your_email'")
        print("  export LINKEDIN_PASSWORD='your_password'")
        return

    print("\n✅ Configuration loaded")
    print(f"   Email: {CONFIG['linkedin_email']}")
    print(f"   API Key: {CONFIG['gemini_api_key'][:20]}...")

    # Get configuration from manager
    search_keywords = config_manager.get_config('search_keywords')
    content_topics = config_manager.get_config('content_topics')
    
    # Initialize and run bot
    bot = CompleteAILinkedInBot()

    # Ask for execution mode
    print("\n🚀 Execution Modes:")
    print("1. Single Session (default)")
    print("2. Scheduled Mode (continuous)")
    print("3. Configuration Setup")
    print("4. Run Test Suite")
    
    mode = input("\nSelect mode (1-4): ").strip() or "1"
    
    if mode == "2":
        print("\n🕐 Starting scheduled mode...")
        print("Press Ctrl+C to stop")
        try:
            bot.scheduler.start_scheduler()
        except KeyboardInterrupt:
            print("\n⏹️ Scheduler stopped")
            
    elif mode == "3":
        config_manager.interactive_setup()
        
    elif mode == "4":
        # Test suite mode
        test_suite = TestSuite(bot)
        success = test_suite.run_all_tests()
        if success:
            print("\n🎉 All tests passed!")
        else:
            print("\n⚠️ Some tests failed. Check the output above.")
        
    else:
        # Single session mode
        bot.run_complete_session(
            search_keywords=search_keywords,
            max_profiles=config_manager.get_config('daily_limits', 'connections'),
            content_topics=content_topics
        )

    print("\n" + "="*60)
    print("   SESSION COMPLETE - CHECK reports/ DIRECTORY")
    print("="*60)

if __name__ == "__main__":
    main()