"""
Carrier Sales API - Backend Server with HappyRobot API Integration
Real-time data from HappyRobot platform with FMCSA API integration
"""

import os
import logging
import mysql.connector
import httpx
import json
import re
import time
import asyncio
from datetime import datetime, timezone, timedelta
from typing import Optional, List, Dict, Any
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Depends, Request, BackgroundTasks
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator
import uvicorn
from dotenv import load_dotenv
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy import Column, String, Integer, Text, Float, DateTime, text
from sqlalchemy.orm import declarative_base
from sqlalchemy.dialects.mysql import insert as mysql_insert
import datetime

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configuration
API_KEY = os.getenv("API_KEY", "secure-api-key-change-this-in-production")
FMCSA_API_KEY = os.getenv("FMCSA_API_KEY", "")
HAPPYROBOT_API_KEY = os.getenv("HAPPYROBOT_API_KEY", "")
HAPPYROBOT_BASE_URL = os.getenv("HAPPYROBOT_BASE_URL", "https://platform.happyrobot.ai/api/v1")
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
DEBUG = os.getenv("DEBUG", "true").lower() == "true"

# Global variable to track database availability
DATABASE_AVAILABLE = False

# Fixed MySQL Configuration - Use environment variables consistently
MYSQL_CONFIG = {
    "host": os.getenv("DB_HOST", "carrier-sales-db.internal"),
    "user": os.getenv("DB_USER", "carrier_user"), 
    "password": os.getenv("DB_PASS", "supersecret"),
    "database": os.getenv("DB_NAME", "carrier_db"),
    "port": int(os.getenv("DB_PORT", "3306")),
    "charset": 'utf8mb4',
    "use_unicode": True,
    "autocommit": True
}

# Warning logs
if API_KEY == "secure-api-key-change-this-in-production":
    logger.warning("Using default API key! Change this in production!")

if not FMCSA_API_KEY:
    logger.warning("No FMCSA API key provided! Using professional mock data only.")
else:
    logger.info(f"FMCSA API key configured: {FMCSA_API_KEY[:10]}...")

if not HAPPYROBOT_API_KEY:
    logger.warning("No HappyRobot API key provided! Using mock data for dashboard.")
else:
    logger.info(f"HappyRobot API key configured: {HAPPYROBOT_API_KEY[:10]}...")
    logger.info(f"HappyRobot Base URL: {HAPPYROBOT_BASE_URL}")

# Database setup
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    # Construct from individual components if not provided
    DATABASE_URL = f"mysql+aiomysql://{MYSQL_CONFIG['user']}:{MYSQL_CONFIG['password']}@{MYSQL_CONFIG['host']}:{MYSQL_CONFIG['port']}/{MYSQL_CONFIG['database']}"
    logger.info(f"Constructed DATABASE_URL: mysql+aiomysql://{MYSQL_CONFIG['user']}:***@{MYSQL_CONFIG['host']}:{MYSQL_CONFIG['port']}/{MYSQL_CONFIG['database']}")

engine = create_async_engine(
    DATABASE_URL, pool_recycle=1800, echo=False   # echo=True for SQL debug
)
AsyncSessionLocal = sessionmaker(
    engine, expire_on_commit=False, class_=AsyncSession
)

async def get_db() -> AsyncSession:          # FastAPI dependency
    async with AsyncSessionLocal() as session:
        yield session

Base = declarative_base()

class CallRecord(Base):
    __tablename__ = "calls"

    happyrobot_call_id = Column(String(64), primary_key=True)
    carrier_mc         = Column(String(32))
    company_name       = Column(String(128))
    call_transcript    = Column(Text)
    call_duration      = Column(Float)
    call_outcome       = Column(String(64))
    sentiment          = Column(String(32))
    equipment_type     = Column(String(64))
    original_rate      = Column(Float)
    proposed_rate      = Column(Float)
    final_outcome      = Column(String(64))
    call_status        = Column(String(32))
    created_at         = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at         = Column(DateTime, default=datetime.datetime.utcnow,
                                onupdate=datetime.datetime.utcnow)

# Enhanced database connection with retry logic
def get_db_connection_with_retry(max_retries=3, delay=5):
    """Get MySQL database connection with retry logic"""
    global DATABASE_AVAILABLE
    
    for attempt in range(max_retries):
        try:
            logger.info(f"Attempting database connection to {MYSQL_CONFIG['host']}:{MYSQL_CONFIG['port']} (attempt {attempt + 1}/{max_retries})")
            connection = mysql.connector.connect(**MYSQL_CONFIG)
            
            # Test the connection
            cursor = connection.cursor()
            cursor.execute("SELECT 1")
            cursor.fetchone()
            cursor.close()
            
            DATABASE_AVAILABLE = True
            logger.info(f"Database connection successful to {MYSQL_CONFIG['host']}")
            return connection
            
        except mysql.connector.Error as err:
            logger.error(f"Database connection attempt {attempt + 1} failed: {err}")
            if attempt < max_retries - 1:
                logger.info(f"Retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                logger.error(f"All {max_retries} database connection attempts failed")
                DATABASE_AVAILABLE = False
                if ENVIRONMENT == "production":
                    logger.info("Production mode: Continuing with mock data")
                return None
    
    return None

# Database connection functions
def get_db_connection():
    """Get MySQL database connection (wrapper)"""
    return get_db_connection_with_retry()

# Enhanced database initialization with proper async table creation
async def initialize_database_async():
    """Initialize database using async SQLAlchemy"""
    try:
        logger.info("Initializing database with async SQLAlchemy...")
        
        async with AsyncSessionLocal() as session:
            # Create all tables
            async with engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            
            # Check if we need to insert sample data
            result = await session.execute(text("SELECT COUNT(*) as count FROM calls"))
            call_count = result.scalar()
            
            if call_count == 0:
                logger.info("No existing call data found")
                # Sample data can be inserted here if needed
                
            await session.commit()
            logger.info("Async database initialization completed")
            return True
            
    except Exception as e:
        logger.error(f"Async database initialization failed: {e}")
        return False

def initialize_database():
    """Enhanced database initialization with both sync and async approaches"""
    global DATABASE_AVAILABLE
    
    # Try async approach for SQLAlchemy tables
    try:
        # Run async initialization - handle event loop properly
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.create_task(initialize_database_async())
            else:
                asyncio.run(initialize_database_async())
        except RuntimeError:
            asyncio.run(initialize_database_async())
        
        DATABASE_AVAILABLE = True
        logger.info("SQLAlchemy tables created successfully")
        
    except Exception as async_error:
        logger.error(f"Async database init failed: {async_error}")
    
    # Try sync approach for additional tables and sample data
    try:
        conn = get_db_connection_with_retry()
        if not conn:
            logger.info("Running in mock data mode")
            DATABASE_AVAILABLE = False
            return
            
        cursor = conn.cursor()
        
        # Create additional tables that might not be in SQLAlchemy models
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS loads (
                load_id VARCHAR(50) PRIMARY KEY,
                origin VARCHAR(255),
                destination VARCHAR(255),
                pickup_datetime DATETIME,
                delivery_datetime DATETIME,
                equipment_type VARCHAR(100),
                loadboard_rate DECIMAL(10,2),
                notes TEXT,
                weight DECIMAL(10,2),
                commodity_type VARCHAR(100),
                num_of_pieces INT,
                miles INT,
                dimensions VARCHAR(100),
                status ENUM('available', 'booked', 'expired') DEFAULT 'available',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                INDEX idx_status (status),
                INDEX idx_equipment (equipment_type)
            )
        """)
        
        # Create negotiations table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS negotiations (
                id INT AUTO_INCREMENT PRIMARY KEY,
                call_id VARCHAR(100),
                happyrobot_call_id VARCHAR(100),
                load_id VARCHAR(50),
                mc_number VARCHAR(20),
                proposed_rate DECIMAL(10,2),
                counter_offer DECIMAL(10,2),
                final_rate DECIMAL(10,2),
                round_number INT DEFAULT 1,
                status ENUM('pending', 'accepted', 'rejected', 'counter_offered') DEFAULT 'pending',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                INDEX idx_happyrobot_call_id (happyrobot_call_id),
                INDEX idx_load_id (load_id),
                INDEX idx_mc_number (mc_number),
                INDEX idx_status (status)
            )
        """)
        
        # Create metrics snapshots table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS metrics_snapshots (
                id INT AUTO_INCREMENT PRIMARY KEY,
                metric_date DATE UNIQUE,
                total_calls INT DEFAULT 0,
                conversion_rate DECIMAL(5,2) DEFAULT 0.00,
                average_negotiated_rate DECIMAL(10,2) DEFAULT 0.00,
                equipment_stats JSON,
                sentiment_stats JSON,
                outcome_stats JSON,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                INDEX idx_metric_date (metric_date)
            )
        """)
        
        conn.commit()
        logger.info("Additional MySQL tables created successfully")
        
        # Insert sample loads if table is empty
        cursor.execute("SELECT COUNT(*) FROM loads")
        if cursor.fetchone()[0] == 0:
            sample_loads = [
                ('LD001', 'Chicago, IL', 'Atlanta, GA', '2025-07-22 08:00:00', '2025-07-24 17:00:00', 'Dry Van', 2500.00, 'Urgent delivery', 45000, 'General Freight', 1, 720, '53ft', 'available'),
                ('LD002', 'Los Angeles, CA', 'Phoenix, AZ', '2025-07-23 09:00:00', '2025-07-24 15:00:00', 'Flatbed', 1800.00, 'Construction materials', 48000, 'Steel', 1, 380, '48ft', 'available'),
                ('LD003', 'Dallas, TX', 'Denver, CO', '2025-07-24 06:00:00', '2025-07-26 18:00:00', 'Refrigerated', 2200.00, 'Temperature controlled', 42000, 'Food Products', 1, 780, '53ft reefer', 'available'),
                ('LD004', 'Miami, FL', 'Jacksonville, FL', '2025-07-22 10:00:00', '2025-07-23 14:00:00', 'Dry Van', 800.00, 'Regional delivery', 38000, 'Retail Goods', 1, 350, '48ft', 'available'),
                ('LD005', 'Seattle, WA', 'Portland, OR', '2025-07-25 07:00:00', '2025-07-26 12:00:00', 'Flatbed', 600.00, 'Short haul', 50000, 'Lumber', 1, 180, '48ft flatbed', 'available')
            ]
            
            cursor.executemany("""
                INSERT INTO loads (load_id, origin, destination, pickup_datetime, 
                                 delivery_datetime, equipment_type, loadboard_rate, notes, weight, 
                                 commodity_type, num_of_pieces, miles, dimensions, status)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, sample_loads)
            
            conn.commit()
            logger.info("Sample loads inserted successfully")
        
        cursor.close()
        conn.close()
        DATABASE_AVAILABLE = True
        
    except Exception as err:
        logger.warning(f"Sync database initialization failed: {err}")
        logger.info("Running in mock data mode")
        DATABASE_AVAILABLE = False

# Enhanced data extraction functions
def extract_mc_number(text: str) -> Optional[str]:
    """Extract MC number from text"""
    if not text:
        return None
    
    # Look for MC number patterns
    patterns = [
        r'mc\s*(?:number\s*)?(\d{4,8})',
        r'motor\s*carrier\s*(?:number\s*)?(\d{4,8})',
        r'\bmc\s*(\d{4,8})\b'
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, text.lower())
        if matches:
            return matches[0]
    
    return None

def extract_company_name(text: str) -> Optional[str]:
    """Extract company name from text"""
    if not text:
        return None
    
    # Patterns to match company names
    company_patterns = [
        r'this is ([A-Za-z0-9\s&\-\',\.]+)(?:\s+(?:trucking|transport|logistics|freight|inc|llc|corp|company))',
        r'calling from ([A-Za-z0-9\s&\-\',\.]+)(?:\s+(?:trucking|transport|logistics|freight))?',
        r'my company is ([A-Za-z0-9\s&\-\',\.]+)',
        r'we are ([A-Za-z0-9\s&\-\',\.]+)(?:\s+(?:trucking|transport))?',
        r'(?:i\'?m|this is|calling from)\s+([A-Za-z0-9\s&\-\',\.]+)(?:\s+(?:trucking|transport|logistics|freight))',
        r'([A-Za-z0-9\s&\-\',\.]+)\s+(?:trucking|transport|logistics|freight)'
    ]
    
    for pattern in company_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            company = matches[0].strip()
            # Clean up the company name
            company = re.sub(r'\s+', ' ', company)  # Multiple spaces to single
            if len(company) > 2 and company.lower() not in ['we', 'my', 'this', 'calling']:
                return company.title()
    
    return None

def extract_equipment_type(text: str) -> Optional[str]:
    """Extract equipment type from text"""
    if not text:
        return None
    
    equipment_keywords = {
        'dry van': ['dry van', 'dry vans', 'van', 'vans'],
        'flatbed': ['flatbed', 'flatbeds', 'flat bed', 'flat beds'],
        'refrigerated': ['refrigerated', 'reefer', 'reefers', 'refrig', 'temp controlled'],
        'step deck': ['step deck', 'stepdeck', 'step decks', 'lowboy'],
        'box truck': ['box truck', 'box trucks', 'straight truck'],
        'tanker': ['tanker', 'tankers', 'tank']
    }
    
    text_lower = text.lower()
    for equipment, keywords in equipment_keywords.items():
        if any(keyword in text_lower for keyword in keywords):
            return equipment.title()
    
    return None

def extract_load_ids(text: str) -> List[str]:
    """Extract load IDs from text"""
    if not text:
        return []
    
    # Look for load ID patterns
    patterns = [
        r'\bld\s*(\d{3,6})\b',
        r'\bload\s*(?:id\s*)?(\d{3,6})\b',
        r'\b(ld\d{3,6})\b'
    ]
    
    load_ids = []
    for pattern in patterns:
        matches = re.findall(pattern, text.lower())
        for match in matches:
            if match.startswith('ld'):
                load_ids.append(match.upper())
            else:
                load_ids.append(f"LD{match}")
    
    return list(set(load_ids))  # Remove duplicates

def extract_rates(text: str) -> Dict[str, Optional[float]]:
    """FIXED: Enhanced rate extraction from text with better patterns"""
    if not text:
        return {"original_rate": None, "proposed_rate": None}
    
    # More comprehensive rate patterns
    rate_patterns = [
        r'\$(\d{1,2},?\d{3,4})',  # $2500, $2,500
        r'(\d{1,2},?\d{3,4})\s*(?:dollars?|bucks?|per\s*load)',
        r'(?:rate|price|offer|pay|paying)\s*(?:of\s*|is\s*|at\s*)?\$?(\d{1,2},?\d{3,4})',
        r'(\d{1,2},?\d{3,4})\s*(?:for\s*(?:this\s*)?(?:load|haul|trip))',
        r'(?:can\s*(?:you\s*)?(?:do|pay|offer))\s*\$?(\d{1,2},?\d{3,4})',
        r'(?:how\s*about|what\s*about)\s*\$?(\d{1,2},?\d{3,4})',
        r'(?:i\s*(?:can|will)\s*(?:do|pay|take))\s*\$?(\d{1,2},?\d{3,4})',
        r'(\d{1,2},?\d{3,4})\s*(?:sound|work|good)'
    ]
    
    rates = []
    text_lower = text.lower()
    
    for pattern in rate_patterns:
        matches = re.findall(pattern, text_lower)
        for match in matches:
            try:
                # Clean the rate string and convert to float
                clean_rate = match.replace(',', '')
                rate = float(clean_rate)
                if 500 <= rate <= 15000:  # Reasonable rate range (expanded)
                    rates.append(rate)
            except (ValueError, TypeError):
                continue
    
    # Remove duplicates and sort
    rates = sorted(list(set(rates)))
    
    result = {"original_rate": None, "proposed_rate": None}
    
    # Strategy: Look for context clues to determine which rate is which
    if len(rates) >= 2:
        # If multiple rates, try to determine which is original vs proposed
        # Look for context words around each rate
        high_rate = max(rates)
        low_rate = min(rates)
        
        # Check if there are indicators of negotiation
        if any(word in text_lower for word in ['counter', 'how about', 'can you do', 'negotiate']):
            result["original_rate"] = high_rate  # Original (higher) rate
            result["proposed_rate"] = low_rate   # Proposed (lower) rate
        else:
            result["original_rate"] = rates[-1]  # Last mentioned
            result["proposed_rate"] = rates[-2]  # Second to last
    elif len(rates) == 1:
        # Single rate - determine if it's original or proposed based on context
        rate = rates[0]
        if any(word in text_lower for word in ['offering', 'posted', 'loadboard', 'paying']):
            result["original_rate"] = rate
        else:
            result["proposed_rate"] = rate
    
    return result

def determine_final_outcome(text: str) -> str:
    """Determine final outcome from text"""
    if not text:
        return "inquiry_only"
    
    text_lower = text.lower()
    
    # Check for different outcomes
    if any(phrase in text_lower for phrase in [
        "book it", "we'll take it", "sounds good", "deal", 
        "transfer to sales", "hand over to", "connect me"
    ]):
        return "transferred to sales"
    elif any(phrase in text_lower for phrase in [
        "not interested", "too low", "pass", "no thanks"
    ]):
        return "declined"
    elif any(phrase in text_lower for phrase in [
        "negotiate", "counter", "how about", "can you do"
    ]):
        return "negotiating"
    elif any(phrase in text_lower for phrase in [
        "call back", "follow up", "let me check", "think about"
    ]):
        return "follow up required"
    else:
        return "inquiry_only"

def enhanced_extract_carrier_info_from_text(text: str) -> Dict:
    """Enhanced carrier information extraction"""
    if not text or text.strip() == "[]":
        # Return demo data for empty transcripts
        return {
            "mc_number": "123456",
            "company_name": "ABC Transportation LLC",
            "equipment_type": "Dry Van",
            "load_ids_discussed": ["LD001"],
            "original_rate": 2500.00,
            "proposed_rate": 2300.00,
            "final_outcome": "transferred to sales"
        }
    
    extracted_info = {
        "mc_number": extract_mc_number(text),
        "company_name": extract_company_name(text),
        "equipment_type": extract_equipment_type(text),
        "load_ids_discussed": extract_load_ids(text),
        "final_outcome": determine_final_outcome(text)
    }
    
    rates = extract_rates(text)
    extracted_info.update(rates)
    
    return extracted_info

def enhanced_classify_call_outcome(transcript: str) -> Dict:
    """Enhanced call outcome classification"""
    if not transcript or transcript.strip() == "[]":
        # Return demo classification for empty transcripts
        return {
            "response_classification": "inquiry_only",
            "response_reason": "Demo classification - carrier inquired about available loads"
        }
    
    transcript_lower = transcript.lower()
    
    # Classification logic
    if any(word in transcript_lower for word in ["book it", "take it", "sounds good", "deal", "we'll take", "booked"]):
        return {
            "response_classification": "load_booked",
            "response_reason": "Carrier accepted and booked a load during the call"
        }
    elif any(word in transcript_lower for word in ["negotiate", "counter", "how about", "can you do", "rate"]):
        return {
            "response_classification": "negotiation",
            "response_reason": "Active rate negotiation detected in the conversation"
        }
    elif any(word in transcript_lower for word in ["not interested", "pass", "no thanks", "too low", "can't do"]):
        return {
            "response_classification": "not_interested",
            "response_reason": "Carrier declined the offered loads or rates"
        }
    elif any(word in transcript_lower for word in ["transfer", "sales", "hand over", "connect me"]):
        return {
            "response_classification": "transferred",
            "response_reason": "Call was transferred to a sales representative"
        }
    else:
        return {
            "response_classification": "inquiry_only",
            "response_reason": "Carrier was gathering information about available loads"
        }

def enhanced_analyze_sentiment(transcript: str) -> str:
    """Enhanced sentiment analysis"""
    if not transcript or transcript.strip() == "[]":
        return "positive"  # Demo positive sentiment
    
    transcript_lower = transcript.lower()
    
    positive_indicators = [
        "interested", "good", "great", "excellent", "perfect", "sounds good", 
        "yes", "fantastic", "awesome", "love it", "definitely", "thank you"
    ]
    
    negative_indicators = [
        "not interested", "too low", "can't do", "no way", "pass", "no", 
        "terrible", "awful", "frustrated", "annoyed", "disappointed"
    ]
    
    neutral_indicators = [
        "maybe", "possibly", "let me think", "not sure", "okay", "alright"
    ]
    
    positive_count = sum(1 for word in positive_indicators if word in transcript_lower)
    negative_count = sum(1 for word in negative_indicators if word in transcript_lower)
    neutral_count = sum(1 for word in neutral_indicators if word in transcript_lower)
    
    if positive_count > negative_count and positive_count > neutral_count:
        return "positive"
    elif negative_count > positive_count and negative_count > neutral_count:
        return "negative"
    else:
        return "neutral"

# Simplified HappyRobot Integration (Webhook-based, no API polling needed)
def validate_happyrobot_config() -> Dict:
    """Simple validation of HappyRobot configuration"""
    if not HAPPYROBOT_API_KEY:
        return {
            "configured": False,
            "status": "missing_api_key",
            "message": "HappyRobot API key not configured"
        }
    
    return {
        "configured": True,
        "status": "webhook_ready", 
        "message": "Ready to receive HappyRobot webhooks",
        "api_key_preview": f"{HAPPYROBOT_API_KEY[:10]}...",
        "webhook_endpoints": [
            "/verify-carrier",
            "/search-loads", 
            "/negotiate-rate",
            "/extract-call-data",
            "/classify-call"
        ]
    }

# Database functions for real-time metrics
async def store_call_data_in_database(call_data: Dict):
    """Store webhook call data in database for real-time metrics"""
    if not DATABASE_AVAILABLE:
        return
    
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO calls 
            (happyrobot_call_id, carrier_mc, company_name, call_transcript, 
             call_outcome, sentiment, equipment_type, original_rate, proposed_rate, final_outcome)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
            call_transcript = VALUES(call_transcript),
            call_outcome = VALUES(call_outcome),
            sentiment = VALUES(sentiment),
            equipment_type = VALUES(equipment_type),
            original_rate = VALUES(original_rate),
            proposed_rate = VALUES(proposed_rate),
            final_outcome = VALUES(final_outcome),
            updated_at = CURRENT_TIMESTAMP
        """, (
            call_data.get("call_id", f"WEBHOOK_{int(datetime.now().timestamp())}"),
            call_data.get("mc_number"),
            call_data.get("company_name"),
            call_data.get("call_transcript", ""),
            call_data.get("classification", "inquiry_only"),
            call_data.get("sentiment", "neutral"),
            call_data.get("equipment_type"),
            call_data.get("original_rate"),
            call_data.get("proposed_rate"),
            call_data.get("final_outcome")
        ))
        
        conn.commit()
        cursor.close()
        conn.close()
        
        logger.info("Webhook call data stored in database for real-time metrics")
        
    except Exception as e:
        logger.error(f"Error storing webhook call data: {e}")

# Global variables for in-memory webhook storage
webhook_events: List[Dict] = []   # in-memory fallback store

async def get_real_time_metrics():
    """FIXED: Get real-time metrics - ONLY count webhook calls, not all database calls"""
    
    # Count in-memory webhook events
    memory_webhook_count = len(webhook_events)
    
    if not DATABASE_AVAILABLE:
        if memory_webhook_count > 0:
            # Use in-memory webhook data
            return await get_webhook_metrics_from_memory()
        else:
            return await get_enhanced_fallback_metrics()
    
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        
        # Get date ranges
        today = datetime.datetime.now().date()
        week_ago = today - timedelta(days=7)
        
        # FIXED: Only count webhook calls (not all database calls)
        cursor.execute("""
            SELECT COUNT(*) as count FROM calls 
            WHERE (happyrobot_call_id LIKE 'HR_%' 
                OR happyrobot_call_id LIKE 'TEST_%' 
                OR happyrobot_call_id LIKE 'WEBHOOK_%'
                OR happyrobot_call_id LIKE 'DEBUG_%')
            AND created_at >= %s
        """, (week_ago,))
        result = cursor.fetchone()
        db_webhook_count = result['count'] if result else 0
        
        # Total webhook calls = database + memory
        total_webhook_calls = db_webhook_count + memory_webhook_count
        
        logger.info(f"Webhook count: DB={db_webhook_count}, Memory={memory_webhook_count}, Total={total_webhook_calls}")
        
        # If no webhook calls at all, return mock data
        if total_webhook_calls == 0:
            logger.info("No webhook calls found, using mock data")
            return await get_enhanced_fallback_metrics()
        
        # Recent webhook calls (last 7 days)
        cursor.execute("""
            SELECT COUNT(*) as count FROM calls 
            WHERE (happyrobot_call_id LIKE 'HR_%' 
                OR happyrobot_call_id LIKE 'TEST_%' 
                OR happyrobot_call_id LIKE 'WEBHOOK_%'
                OR happyrobot_call_id LIKE 'DEBUG_%')
            AND DATE(created_at) >= %s
        """, (week_ago,))
        result = cursor.fetchone()
        db_recent_webhook_calls = result['count'] if result else 0
        recent_webhook_calls = db_recent_webhook_calls + memory_webhook_count
        
        # Conversion rate (booked webhook calls only)
        cursor.execute("""
            SELECT COUNT(*) as count FROM calls 
            WHERE (call_outcome = 'load_booked' 
                OR call_outcome = 'transferred to sales'
                OR final_outcome = 'transferred to sales')
            AND (happyrobot_call_id LIKE 'HR_%' 
                OR happyrobot_call_id LIKE 'TEST_%' 
                OR happyrobot_call_id LIKE 'WEBHOOK_%'
                OR happyrobot_call_id LIKE 'DEBUG_%')
            AND created_at >= %s
        """, (week_ago,))
        result = cursor.fetchone()
        db_booked_webhook_calls = result['count'] if result else 0
        
        # Add in-memory booked webhook calls
        memory_booked = len([e for e in webhook_events if e.get('final_outcome') in ['transferred to sales', 'load_booked']])
        total_booked_webhook_calls = db_booked_webhook_calls + memory_booked
        
        conversion_rate = (total_booked_webhook_calls / total_webhook_calls * 100) if total_webhook_calls > 0 else 0
        
        # Average negotiated rate from webhook calls only
        cursor.execute("""
            SELECT AVG(proposed_rate) as avg_rate FROM calls 
            WHERE proposed_rate > 0 
            AND (happyrobot_call_id LIKE 'HR_%' 
                OR happyrobot_call_id LIKE 'TEST_%' 
                OR happyrobot_call_id LIKE 'WEBHOOK_%'
                OR happyrobot_call_id LIKE 'DEBUG_%')
            AND created_at >= %s
        """, (week_ago,))
        result = cursor.fetchone()
        avg_rate = result['avg_rate'] if result and result['avg_rate'] else 0
        
        # Add in-memory webhook rates
        memory_rates = [e.get('proposed_rate', 0) for e in webhook_events if e.get('proposed_rate', 0) > 0]
        if memory_rates:
            all_rates = ([avg_rate] if avg_rate else []) + memory_rates
            avg_rate = sum(all_rates) / len(all_rates)
        
        if avg_rate == 0:
            avg_rate = 2650  # Default fallback
        
        # FIXED: Data source determination based on webhook calls only
        data_source = "enhanced_mock_data"  # Default
        if total_webhook_calls > 0:
            if db_webhook_count > 0:
                data_source = "webhook_database_real_time"
            elif memory_webhook_count > 0:
                data_source = "webhook_memory_real_time"
        
        # Classifications from webhook calls only
        cursor.execute("""
            SELECT 
                COALESCE(call_outcome, final_outcome, 'inquiry_only') as type, 
                COUNT(*) as count 
            FROM calls 
            WHERE (happyrobot_call_id LIKE 'HR_%' 
                OR happyrobot_call_id LIKE 'TEST_%' 
                OR happyrobot_call_id LIKE 'WEBHOOK_%'
                OR happyrobot_call_id LIKE 'DEBUG_%')
            AND created_at >= %s
            GROUP BY COALESCE(call_outcome, final_outcome, 'inquiry_only')
        """, (week_ago,))
        db_classifications = {row['type']: row['count'] for row in cursor.fetchall()}
        
        # Add in-memory classifications
        memory_classifications = {}
        for event in webhook_events:
            outcome = event.get('final_outcome', 'inquiry_only')
            memory_classifications[outcome] = memory_classifications.get(outcome, 0) + 1
        
        # Merge classifications
        all_classifications = {}
        all_outcomes = set(list(db_classifications.keys()) + list(memory_classifications.keys()))
        for outcome in all_outcomes:
            all_classifications[outcome] = db_classifications.get(outcome, 0) + memory_classifications.get(outcome, 0)
        
        classifications = [{"type": k, "count": v} for k, v in all_classifications.items()]
        
        # Sentiments from webhook calls only
        cursor.execute("""
            SELECT sentiment as type, COUNT(*) as count 
            FROM calls 
            WHERE sentiment IS NOT NULL 
            AND (happyrobot_call_id LIKE 'HR_%' 
                OR happyrobot_call_id LIKE 'TEST_%' 
                OR happyrobot_call_id LIKE 'WEBHOOK_%'
                OR happyrobot_call_id LIKE 'DEBUG_%')
            AND created_at >= %s
            GROUP BY sentiment
        """, (week_ago,))
        db_sentiments = {row['type']: row['count'] for row in cursor.fetchall()}
        
        memory_sentiments = {}
        for event in webhook_events:
            sentiment = event.get('sentiment', 'neutral')
            memory_sentiments[sentiment] = memory_sentiments.get(sentiment, 0) + 1
        
        all_sentiments = {}
        all_sentiment_types = set(list(db_sentiments.keys()) + list(memory_sentiments.keys()))
        for sentiment in all_sentiment_types:
            all_sentiments[sentiment] = db_sentiments.get(sentiment, 0) + memory_sentiments.get(sentiment, 0)
            
        sentiments = [{"type": k, "count": v} for k, v in all_sentiments.items()]
        
        # Equipment from webhook calls only
        cursor.execute("""
            SELECT equipment_type as type, COUNT(*) as calls 
            FROM calls 
            WHERE equipment_type IS NOT NULL 
            AND (happyrobot_call_id LIKE 'HR_%' 
                OR happyrobot_call_id LIKE 'TEST_%' 
                OR happyrobot_call_id LIKE 'WEBHOOK_%'
                OR happyrobot_call_id LIKE 'DEBUG_%')
            AND created_at >= %s
            GROUP BY equipment_type
        """, (week_ago,))
        db_equipment = {row['type']: row['calls'] for row in cursor.fetchall()}
        
        memory_equipment = {}
        for event in webhook_events:
            equipment = event.get('equipment_type')
            if equipment:
                memory_equipment[equipment] = memory_equipment.get(equipment, 0) + 1
        
        all_equipment = {}
        all_equipment_types = set(list(db_equipment.keys()) + list(memory_equipment.keys()))
        for equip in all_equipment_types:
            all_equipment[equip] = db_equipment.get(equip, 0) + memory_equipment.get(equip, 0)
            
        equipment_performance = [{"type": k, "calls": v} for k, v in all_equipment.items()]
        
        cursor.close()
        conn.close()
        
        # Use minimal fallback data if categories are empty
        if not classifications:
            classifications = [
                {"type": "inquiry_only", "count": max(1, int(total_webhook_calls * 0.5))},
                {"type": "negotiation", "count": max(1, int(total_webhook_calls * 0.3))},
                {"type": "load_booked", "count": max(0, int(total_webhook_calls * 0.2))}
            ]
        
        if not sentiments:
            sentiments = [
                {"type": "positive", "count": max(1, int(total_webhook_calls * 0.6))},
                {"type": "neutral", "count": max(1, int(total_webhook_calls * 0.3))},
                {"type": "negative", "count": max(0, int(total_webhook_calls * 0.1))}
            ]
        
        if not equipment_performance:
            equipment_performance = [
                {"type": "Dry Van", "calls": max(1, int(total_webhook_calls * 0.5))},
                {"type": "Flatbed", "calls": max(0, int(total_webhook_calls * 0.3))},
                {"type": "Other", "calls": max(0, int(total_webhook_calls * 0.2))}
            ]
        
        metrics = {
            "summary": {
                "total_calls": total_webhook_calls,
                "recent_calls": recent_webhook_calls,
                "conversion_rate": round(conversion_rate, 1),
                "average_negotiated_rate": int(avg_rate)
            },
            "classifications": classifications,
            "sentiments": sentiments,
            "equipment_performance": equipment_performance,
            "recent_activity": [],
            "updated_at": datetime.datetime.now(timezone.utc).isoformat(),
            "data_source": data_source
        }
        
        logger.info(f"Webhook metrics: {total_webhook_calls} calls, source: {data_source}")
        return metrics
        
    except Exception as e:
        logger.error(f"Error calculating webhook metrics: {e}")
        if memory_webhook_count > 0:
            return await get_webhook_metrics_from_memory()
        return await get_enhanced_fallback_metrics()

async def get_webhook_metrics_from_memory():
    """Get metrics from in-memory webhook events only"""
    memory_webhook_count = len(webhook_events)
    
    if memory_webhook_count == 0:
        return await get_enhanced_fallback_metrics()
    
    # Calculate metrics from in-memory webhook data only
    memory_booked = len([e for e in webhook_events if e.get('final_outcome') in ['transferred to sales', 'load_booked']])
    conversion_rate = (memory_booked / memory_webhook_count * 100) if memory_webhook_count > 0 else 0
    
    memory_rates = [e.get('proposed_rate', 0) for e in webhook_events if e.get('proposed_rate', 0) > 0]
    avg_rate = sum(memory_rates) / len(memory_rates) if memory_rates else 2650
    
    # Classifications from memory
    memory_classifications = {}
    for event in webhook_events:
        outcome = event.get('final_outcome', 'inquiry_only')
        memory_classifications[outcome] = memory_classifications.get(outcome, 0) + 1
    classifications = [{"type": k, "count": v} for k, v in memory_classifications.items()]
    
    # Sentiments from memory
    memory_sentiments = {}
    for event in webhook_events:
        sentiment = event.get('sentiment', 'neutral')
        memory_sentiments[sentiment] = memory_sentiments.get(sentiment, 0) + 1
    sentiments = [{"type": k, "count": v} for k, v in memory_sentiments.items()]
    
    # Equipment from memory
    memory_equipment = {}
    for event in webhook_events:
        equipment = event.get('equipment_type')
        if equipment:
            memory_equipment[equipment] = memory_equipment.get(equipment, 0) + 1
    equipment_performance = [{"type": k, "calls": v} for k, v in memory_equipment.items()]
    
    return {
        "summary": {
            "total_calls": memory_webhook_count,
            "recent_calls": memory_webhook_count,
            "conversion_rate": round(conversion_rate, 1),
            "average_negotiated_rate": int(avg_rate)
        },
        "classifications": classifications,
        "sentiments": sentiments,
        "equipment_performance": equipment_performance,
        "recent_activity": [],
        "updated_at": datetime.datetime.now(timezone.utc).isoformat(),
        "data_source": "webhook_memory_real_time"
    }

async def get_enhanced_fallback_metrics():
    """Enhanced fallback metrics with realistic data"""
    from random import randint, uniform
    
    # More realistic numbers based on actual call center performance
    total_calls = randint(20, 45)
    recent_calls = randint(6, 15)
    conversion_rate = round(uniform(12.5, 18.7), 1)
    avg_rate = randint(2400, 2950)
    
    metrics = {
        "summary": {
            "total_calls": total_calls,
            "recent_calls": recent_calls,
            "conversion_rate": conversion_rate,
            "average_negotiated_rate": avg_rate
        },
        "classifications": [
            {"type": "inquiry_only", "count": randint(8, 18)},
            {"type": "negotiation", "count": randint(4, 12)},
            {"type": "not_interested", "count": randint(1, 6)},
            {"type": "load_booked", "count": randint(2, 8)},
            {"type": "transferred", "count": randint(1, 4)}
        ],
        "sentiments": [
            {"type": "positive", "count": randint(12, 25)},
            {"type": "neutral", "count": randint(6, 15)},
            {"type": "negative", "count": randint(1, 8)}
        ],
        "equipment_performance": [
            {"type": "Dry Van", "calls": randint(8, 22)},
            {"type": "Flatbed", "calls": randint(4, 16)},
            {"type": "Refrigerated", "calls": randint(3, 12)},
            {"type": "Step Deck", "calls": randint(1, 8)}
        ],
        "recent_activity": [],
        "updated_at": datetime.datetime.now(timezone.utc).isoformat(),
        "data_source": "enhanced_mock_data"
    }
    
    return metrics

# Mock Data
MOCK_CARRIER_DATA = {
    "123456": {
        "mc_number": "123456",
        "company_name": "ABC Transportation LLC",
        "address": "123 Trucking Way, Transport City, TX 75001",
        "phone": "(555) 123-4567",
        "email": "dispatch@abctransport.com",
        "dot_number": "1234567",
        "equipment_types": ["Dry Van", "Refrigerated"],
        "fleet_size": 25,
        "drivers": 30,
        "safety_rating": "Satisfactory",
        "insurance_status": "Active",
        "authority_status": "Active",
        "years_in_business": 8,
        "preferred_lanes": ["TX-CA", "TX-FL", "IL-TX"],
        "average_rate_per_mile": 2.85,
        "last_used": "2025-07-15",
        "out_of_service": "",
        "data_source": "professional_mock"
    },
    "789012": {
        "mc_number": "789012",
        "company_name": "Express Freight Solutions",
        "address": "456 Highway Blvd, Logistics Park, CA 90210",
        "phone": "(555) 987-6543",
        "email": "ops@expressfreight.com",
        "dot_number": "7890123",
        "equipment_types": ["Flatbed", "Step Deck"],
        "fleet_size": 40,
        "drivers": 45,
        "safety_rating": "Satisfactory",
        "insurance_status": "Active",
        "authority_status": "Active",
        "years_in_business": 12,
        "preferred_lanes": ["CA-TX", "CA-AZ", "NV-CA"],
        "average_rate_per_mile": 3.20,
        "last_used": "2025-07-18",
        "out_of_service": "",
        "data_source": "professional_mock"
    }
}

# Mock load data
MOCK_LOAD_DATA = [
    {
        "load_id": "LD001",
        "origin": "Chicago, IL",
        "destination": "Atlanta, GA",
        "pickup_datetime": "2025-07-22 08:00:00",
        "delivery_datetime": "2025-07-24 17:00:00",
        "equipment_type": "Dry Van",
        "loadboard_rate": 2500.00,
        "notes": "Urgent delivery",
        "weight": 45000,
        "commodity_type": "General Freight",
        "num_of_pieces": 1,
        "miles": 720,
        "dimensions": "53ft",
        "status": "available",
        "created_at": "2025-07-20T12:57:14",
        "updated_at": "2025-07-20T12:57:14",
        "id": "LD001",
        "pickup_location": "Chicago, IL",
        "delivery_location": "Atlanta, GA",
        "rate": 2500.00,
        "pickup_date": "2025-07-22",
        "delivery_date": "2025-07-24"
    },
    {
        "load_id": "LD002",
        "origin": "Los Angeles, CA",
        "destination": "Phoenix, AZ",
        "pickup_datetime": "2025-07-23 09:00:00",
        "delivery_datetime": "2025-07-24 15:00:00",
        "equipment_type": "Flatbed",
        "loadboard_rate": 1800.00,
        "notes": "Construction materials",
        "weight": 48000,
        "commodity_type": "Steel",
        "num_of_pieces": 1,
        "miles": 380,
        "dimensions": "48ft",
        "status": "available",
        "created_at": "2025-07-20T12:57:14",
        "updated_at": "2025-07-20T12:57:14",
        "id": "LD002",
        "pickup_location": "Los Angeles, CA",
        "delivery_location": "Phoenix, AZ",
        "rate": 1800.00,
        "pickup_date": "2025-07-23",
        "delivery_date": "2025-07-24"
    }
]

# FMCSA API Integration
async def query_fmcsa_api(mc_number: str, api_key: str):
    """Query the real FMCSA API for carrier information"""
    if not api_key:
        logger.warning("No FMCSA API key available")
        return None
    
    try:
        url = f"https://mobile.fmcsa.dot.gov/qc/services/carriers/docket-number/{mc_number}"
        params = {"webKey": api_key}
        
        logger.info(f"Querying FMCSA API for MC {mc_number}")
        
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(url, params=params)
            
            if response.status_code == 200:
                try:
                    data = response.json()
                    logger.info(f"FMCSA API returned data for MC {mc_number}")
                    
                    if isinstance(data, list) and len(data) > 0:
                        data = data[0]
                    
                    if data and isinstance(data, dict):
                        carrier_data = {
                            "mc_number": mc_number,
                            "company_name": data.get("legalName", "Unknown"),
                            "address": f"{data.get('phyStreet', '')} {data.get('phyCity', '')} {data.get('phyState', '')} {data.get('phyZipcode', '')}".strip(),
                            "phone": data.get("telephone", ""),
                            "email": data.get("emailAddress", ""),
                            "dot_number": str(data.get("dotNumber", "")),
                            "safety_rating": data.get("safetyRating", "Not Rated"),
                            "authority_status": "Active" if data.get("allowToOperate") == "Y" else "Inactive",
                            "insurance_status": "Active" if data.get("bippdInsuranceOnFile") == "Y" else "Unknown",
                            "out_of_service": data.get("outOfServiceDate", ""),
                            "fleet_size": int(data.get("totalPowerUnits", 0)),
                            "drivers": int(data.get("totalDrivers", 0)),
                            "data_source": "fmcsa_api",
                            "last_updated": data.get("recordStatus", ""),
                            "equipment_types": ["Various"],
                            "years_in_business": "Unknown",
                            "preferred_lanes": [],
                            "average_rate_per_mile": 0.0
                        }
                        return carrier_data
                        
                except Exception as parse_error:
                    logger.error(f"Error parsing FMCSA API response: {parse_error}")
                    
            else:
                logger.warning(f"FMCSA API returned status {response.status_code} for MC {mc_number}")
                    
            return None
            
    except Exception as e:
        logger.warning(f"FMCSA API query failed for MC {mc_number}: {e}")
        return None

# Pydantic Models
class LoadSearchRequest(BaseModel):
    equipment_type: str
    pickup_state: Optional[str] = None
    delivery_state: Optional[str] = None
    pickup_city: Optional[str] = None
    delivery_city: Optional[str] = None
    loadboard_rate: Optional[float] = None
    weight: Optional[int] = None
    
    @field_validator('loadboard_rate', 'weight')
    @classmethod
    def validate_positive_numbers(cls, v):
        if v is not None and v <= 0:
            raise ValueError('Must be positive')
        return v

class CarrierVerificationRequest(BaseModel):
    mc_number: str
    
    @field_validator('mc_number')
    @classmethod
    def validate_mc_number(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError('MC number is required')
        return v.strip()

class RateNegotiationRequest(BaseModel):
    load_id: str
    proposed_rate: float
    carrier_mc: Optional[str] = None
    mc_number: Optional[str] = None
    
    @field_validator('proposed_rate')
    @classmethod
    def validate_proposed_rate(cls, v):
        if v <= 0:
            raise ValueError('Proposed rate must be positive')
        return v
    
    def get_mc_number(self):
        return self.carrier_mc or self.mc_number or "unknown"

class CallDataExtractionRequest(BaseModel):
    call_transcript: str
    call_duration: Optional[int] = None

class CallClassificationRequest(BaseModel):
    call_transcript: str

# Enhanced lifespan event handler
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler with proper async database initialization"""
    # Startup
    logger.info("Starting Carrier Sales API with HappyRobot Integration...")
    
    # Initialize database
    initialize_database()
    
    # Also ensure async tables are created
    try:
        await initialize_database_async()
    except Exception as e:
        logger.warning(f"Async database init in lifespan failed: {e}")
    
    logger.info("Carrier Sales API started successfully")
    yield
    # Shutdown
    logger.info("Shutting down Carrier Sales API...")

# Initialize FastAPI app
app = FastAPI(
    title="Carrier Sales API with HappyRobot Integration",
    description="AI-powered freight broker assistant with real HappyRobot call data processing and FMCSA integration",
    version="1.3.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()

def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify API key authentication"""
    if credentials.credentials != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return credentials.credentials

# Middleware for request logging
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = datetime.datetime.now()
    logger.info(f"{request.method} {request.url.path} from {request.client.host}")
    response = await call_next(request)
    process_time = (datetime.datetime.now() - start_time).total_seconds()
    logger.info(f"{request.method} {request.url.path} -> {response.status_code} ({process_time:.3f}s)")
    return response

async def store_webhook_call_data(
    call_info: Dict,
    db: Optional[AsyncSession] = None,
) -> None:

    global webhook_events
    
    try:
        if db is None or not DATABASE_AVAILABLE:
            # Store in memory as fallback
            webhook_events.append(call_info)
            logger.info("Stored webhook in-memory (%d total)", len(webhook_events))
            return

        # Try to store in database using SQLAlchemy async
        stmt = mysql_insert(CallRecord).values(call_info)
        # Build ON DUPLICATE KEY UPDATE for every column except PK & created_at
        update_cols = {
            col.name: stmt.inserted[col.name]
            for col in CallRecord.__table__.c
            if col.name not in ("happyrobot_call_id", "created_at")
        }
        stmt = stmt.on_duplicate_key_update(**update_cols)

        await db.execute(stmt)
        await db.commit()
        logger.info("Stored webhook in MySQL: %s", call_info["happyrobot_call_id"])
        
        # Also add to memory for immediate dashboard updates
        webhook_events.append(call_info)
        
    except Exception as e:
        await db.rollback() if db else None
        logger.error("MySQL error, storing in memory only: %s", e)
        webhook_events.append(call_info)

def get_in_memory_events(limit: int = 50) -> List[Dict]:
    """Return the most-recent `limit` events kept only in RAM."""
    return webhook_events[-limit:][::-1]

# API Endpoints
@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Enhanced Carrier Sales API with HappyRobot Webhook Integration",
        "version": "1.3.0",
        "environment": ENVIRONMENT,
        "integration_approach": "webhook_based_real_time",
        "fmcsa_api_available": bool(FMCSA_API_KEY),
        "happyrobot_webhooks_ready": bool(HAPPYROBOT_API_KEY),
        "database_available": DATABASE_AVAILABLE,
        "documentation": "/docs",
        "health_check": "/health",
        "webhook_endpoints": [
            "/verify-carrier",
            "/search-loads", 
            "/negotiate-rate",
            "/extract-call-data",
            "/classify-call"
        ],
        "features": [
            "Real-time webhook processing",
            "Enhanced data extraction from transcripts", 
            "Intelligent call classification",
            "FMCSA carrier verification",
            "Live dashboard with webhook metrics",
            "No API polling required"
        ]
    }

@app.get("/health")
async def health_check():
    """Enhanced health check endpoint"""
    try:
        database_status = "connected" if DATABASE_AVAILABLE else "mock"
        
        if DATABASE_AVAILABLE:
            conn = get_db_connection()
            if conn:
                cursor = conn.cursor()
                cursor.execute("SELECT 1")
                cursor.fetchone()
                cursor.close()
                conn.close()
                database_status = "connected"
            else:
                database_status = "mock"
        
        # Check HappyRobot webhook readiness
        happyrobot_status = "webhook_ready" if HAPPYROBOT_API_KEY else "not_configured"
        
        return {
            "status": "healthy",
            "timestamp": datetime.datetime.now(timezone.utc).isoformat(),
            "environment": ENVIRONMENT,
            "services": {
                "database": database_status,
                "fmcsa_api": "configured" if FMCSA_API_KEY else "mock_only",
                "happyrobot_webhooks": happyrobot_status
            },
            "version": "1.3.0",
            "integration_approach": "webhook_based",
            "features": {
                "real_time_extraction": True,
                "enhanced_classification": True,
                "live_dashboard": True,
                "webhook_metrics": True,
                "fmcsa_integration": bool(FMCSA_API_KEY)
            }
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "degraded",
            "timestamp": datetime.datetime.now(timezone.utc).isoformat(),
            "environment": ENVIRONMENT,
            "error": str(e)
        }

@app.get("/dashboard-metrics")
async def get_dashboard_metrics():
    """Get real-time dashboard metrics"""
    try:
        logger.info("Calculating real-time dashboard metrics...")
        
        # Get real metrics
        metrics = await get_real_time_metrics()

        if not DATABASE_AVAILABLE and len(webhook_events) > 0:
            metrics["data_source"] = "webhook_memory_real_time"
            metrics["summary"]["total_calls"] = len(webhook_events)
            metrics["summary"]["recent_calls"] = len(webhook_events[-10:])
        
        return {
            "success": True,
            "metrics": metrics
        }
        
    except Exception as e:
        logger.error(f"Dashboard metrics error: {e}")
        # Return fallback metrics if everything fails
        fallback_metrics = await get_enhanced_fallback_metrics()
        return {
            "success": True,
            "metrics": fallback_metrics
        }

@app.post("/verify-carrier")
async def verify_carrier(request: CarrierVerificationRequest, api_key: str = Depends(verify_api_key)):
    """Enhanced carrier verification using real FMCSA API with fallback to mock data"""
    try:
        mc_number = request.mc_number.strip()
        
        # Try real FMCSA API first
        if FMCSA_API_KEY:
            logger.info(f"Attempting FMCSA API lookup for MC {mc_number}")
            fmcsa_data = await query_fmcsa_api(mc_number, FMCSA_API_KEY)
            
            if fmcsa_data:
                company_name = fmcsa_data.get("company_name", "").strip()
                authority_status = fmcsa_data.get("authority_status", "").strip()
                
                if (company_name not in ["Unknown", "", "N/A"] and 
                    authority_status not in ["Inactive", "", "N/A"]):
                    
                    logger.info(f"Successfully retrieved valid data from FMCSA API for MC {mc_number}")
                    fmcsa_data["verified_at"] = datetime.datetime.now(timezone.utc).isoformat()
                    
                    return {
                        "success": True,
                        "carrier": fmcsa_data,
                        "verification_status": "verified",
                        "data_source": "fmcsa_api"
                    }
        
        # Fallback to professional mock data
        logger.info(f"Using mock data for MC {mc_number}")
        if mc_number in MOCK_CARRIER_DATA:
            carrier_data = MOCK_CARRIER_DATA[mc_number].copy()
            carrier_data["verified_at"] = datetime.datetime.now(timezone.utc).isoformat()
            
            return {
                "success": True,
                "carrier": carrier_data,
                "verification_status": "verified",
                "data_source": "professional_mock"
            }
        else:
            return {
                "success": False,
                "carrier": None,
                "verification_status": "not_found",
                "message": f"No carrier found with MC number {mc_number}",
                "suggested_action": "Please verify the MC number and try again",
                "data_source": "not_found"
            }
            
    except Exception as e:
        logger.error(f"Carrier verification error: {e}")
        raise HTTPException(status_code=500, detail="Carrier verification failed")

@app.post("/search-loads")
async def search_loads(request: LoadSearchRequest, api_key: str = Depends(verify_api_key)):
    """Search for available loads based on criteria"""
    try:
        # Use mock data if database not available
        if not DATABASE_AVAILABLE:
            filtered_loads = []
            for load in MOCK_LOAD_DATA:
                if request.equipment_type and load["equipment_type"].lower() != request.equipment_type.lower():
                    continue
                if request.pickup_state and request.pickup_state.upper() not in load["origin"].upper():
                    continue
                if request.delivery_state and request.delivery_state.upper() not in load["destination"].upper():
                    continue
                if request.pickup_city and request.pickup_city.upper() not in load["origin"].upper():
                    continue
                if request.delivery_city and request.delivery_city.upper() not in load["destination"].upper():
                    continue
                if request.loadboard_rate and load["loadboard_rate"] < request.loadboard_rate:
                    continue
                if request.weight and load["weight"] > request.weight:
                    continue
                    
                filtered_loads.append(load)
            
            filtered_loads.sort(key=lambda x: x["loadboard_rate"], reverse=True)
            
            return {
                "success": True,
                "loads": filtered_loads[:20],
                "total_found": len(filtered_loads),
                "search_criteria": request.dict(),
                "data_source": "mock"
            }
        
        # Database query
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        
        query = "SELECT * FROM loads WHERE status = 'available'"
        params = []
        
        if request.equipment_type:
            query += " AND LOWER(equipment_type) = LOWER(%s)"
            params.append(request.equipment_type)
        if request.pickup_state:
            query += " AND UPPER(origin) LIKE UPPER(%s)"
            params.append(f"%{request.pickup_state}%")
        if request.delivery_state:
            query += " AND UPPER(destination) LIKE UPPER(%s)"
            params.append(f"%{request.delivery_state}%")
        if request.pickup_city:
            query += " AND UPPER(origin) LIKE UPPER(%s)"
            params.append(f"%{request.pickup_city}%")
        if request.delivery_city:
            query += " AND UPPER(destination) LIKE UPPER(%s)"
            params.append(f"%{request.delivery_city}%")
        if request.loadboard_rate:
            query += " AND loadboard_rate >= %s"
            params.append(request.loadboard_rate)
        if request.weight:
            query += " AND weight <= %s"
            params.append(request.weight)
            
        query += " ORDER BY loadboard_rate DESC LIMIT 20"
        
        cursor.execute(query, params)
        loads = cursor.fetchall()
        
        # Convert dates for JSON serialization
        for load in loads:
            if load.get('pickup_datetime'):
                load['pickup_datetime'] = load['pickup_datetime'].strftime('%Y-%m-%d %H:%M:%S')
            if load.get('delivery_datetime'):
                load['delivery_datetime'] = load['delivery_datetime'].strftime('%Y-%m-%d %H:%M:%S')
            if load.get('created_at'):
                load['created_at'] = load['created_at'].isoformat()
            if load.get('updated_at'):
                load['updated_at'] = load['updated_at'].isoformat()
            
            # Add aliases for compatibility
            load['id'] = load['load_id']
            load['pickup_location'] = load['origin']
            load['delivery_location'] = load['destination']
            load['rate'] = load['loadboard_rate']
            load['pickup_date'] = load['pickup_datetime'][:10] if load.get('pickup_datetime') else None
            load['delivery_date'] = load['delivery_datetime'][:10] if load.get('delivery_datetime') else None
        
        cursor.close()
        conn.close()
        
        return {
            "success": True,
            "loads": loads,
            "total_found": len(loads),
            "search_criteria": request.dict(),
            "data_source": "database"
        }
        
    except Exception as e:
        logger.error(f"Load search error: {e}")
        raise HTTPException(status_code=500, detail="Load search failed")

RATE_NEGOTIATION_CONFIG = {
    "max_above_posted": 1000,      # Don't accept more than $1000 above posted rate
    "max_below_posted": 1000,      # Don't accept more than $1000 below posted rate
    "auto_accept_range": 100,      # Auto-accept within $100 of posted rate
    "counter_offer_range": 500,    # Counter-offer if within $500 below posted
    "counter_offer_discount": 200, # Counter-offer at $200 below posted rate
    "final_offer_discount": 300    # Final offer at $300 below posted rate
}

@app.post("/negotiate-rate")
async def negotiate_rate(request: RateNegotiationRequest, api_key: str = Depends(verify_api_key)):
    """FIXED: Handle rate negotiation with strict dollar amount limits"""
    try:
        # Find load
        load = None
        
        if not DATABASE_AVAILABLE:
            for mock_load in MOCK_LOAD_DATA:
                if mock_load["load_id"] == request.load_id:
                    load = mock_load
                    break
        else:
            conn = get_db_connection()
            cursor = conn.cursor(dictionary=True)
            cursor.execute("SELECT * FROM loads WHERE load_id = %s", (request.load_id,))
            load = cursor.fetchone()
            cursor.close()
            conn.close()
        
        if not load:
            raise HTTPException(status_code=404, detail="Load not found")
        
        current_rate = float(load['loadboard_rate'])
        proposed_rate = request.proposed_rate
        
        max_acceptable_rate = current_rate + RATE_NEGOTIATION_CONFIG["max_above_posted"]
        min_acceptable_rate = current_rate - RATE_NEGOTIATION_CONFIG["max_below_posted"]
        
        logger.info(f"💰 Rate negotiation: Load {request.load_id}, Posted=${current_rate}, Proposed=${proposed_rate}, Range=${min_acceptable_rate}-${max_acceptable_rate}")
        
        # Check if proposed rate is completely outside acceptable range
        if proposed_rate > max_acceptable_rate:
            status = "rejected"
            counter_offer = current_rate
            response_message = f"Sorry, ${proposed_rate:.2f} is too high. Our posted rate for load {request.load_id} is ${current_rate:.2f}."
        elif proposed_rate < min_acceptable_rate:
            status = "rejected"
            counter_offer = min_acceptable_rate
            response_message = f"Sorry, ${proposed_rate:.2f} is too low. Our minimum rate for load {request.load_id} is ${min_acceptable_rate:.2f}."
        elif abs(proposed_rate - current_rate) <= RATE_NEGOTIATION_CONFIG["auto_accept_range"]:
            status = "accepted"
            counter_offer = proposed_rate
            response_message = f"Perfect! We can accept ${proposed_rate:.2f} for load {request.load_id}."
        elif proposed_rate >= current_rate - RATE_NEGOTIATION_CONFIG["counter_offer_range"]:
            status = "counter_offered"
            counter_offer = current_rate - RATE_NEGOTIATION_CONFIG["counter_offer_discount"]
            response_message = f"We're close! How about ${counter_offer:.2f} for load {request.load_id}? Our posted rate is ${current_rate:.2f}."
        else:  # Between counter_offer_range and max_below_posted
            status = "rejected"
            counter_offer = current_rate - RATE_NEGOTIATION_CONFIG["final_offer_discount"]
            response_message = f"Sorry, ${proposed_rate:.2f} is too low. Our best rate for load {request.load_id} is ${counter_offer:.2f}."
        
        call_id = f"CALL_{request.load_id}_{int(datetime.datetime.now().timestamp())}"
        
        # Log the negotiation decision
        logger.info(f"Negotiation result: {status} - {response_message}")
        
        # Record negotiation in database if available
        if DATABASE_AVAILABLE:
            try:
                conn = get_db_connection()
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO negotiations (call_id, load_id, mc_number, proposed_rate, counter_offer, final_rate, status)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                """, (call_id, request.load_id, request.get_mc_number(), proposed_rate, counter_offer, counter_offer, status))
                conn.commit()
                cursor.close()
                conn.close()
                logger.info(f"Negotiation recorded in database")
            except Exception as db_error:
                logger.warning(f"Could not record negotiation in database: {db_error}")
        
        return {
            "success": True,
            "negotiation_result": {
                "status": status,
                "original_rate": current_rate,
                "proposed_rate": proposed_rate,
                "counter_offer": counter_offer,
                "final_rate": counter_offer,
                "response_message": response_message,
                "call_id": call_id,
                "mc_number": request.get_mc_number(),
                "negotiation_config_used": RATE_NEGOTIATION_CONFIG
            }
        }
        
    except Exception as e:
        logger.error(f"Rate negotiation error: {e}")
        raise HTTPException(status_code=500, detail=f"Rate negotiation failed: {str(e)}")

@app.get("/test-rate-negotiation/{load_id}/{proposed_rate}")
async def test_rate_negotiation(load_id: str, proposed_rate: float, api_key: str = Depends(verify_api_key)):
    """Test endpoint to see how rate negotiation logic responds to different scenarios"""
    try:
        # Create test request
        test_request = RateNegotiationRequest(
            load_id=load_id,
            proposed_rate=proposed_rate,
            mc_number="TEST123"
        )
        
        # Call the actual negotiate_rate function
        result = await negotiate_rate(test_request, api_key)
        
        return {
            "success": True,
            "test_scenario": {
                "load_id": load_id,
                "proposed_rate": proposed_rate
            },
            "negotiation_config": RATE_NEGOTIATION_CONFIG,
            "result": result,
            "timestamp": datetime.datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "test_scenario": {
                "load_id": load_id,
                "proposed_rate": proposed_rate
            }
        }

@app.get("/rate-negotiation-config")
async def get_rate_negotiation_config(api_key: str = Depends(verify_api_key)):
    """Get current rate negotiation configuration settings"""
    return {
        "success": True,
        "config": RATE_NEGOTIATION_CONFIG,
        "explanation": {
            "max_above_posted": "Maximum amount above posted rate that will be accepted",
            "max_below_posted": "Maximum amount below posted rate that will be considered",
            "auto_accept_range": "Automatically accept rates within this range of posted rate",
            "counter_offer_range": "Make counter-offer if proposed rate is within this range below posted",
            "counter_offer_discount": "Amount below posted rate to counter-offer at",
            "final_offer_discount": "Amount below posted rate for final rejected offer"
        },
        "examples": {
            "posted_rate": 2500,
            "auto_accept": f"${2500 - RATE_NEGOTIATION_CONFIG['auto_accept_range']} - ${2500 + RATE_NEGOTIATION_CONFIG['auto_accept_range']}",
            "counter_offer": f"${2500 - RATE_NEGOTIATION_CONFIG['counter_offer_range']} - ${2500 - RATE_NEGOTIATION_CONFIG['auto_accept_range'] - 1}",
            "reject": f"Below ${2500 - RATE_NEGOTIATION_CONFIG['counter_offer_range']} or above ${2500 + RATE_NEGOTIATION_CONFIG['max_above_posted']}"
        }
    }

@app.post("/extract-call-data")
async def extract_call_data(request: CallDataExtractionRequest, api_key: str = Depends(verify_api_key)):
    """Extract structured data from call transcripts - Enhanced for HappyRobot"""
    try:
        logger.info(f"📞 Processing call transcript for data extraction: {len(request.call_transcript)} chars")
        
        # Extract detailed information from transcript
        extracted_info = enhanced_extract_carrier_info_from_text(request.call_transcript)
        
        # Store in database for metrics
        call_data = {
            "call_transcript": request.call_transcript,
            "mc_number": extracted_info.get("mc_number"),
            "company_name": extracted_info.get("company_name"),
            "equipment_type": extracted_info.get("equipment_type"),
            "original_rate": extracted_info.get("original_rate"),
            "proposed_rate": extracted_info.get("proposed_rate"),
            "final_outcome": extracted_info.get("final_outcome"),
            "sentiment": enhanced_analyze_sentiment(request.call_transcript)
        }
        
        # Store in database for real-time metrics
        await store_call_data_in_database(call_data)
        
        # Return in format expected by HappyRobot
        response_data = {
            "call_transcript": request.call_transcript,
            "mc_number": extracted_info.get("mc_number") or "",
            "company_name": extracted_info.get("company_name") or "",
            "equipment_type": extracted_info.get("equipment_type") or "",
            "load_ids_discussed": ",".join(extracted_info.get("load_ids_discussed", [])),
            "original_rate": str(extracted_info.get("original_rate", 0)),
            "proposed_rate": str(extracted_info.get("proposed_rate", 0)),
            "final_outcome": extracted_info.get("final_outcome", "inquiry_only")
        }
        
        logger.info(f"Successfully extracted call data: MC={response_data['mc_number']}, Company={response_data['company_name']}")
        
        return response_data
        
    except Exception as e:
        logger.error(f"Call data extraction error: {e}")
        raise HTTPException(status_code=500, detail="Call data extraction failed")

@app.post("/classify-call")
async def classify_call(request: CallClassificationRequest, api_key: str = Depends(verify_api_key)):
    """Classify call intent and outcome - Enhanced for HappyRobot"""
    try:
        logger.info(f"Classifying call transcript: {len(request.call_transcript)} chars")
        
        # Get enhanced classification
        classification = enhanced_classify_call_outcome(request.call_transcript)
        sentiment = enhanced_analyze_sentiment(request.call_transcript)
        
        # Store classification in database for metrics
        call_data = {
            "call_transcript": request.call_transcript,
            "classification": classification["response_classification"],
            "sentiment": sentiment
        }
        
        await store_call_data_in_database(call_data)
        
        # Return in format expected by HappyRobot
        response_data = {
            "call_transcript": request.call_transcript,
            "response_classification": classification["response_classification"],
            "response_reason": classification["response_reason"]
        }
        
        logger.info(f"Call classified as: {response_data['response_classification']}")
        
        return response_data
        
    except Exception as e:
        logger.error(f"Call classification error: {e}")
        raise HTTPException(status_code=500, detail="Call classification failed")

async def validate_and_negotiate_rates(extracted_info: Dict, db: Optional[AsyncSession] = None) -> Dict:
    """
    FIXED: Actually validate proposed rates against load rates and perform negotiation
    """
    result = {
        "negotiation_performed": False,
        "negotiation_result": None,
        "rates_validated": False,
        "load_found": False
    }
    
    # Check if we have both a load ID and proposed rate
    load_ids = extracted_info.get("load_ids_discussed", [])
    proposed_rate = extracted_info.get("proposed_rate")
    
    if not load_ids or not proposed_rate:
        logger.info("No load IDs or proposed rate found - skipping negotiation")
        return result
    
    # Get the first load ID mentioned
    load_id = load_ids[0]
    logger.info(f"Validating rate ${proposed_rate} against load {load_id}")
    
    # Find the load and its current rate
    load = None
    current_rate = None
    
    try:
        if DATABASE_AVAILABLE:
            conn = get_db_connection()
            cursor = conn.cursor(dictionary=True)
            cursor.execute("SELECT * FROM loads WHERE load_id = %s AND status = 'available'", (load_id,))
            load = cursor.fetchone()
            cursor.close()
            conn.close()
        else:
            # Check mock data
            for mock_load in MOCK_LOAD_DATA:
                if mock_load["load_id"] == load_id:
                    load = mock_load
                    break
        
        if not load:
            logger.warning(f"Load {load_id} not found - cannot validate rate")
            return result
        
        result["load_found"] = True
        current_rate = float(load.get('loadboard_rate', load.get('rate', 0)))
        
        if current_rate <= 0:
            logger.warning(f"Load {load_id} has invalid rate: {current_rate}")
            return result
        
        # Perform negotiation logic (same as /negotiate-rate endpoint)
        proposed_rate = float(proposed_rate)
        
        if proposed_rate >= current_rate * 0.95:  # Accept if within 5%
            status = "accepted"
            counter_offer = proposed_rate
            response_message = f"Great! We can accept ${proposed_rate:.2f} for load {load_id}."
            final_outcome = "load_booked"
        elif proposed_rate >= current_rate * 0.90:  # Counter-offer if within 10%
            status = "counter_offered"
            counter_offer = current_rate * 0.93  # Counter at 93% of original
            response_message = f"We're close! How about ${counter_offer:.2f} for load {load_id}?"
            final_outcome = "negotiation"
        else:  # Reject if too low
            status = "rejected"
            counter_offer = current_rate * 0.90  # Best offer at 90%
            response_message = f"Sorry, ${proposed_rate:.2f} is too low. Our best rate for load {load_id} is ${counter_offer:.2f}."
            final_outcome = "not_interested"
        
        # Store negotiation result
        negotiation_result = {
            "load_id": load_id,
            "original_rate": current_rate,
            "proposed_rate": proposed_rate,
            "counter_offer": counter_offer,
            "status": status,
            "response_message": response_message,
            "final_outcome": final_outcome
        }
        
        # Update the extracted info with negotiation results
        extracted_info["final_outcome"] = final_outcome
        extracted_info["negotiated_rate"] = counter_offer
        extracted_info["negotiation_status"] = status
        
        # Log negotiation results
        logger.info(f"Rate negotiation: ${proposed_rate} vs ${current_rate} -> {status} (${counter_offer})")
        
        result.update({
            "negotiation_performed": True,
            "negotiation_result": negotiation_result,
            "rates_validated": True
        })
        
        return result
        
    except Exception as e:
        logger.error(f"Error in rate validation: {e}")
        return result

@app.post("/webhooks/happyrobot/call-completed")
async def happyrobot_call_completed(
    request: Request,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
):
    try:
        payload = await request.json()
        logger.info("🎯 HappyRobot webhook received: call_id=%s", payload.get("call_id", "unknown"))

        # Normalise transcript (string or list → string)
        transcript_raw = payload.get("transcript", payload.get("call_transcript", ""))
        if isinstance(transcript_raw, list):
            transcript_raw = " ".join(str(part) for part in transcript_raw)

        # Build base dict with current timestamp
        call_info = {
            "happyrobot_call_id": payload.get(
                "call_id", f"HR_{int(datetime.datetime.utcnow().timestamp())}"
            ),
            "call_transcript": transcript_raw,
            "call_duration": float(payload.get("duration", payload.get("call_duration", 0))),
            "call_status": payload.get("status", "completed"),
            # Placeholders — will be filled by NLP helpers
            "carrier_mc": None,
            "company_name": None,
            "equipment_type": None,
            "original_rate": None,
            "proposed_rate": None,
            "final_outcome": "inquiry_only",
            "sentiment": "neutral",
            "call_outcome": "inquiry_only",
            "created_at": datetime.datetime.utcnow()
        }

        # NLP extraction from transcript
        extracted_info = {}
        if call_info["call_transcript"]:
            extracted_info = enhanced_extract_carrier_info_from_text(call_info["call_transcript"])
            call_info.update({
                "carrier_mc": extracted_info.get("mc_number"),
                "company_name": extracted_info.get("company_name"),
                "equipment_type": extracted_info.get("equipment_type"),
                "original_rate": extracted_info.get("original_rate"),
                "proposed_rate": extracted_info.get("proposed_rate"),
                "final_outcome": extracted_info.get("final_outcome", "inquiry_only"),
                "sentiment": enhanced_analyze_sentiment(call_info["call_transcript"]),
                "call_outcome": extracted_info.get("final_outcome", "inquiry_only")
            })

        # Perform rate validation and negotiation
        negotiation_result = await validate_and_negotiate_rates(extracted_info, db)
        
        # Update call info with negotiation results
        if negotiation_result["negotiation_performed"]:
            neg_data = negotiation_result["negotiation_result"]
            call_info.update({
                "final_outcome": neg_data["final_outcome"],
                "call_outcome": neg_data["final_outcome"],
                "proposed_rate": neg_data["counter_offer"],  # Use negotiated rate
                "original_rate": neg_data["original_rate"]
            })
            
            logger.info("Rate negotiation completed: %s -> %s", 
                       neg_data["status"], neg_data["response_message"])

        # Store the webhook data  
        await store_webhook_call_data(call_info, db)

        # Build response with negotiation info
        response_data = {
            "success": True,
            "message": "Call data processed successfully",
            "call_id": call_info["happyrobot_call_id"],
            "extracted_data": {
                "mc_number": call_info["carrier_mc"],
                "company_name": call_info["company_name"],
                "equipment_type": call_info["equipment_type"],
                "sentiment": call_info["sentiment"],
                "outcome": call_info["final_outcome"],
            },
            "stored_in": "database" if DATABASE_AVAILABLE else "memory",
            "timestamp": datetime.datetime.utcnow().isoformat()
        }

        # Add negotiation results to response
        if negotiation_result["negotiation_performed"]:
            response_data["rate_negotiation"] = negotiation_result["negotiation_result"]
            response_data["message"] = negotiation_result["negotiation_result"]["response_message"]
        else:
            response_data["rate_negotiation"] = {
                "status": "no_negotiation",
                "reason": "No rates or load IDs detected in transcript"
            }

        logger.info("Webhook processed: %s -> %s (%s)", 
                   call_info["happyrobot_call_id"], 
                   call_info["final_outcome"],
                   call_info["sentiment"])

        return response_data

    except Exception as e:
        logger.error("Webhook processing error: %s", e)
        import traceback
        logger.error("Stack trace: %s", traceback.format_exc())
        
        return {
            "success": False, 
            "error": str(e),
            "timestamp": datetime.datetime.utcnow().isoformat()
        }

# HappyRobot integration test endpoint
@app.get("/test-happyrobot")
async def test_happyrobot_integration(api_key: str = Depends(verify_api_key)):
    """Test HappyRobot webhook readiness (no API calls needed)"""
    try:
        logger.info("Testing HappyRobot webhook readiness...")
        
        # Validate configuration
        config = validate_happyrobot_config()
        
        # Test database connection for storing webhook data
        db_status = "connected" if DATABASE_AVAILABLE else "mock_mode"
        
        # Count existing webhook calls in database
        webhook_call_count = 0
        if DATABASE_AVAILABLE:
            try:
                conn = get_db_connection()
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM calls WHERE happyrobot_call_id LIKE 'HR_%'")
                webhook_call_count = cursor.fetchone()[0]
                cursor.close()
                conn.close()
            except:
                webhook_call_count = 0
        else:
            webhook_call_count = len(webhook_events)
        
        return {
            "success": True,
            "webhook_status": "ready",
            "happyrobot_config": config,
            "database_status": db_status,
            "webhook_calls_received": webhook_call_count,
            "webhook_endpoints": [
                f"{ENVIRONMENT == 'production' and 'https://carrier-sales-kavin.fly.dev' or 'http://localhost:8000'}/verify-carrier",
                f"{ENVIRONMENT == 'production' and 'https://carrier-sales-kavin.fly.dev' or 'http://localhost:8000'}/search-loads",
                f"{ENVIRONMENT == 'production' and 'https://carrier-sales-kavin.fly.dev' or 'http://localhost:8000'}/negotiate-rate",
                f"{ENVIRONMENT == 'production' and 'https://carrier-sales-kavin.fly.dev' or 'http://localhost:8000'}/extract-call-data",
                f"{ENVIRONMENT == 'production' and 'https://carrier-sales-kavin.fly.dev' or 'http://localhost:8000'}/classify-call"
            ],
            "message": "Webhooks ready to receive HappyRobot calls",
            "timestamp": datetime.datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        logger.error(f"HappyRobot webhook test error: {e}")
        return {
            "success": False,
            "error": str(e),
            "webhook_status": "error",
            "timestamp": datetime.datetime.now(timezone.utc).isoformat()
        }

@app.get("/test-fmcsa/{mc_number}")
async def test_fmcsa_api(mc_number: str, api_key: str = Depends(verify_api_key)):
    """Test FMCSA API connectivity"""
    if not FMCSA_API_KEY:
        return {
            "success": False,
            "error": "No FMCSA API key configured",
            "mc_number": mc_number
        }
    
    try:
        result = await query_fmcsa_api(mc_number, FMCSA_API_KEY)
        
        return {
            "success": bool(result),
            "mc_number": mc_number,
            "fmcsa_data": result,
            "api_key_partial": f"{FMCSA_API_KEY[:10]}...",
            "timestamp": datetime.datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        logger.error(f"FMCSA API test error: {e}")
        return {
            "success": False,
            "error": str(e),
            "mc_number": mc_number,
            "timestamp": datetime.datetime.now(timezone.utc).isoformat()
        }

@app.get("/webhooks/debug")
async def webhook_debug_info():
    """Debug endpoint to show webhook configuration"""
    try:
        # Use the correct API_BASE_URL logic
        api_base_url = "https://carrier-sales-kavin.fly.dev" if ENVIRONMENT == "production" else "http://localhost:8000"
        
        webhook_urls = {
            "call_completed": f"{api_base_url}/webhooks/happyrobot/call-completed",
            "verify_carrier": f"{api_base_url}/verify-carrier",
            "search_loads": f"{api_base_url}/search-loads",
            "negotiate_rate": f"{api_base_url}/negotiate-rate",
            "classify_call": f"{api_base_url}/classify-call",
            "extract_call_data": f"{api_base_url}/extract-call-data"
        }
        
        # Get recent webhook activity
        recent_webhooks = []
        webhook_count = 0
        if DATABASE_AVAILABLE:
            try:
                conn = get_db_connection()
                cursor = conn.cursor(dictionary=True)
                cursor.execute("""
                    SELECT COUNT(*) as count FROM calls 
                    WHERE happyrobot_call_id LIKE 'HR_%' OR happyrobot_call_id LIKE 'WEBHOOK_%'
                """)
                result = cursor.fetchone()
                webhook_count = result['count'] if result else 0
                
                cursor.execute("""
                    SELECT happyrobot_call_id, final_outcome, created_at 
                    FROM calls 
                    WHERE happyrobot_call_id LIKE 'HR_%' OR happyrobot_call_id LIKE 'WEBHOOK_%'
                    ORDER BY created_at DESC 
                    LIMIT 10
                """)
                recent_webhooks = cursor.fetchall()
                cursor.close()
                conn.close()
            except Exception as db_error:
                logger.error(f"Database query error: {db_error}")
        else:
            webhook_count = len(webhook_events)
            recent_webhooks = [{"happyrobot_call_id": e["happyrobot_call_id"], 
                               "final_outcome": e["final_outcome"], 
                               "created_at": datetime.datetime.now()} for e in webhook_events[-10:]]
        
        return {
            "webhook_endpoints": webhook_urls,
            "authentication": f"Bearer {API_KEY}",
            "recent_webhook_calls": webhook_count,
            "recent_activity": recent_webhooks,
            "database_available": DATABASE_AVAILABLE,
            "environment": ENVIRONMENT,
            "instructions": {
                "setup": "Configure these URLs in your HappyRobot campaign webhooks",
                "auth": "Use Bearer token authentication with the API key above",
                "method": "POST requests with JSON payload"
            }
        }
        
    except Exception as e:
        logger.error(f"Webhook debug error: {e}")
        return {
            "error": str(e),
            "environment": ENVIRONMENT,
            "api_key_configured": bool(API_KEY)
        }

@app.get("/dashboard/activity")
async def get_dashboard_activity():
    """FIXED: Get recent real-time activity for dashboard"""
    try:
        recent_activity = []
        
        # Get database activity
        if DATABASE_AVAILABLE:
            conn = get_db_connection()
            cursor = conn.cursor(dictionary=True)
            
            cursor.execute("""
                SELECT 
                    happyrobot_call_id,
                    carrier_mc,
                    company_name,
                    final_outcome,
                    sentiment,
                    equipment_type,
                    created_at
                FROM calls 
                WHERE created_at >= NOW() - INTERVAL 24 HOUR
                AND (happyrobot_call_id LIKE 'HR_%' OR happyrobot_call_id LIKE 'TEST_%')
                ORDER BY created_at DESC 
                LIMIT 20
            """)
            
            db_results = cursor.fetchall()
            
            for row in db_results:
                recent_activity.append({
                    "id": row["happyrobot_call_id"],
                    "timestamp": row["created_at"].isoformat() if row["created_at"] else None,
                    "carrier": row["company_name"] or f"MC {row['carrier_mc']}" if row["carrier_mc"] else "Unknown Carrier",
                    "outcome": row["final_outcome"] or "inquiry_only",
                    "sentiment": row["sentiment"] or "neutral",
                    "equipment": row["equipment_type"] or "Not specified",
                    "source": "database"
                })
            
            cursor.close()
            conn.close()
        
        # Add in-memory activity (from recent webhook calls)
        for event in webhook_events[-20:]:  # Get last 20 in-memory events
            recent_activity.append({
                "id": event["happyrobot_call_id"],
                "timestamp": event.get("created_at", datetime.datetime.utcnow()).isoformat(),
                "carrier": event.get("company_name") or f"MC {event.get('carrier_mc')}" if event.get('carrier_mc') else "Unknown Carrier",
                "outcome": event["final_outcome"],
                "sentiment": event["sentiment"],
                "equipment": event["equipment_type"] or "Not specified",
                "source": "memory"
            })
        
        # Sort by timestamp and remove duplicates
        recent_activity = sorted(recent_activity, key=lambda x: x["timestamp"] or "", reverse=True)[:20]
        
        # Determine data source
        data_source = "mock"
        if len(recent_activity) > 0:
            has_db_activity = any(a["source"] == "database" for a in recent_activity)
            has_memory_activity = any(a["source"] == "memory" for a in recent_activity)
            
            if has_db_activity:
                data_source = "webhook_database"
            elif has_memory_activity:
                data_source = "webhook_memory"
    
        return {
            "success": True,
            "activity": recent_activity,
            "total_today": len(recent_activity),
            "data_source": data_source,
            "webhook_events_count": len(webhook_events),
            "database_available": DATABASE_AVAILABLE
        }
        
    except Exception as e:
        logger.error(f"Error getting dashboard activity: {e}")
        return {
            "success": False, 
            "activity": [], 
            "error": str(e),
            "webhook_events_count": len(webhook_events),
            "database_available": DATABASE_AVAILABLE
        }

def read_html_file(file_path: str) -> str:
    """Read HTML file content"""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except FileNotFoundError:
        logger.error(f"HTML file not found: {file_path}")
        return "<h1>Dashboard file not found</h1><p>Please ensure index.html exists in the project directory.</p>"
    except Exception as e:
        logger.error(f"Error reading HTML file: {e}")
        return f"<h1>Error loading dashboard</h1><p>{str(e)}</p>"

@app.get("/", response_class=HTMLResponse)
async def get_dashboard():
    """Serve the HTML dashboard from index.html file"""
    html_content = read_html_file("dashboard/index.html")
    return HTMLResponse(content=html_content)

@app.get("/dashboard", response_class=HTMLResponse) 
async def get_dashboard_alt():
    """Alternative dashboard endpoint"""
    return await get_dashboard()

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=DEBUG,
        access_log=True,
        log_level="info"
    )