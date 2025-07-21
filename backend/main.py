"""
Carrier Sales API - Backend Server with HappyRobot API Integration
Real-time data from HappyRobot platform with FMCSA API integration
"""

import os
import logging
import mysql.connector
import httpx
import json
from datetime import datetime, timezone, timedelta
from typing import Optional, List, Dict, Any
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Depends, Request, BackgroundTasks
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator
import uvicorn
from dotenv import load_dotenv
import asyncio

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
HAPPYROBOT_BASE_URL = os.getenv("HAPPYROBOT_BASE_URL", "https://app.happyrobot.ai/api/v1")
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
DEBUG = os.getenv("DEBUG", "true").lower() == "true"

# Global variable to track database availability
DATABASE_AVAILABLE = False

# MySQL Configuration
MYSQL_CONFIG = {
    'host': os.getenv("MYSQL_HOST", "localhost"),
    'port': int(os.getenv("MYSQL_PORT", "3306")),
    'user': os.getenv("MYSQL_USER", "root"),
    'password': os.getenv("MYSQL_PASSWORD", "kavin2002"),
    'database': os.getenv("MYSQL_DATABASE", "carrier_sales"),
    'charset': 'utf8mb4',
    'use_unicode': True,
    'autocommit': True
}

# Warning logs
if API_KEY == "secure-api-key-change-this-in-production":
    logger.warning("⚠️  Using default API key! Change this in production!")

if not FMCSA_API_KEY:
    logger.warning("⚠️  No FMCSA API key provided! Using professional mock data only.")
else:
    logger.info(f"✅ FMCSA API key configured: {FMCSA_API_KEY[:10]}...")

if not HAPPYROBOT_API_KEY:
    logger.warning("⚠️  No HappyRobot API key provided! Using mock data for dashboard.")
else:
    logger.info(f"✅ HappyRobot API key configured: {HAPPYROBOT_API_KEY[:10]}...")
    logger.info(f"✅ HappyRobot Base URL: {HAPPYROBOT_BASE_URL}")

# Database connection functions
def get_db_connection():
    """Get MySQL database connection"""
    global DATABASE_AVAILABLE
    try:
        connection = mysql.connector.connect(**MYSQL_CONFIG)
        DATABASE_AVAILABLE = True
        return connection
    except mysql.connector.Error as err:
        logger.error(f"Database connection error: {err}")
        DATABASE_AVAILABLE = False
        if ENVIRONMENT == "production":
            logger.info("🚀 Production mode: Continuing with mock data")
            return None
        else:
            return None

def initialize_database():
    """Initialize database with required tables and sample data"""
    global DATABASE_AVAILABLE
    
    try:
        conn = get_db_connection()
        if not conn:
            logger.info("📦 Running in mock data mode")
            DATABASE_AVAILABLE = False
            return
            
        cursor = conn.cursor()
        
        # Create loads table
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
        
        # Enhanced calls table for HappyRobot data
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS calls (
                id INT AUTO_INCREMENT PRIMARY KEY,
                happyrobot_call_id VARCHAR(100) UNIQUE,
                carrier_mc VARCHAR(20),
                company_name VARCHAR(255),
                call_duration INT DEFAULT 0,
                call_transcript TEXT,
                call_outcome ENUM('interested', 'not_interested', 'callback', 'booked', 'follow_up_needed', 'negotiating') DEFAULT 'interested',
                sentiment ENUM('positive', 'neutral', 'negative') DEFAULT 'neutral',
                confidence_score DECIMAL(3,2) DEFAULT 0.00,
                loads_discussed TEXT,
                follow_up_date DATE,
                call_status VARCHAR(50) DEFAULT 'completed',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                INDEX idx_happyrobot_id (happyrobot_call_id),
                INDEX idx_carrier_mc (carrier_mc),
                INDEX idx_call_outcome (call_outcome),
                INDEX idx_sentiment (sentiment),
                INDEX idx_created_at (created_at)
            )
        """)
        
        # Enhanced negotiations table
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
        
        # Daily metrics snapshots
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
        logger.info("✅ MySQL database initialized successfully")
        DATABASE_AVAILABLE = True
        
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
            logger.info("✅ Sample loads inserted successfully")
        
        cursor.close()
        conn.close()
        
    except Exception as err:
        logger.warning(f"⚠️  Database initialization failed: {err}")
        logger.info("📦 Running in mock data mode")
        DATABASE_AVAILABLE = False

# Enhanced HappyRobot API Integration with better debugging
async def test_happyrobot_connection() -> Dict:
    """Test HappyRobot API connection with multiple authentication methods"""
    if not HAPPYROBOT_API_KEY:
        return {
            "success": False,
            "error": "No API key configured",
            "details": "HAPPYROBOT_API_KEY environment variable is not set"
        }
    
    # Test different authentication methods and endpoints
    test_results = []
    
    # Test 1: Bearer token authentication
    try:
        headers = {
            "Authorization": f"Bearer {HAPPYROBOT_API_KEY}",
            "Content-Type": "application/json"
        }
        
        logger.info(f"Testing HappyRobot connection with Bearer token to: {HAPPYROBOT_BASE_URL}")
        
        async with httpx.AsyncClient(timeout=15.0) as client:
            # Try different possible endpoints
            test_endpoints = [
                f"{HAPPYROBOT_BASE_URL}/calls",
                f"{HAPPYROBOT_BASE_URL}/campaigns", 
                f"{HAPPYROBOT_BASE_URL}/me",
                f"{HAPPYROBOT_BASE_URL}/account",
                f"{HAPPYROBOT_BASE_URL.rstrip('/v1')}/calls",  # Try without /v1
                f"{HAPPYROBOT_BASE_URL.rstrip('/api/v1')}/api/calls"  # Different structure
            ]
            
            for endpoint in test_endpoints:
                try:
                    response = await client.get(endpoint, headers=headers)
                    test_results.append({
                        "endpoint": endpoint,
                        "status_code": response.status_code,
                        "response": response.text[:200],
                        "success": response.status_code < 400
                    })
                    
                    if response.status_code < 400:
                        logger.info(f"✅ Success! Working endpoint: {endpoint}")
                        return {
                            "success": True,
                            "working_endpoint": endpoint,
                            "status_code": response.status_code,
                            "response_preview": response.text[:200],
                            "all_tests": test_results
                        }
                        
                except Exception as e:
                    test_results.append({
                        "endpoint": endpoint,
                        "error": str(e),
                        "success": False
                    })
                    
    except Exception as e:
        test_results.append({
            "method": "Bearer token",
            "error": str(e),
            "success": False
        })
    
    # Test 2: API Key in header
    try:
        headers = {
            "X-API-Key": HAPPYROBOT_API_KEY,
            "Content-Type": "application/json"
        }
        
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.get(f"{HAPPYROBOT_BASE_URL}/calls", headers=headers)
            test_results.append({
                "method": "X-API-Key header",
                "endpoint": f"{HAPPYROBOT_BASE_URL}/calls",
                "status_code": response.status_code,
                "response": response.text[:200],
                "success": response.status_code < 400
            })
            
            if response.status_code < 400:
                return {
                    "success": True,
                    "working_method": "X-API-Key",
                    "endpoint": f"{HAPPYROBOT_BASE_URL}/calls",
                    "status_code": response.status_code,
                    "all_tests": test_results
                }
                
    except Exception as e:
        test_results.append({
            "method": "X-API-Key header",
            "error": str(e),
            "success": False
        })
    
    # Test 3: API Key as query parameter
    try:
        params = {"api_key": HAPPYROBOT_API_KEY}
        
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.get(f"{HAPPYROBOT_BASE_URL}/calls", params=params)
            test_results.append({
                "method": "Query parameter",
                "endpoint": f"{HAPPYROBOT_BASE_URL}/calls",
                "status_code": response.status_code,
                "response": response.text[:200],
                "success": response.status_code < 400
            })
            
            if response.status_code < 400:
                return {
                    "success": True,
                    "working_method": "Query parameter",
                    "endpoint": f"{HAPPYROBOT_BASE_URL}/calls",
                    "status_code": response.status_code,
                    "all_tests": test_results
                }
                
    except Exception as e:
        test_results.append({
            "method": "Query parameter",
            "error": str(e),
            "success": False
        })
    
    # Return all failed attempts
    return {
        "success": False,
        "error": "All authentication methods failed",
        "api_key_preview": f"{HAPPYROBOT_API_KEY[:10]}...",
        "base_url": HAPPYROBOT_BASE_URL,
        "all_tests": test_results,
        "suggestions": [
            "Verify your API key is correct",
            "Check if you need to verify your email or account",
            "Try accessing the HappyRobot dashboard to confirm your account status",
            "Contact HappyRobot support for API documentation"
        ]
    }

async def fetch_happyrobot_calls(limit: int = 100) -> List[Dict]:
    """Fetch calls from HappyRobot API with enhanced error handling"""
    if not HAPPYROBOT_API_KEY:
        logger.warning("No HappyRobot API key available")
        return []
    
    try:
        # First test the connection
        connection_test = await test_happyrobot_connection()
        
        if not connection_test.get("success"):
            logger.error(f"HappyRobot connection test failed: {connection_test.get('error')}")
            return []
        
        # Use the working endpoint and method
        working_endpoint = connection_test.get("working_endpoint", f"{HAPPYROBOT_BASE_URL}/calls")
        working_method = connection_test.get("working_method", "Bearer")
        
        if working_method == "Bearer":
            headers = {
                "Authorization": f"Bearer {HAPPYROBOT_API_KEY}",
                "Content-Type": "application/json"
            }
            params = {"limit": limit, "page": 1}
        elif working_method == "X-API-Key":
            headers = {
                "X-API-Key": HAPPYROBOT_API_KEY,
                "Content-Type": "application/json"
            }
            params = {"limit": limit, "page": 1}
        else:  # Query parameter
            headers = {"Content-Type": "application/json"}
            params = {"api_key": HAPPYROBOT_API_KEY, "limit": limit, "page": 1}
        
        logger.info(f"Fetching calls from HappyRobot API: {working_endpoint}")
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(working_endpoint, headers=headers, params=params)
            
            if response.status_code == 200:
                data = response.json()
                calls = data.get("data", []) if isinstance(data, dict) else data
                logger.info(f"Successfully fetched {len(calls)} calls from HappyRobot")
                return calls
            else:
                logger.warning(f"HappyRobot API returned status {response.status_code}")
                logger.warning(f"Response: {response.text[:200]}")
                return []
                
    except Exception as e:
        logger.error(f"Error fetching HappyRobot calls: {e}")
        return []

async def fetch_happyrobot_call_transcript(call_id: str) -> str:
    """Fetch call transcript from HappyRobot API"""
    if not HAPPYROBOT_API_KEY:
        return ""
    
    try:
        headers = {
            "Authorization": f"Bearer {HAPPYROBOT_API_KEY}",
            "Content-Type": "application/json"
        }
        
        url = f"{HAPPYROBOT_BASE_URL}/calls/{call_id}/transcript"
        
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.get(url, headers=headers)
            
            if response.status_code == 200:
                data = response.json()
                transcript = data.get("transcript", "") if isinstance(data, dict) else str(data)
                return transcript
            else:
                logger.warning(f"Failed to fetch transcript for call {call_id}: {response.status_code}")
                return ""
                
    except Exception as e:
        logger.error(f"Error fetching transcript for call {call_id}: {e}")
        return ""

async def process_happyrobot_call_data(call_data: Dict) -> Dict:
    """Process HappyRobot call data and extract relevant information"""
    try:
        # Extract basic call info
        call_id = call_data.get("id", "")
        duration = call_data.get("duration", 0)
        status = call_data.get("status", "completed")
        created_at = call_data.get("created_at", "")
        
        # Fetch transcript if available
        transcript = ""
        if call_id:
            transcript = await fetch_happyrobot_call_transcript(call_id)
        
        # Extract carrier info from transcript or call data
        carrier_info = extract_carrier_info_from_text(transcript)
        
        # Determine call outcome and sentiment
        outcome = classify_call_outcome(transcript)
        sentiment = analyze_sentiment(transcript)
        
        processed_call = {
            "happyrobot_call_id": call_id,
            "call_duration": duration,
            "call_transcript": transcript,
            "call_status": status,
            "call_outcome": outcome,
            "sentiment": sentiment,
            "carrier_mc": carrier_info.get("mc_number", ""),
            "company_name": carrier_info.get("company_name", ""),
            "created_at": created_at,
            "confidence_score": 0.8  # Default confidence
        }
        
        return processed_call
        
    except Exception as e:
        logger.error(f"Error processing HappyRobot call data: {e}")
        return {}

def extract_carrier_info_from_text(text: str) -> Dict:
    """Extract carrier information from call transcript"""
    import re
    
    carrier_info = {}
    
    if not text:
        return carrier_info
    
    text_lower = text.lower()
    
    # Extract MC number
    mc_matches = re.findall(r'mc\s*(?:number\s*)?(\d{4,8})', text_lower)
    if mc_matches:
        carrier_info["mc_number"] = mc_matches[0]
    
    # Extract company name
    company_patterns = [
        r'this is ([A-Za-z\s&]+) (?:trucking|transport|logistics|freight|inc|llc)',
        r'calling from ([A-Za-z\s&]+)',
        r'my company is ([A-Za-z\s&]+)',
        r'we are ([A-Za-z\s&]+) (?:trucking|transport)'
    ]
    
    for pattern in company_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            carrier_info["company_name"] = matches[0].strip().title()
            break
    
    return carrier_info

def classify_call_outcome(transcript: str) -> str:
    """Classify call outcome based on transcript"""
    if not transcript:
        return "interested"
    
    transcript_lower = transcript.lower()
    
    # Check for booking indicators
    if any(word in transcript_lower for word in ["book it", "take it", "sounds good", "deal", "yes", "we'll take"]):
        return "booked"
    
    # Check for follow-up indicators
    elif any(word in transcript_lower for word in ["call back", "follow up", "think about", "let me check"]):
        return "follow_up_needed"
    
    # Check for not interested indicators
    elif any(word in transcript_lower for word in ["not interested", "pass", "no thanks", "too low"]):
        return "not_interested"
    
    # Check for negotiation indicators
    elif any(word in transcript_lower for word in ["rate", "price", "negotiate", "counter"]):
        return "negotiating"
    
    # Default to interested
    else:
        return "interested"

def analyze_sentiment(transcript: str) -> str:
    """Analyze sentiment of call transcript"""
    if not transcript:
        return "neutral"
    
    transcript_lower = transcript.lower()
    
    positive_words = ["interested", "good", "great", "excellent", "perfect", "sounds good", "yes", "fantastic"]
    negative_words = ["not interested", "too low", "can't do", "no way", "pass", "no", "terrible", "awful"]
    
    positive_count = sum(1 for word in positive_words if word in transcript_lower)
    negative_count = sum(1 for word in negative_words if word in transcript_lower)
    
    if positive_count > negative_count:
        return "positive"
    elif negative_count > positive_count:
        return "negative"
    else:
        return "neutral"

async def sync_happyrobot_calls_to_database():
    """Sync HappyRobot calls to local database"""
    if not DATABASE_AVAILABLE:
        logger.info("Database not available, skipping sync")
        return
    
    try:
        # Fetch recent calls from HappyRobot
        calls = await fetch_happyrobot_calls(limit=50)
        
        if not calls:
            logger.info("No calls fetched from HappyRobot")
            return
        
        conn = get_db_connection()
        cursor = conn.cursor()
        
        synced_count = 0
        
        for call_data in calls:
            processed_call = await process_happyrobot_call_data(call_data)
            
            if not processed_call.get("happyrobot_call_id"):
                continue
            
            # Insert or update call in database
            cursor.execute("""
                INSERT INTO calls 
                (happyrobot_call_id, carrier_mc, company_name, call_duration, call_transcript, 
                 call_outcome, sentiment, call_status, created_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE
                call_duration = VALUES(call_duration),
                call_transcript = VALUES(call_transcript),
                call_outcome = VALUES(call_outcome),
                sentiment = VALUES(sentiment),
                call_status = VALUES(call_status),
                updated_at = CURRENT_TIMESTAMP
            """, (
                processed_call["happyrobot_call_id"],
                processed_call["carrier_mc"],
                processed_call["company_name"],
                processed_call["call_duration"],
                processed_call["call_transcript"],
                processed_call["call_outcome"],
                processed_call["sentiment"],
                processed_call["call_status"],
                processed_call["created_at"]
            ))
            
            synced_count += 1
        
        conn.commit()
        cursor.close()
        conn.close()
        
        logger.info(f"✅ Synced {synced_count} calls from HappyRobot to database")
        
    except Exception as e:
        logger.error(f"Error syncing HappyRobot calls: {e}")

async def calculate_real_time_metrics_from_happyrobot():
    """Calculate dashboard metrics from HappyRobot data"""
    try:
        # First, sync recent calls
        await sync_happyrobot_calls_to_database()
        
        if DATABASE_AVAILABLE:
            # Calculate from database
            return await calculate_metrics_from_database()
        else:
            # Calculate directly from HappyRobot API
            return await calculate_metrics_from_happyrobot_api()
            
    except Exception as e:
        logger.error(f"Error calculating real-time metrics: {e}")
        return await get_fallback_metrics()

async def calculate_metrics_from_database():
    """Calculate metrics from synced database data"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        
        # Get date ranges
        today = datetime.now().date()
        week_ago = today - timedelta(days=7)
        
        # Total calls
        cursor.execute("SELECT COUNT(*) as total FROM calls")
        total_calls = cursor.fetchone()['total']
        
        # Recent calls (last 7 days)
        cursor.execute("SELECT COUNT(*) as recent FROM calls WHERE DATE(created_at) >= %s", (week_ago,))
        recent_calls = cursor.fetchone()['recent']
        
        # Conversion rate (booked calls)
        cursor.execute("SELECT COUNT(*) as booked FROM calls WHERE call_outcome = 'booked'")
        booked_calls = cursor.fetchone()['booked']
        conversion_rate = (booked_calls / total_calls * 100) if total_calls > 0 else 0
        
        # Average negotiated rate (from negotiations table)
        cursor.execute("SELECT AVG(final_rate) as avg_rate FROM negotiations WHERE status = 'accepted' AND final_rate > 0")
        avg_rate_result = cursor.fetchone()
        avg_rate = avg_rate_result['avg_rate'] if avg_rate_result['avg_rate'] else 2500
        
        # Call classifications
        cursor.execute("""
            SELECT call_outcome as type, COUNT(*) as count 
            FROM calls 
            WHERE call_outcome IS NOT NULL 
            GROUP BY call_outcome
        """)
        classifications = [{"type": row['type'], "count": row['count']} for row in cursor.fetchall()]
        
        # Sentiments
        cursor.execute("""
            SELECT sentiment as type, COUNT(*) as count 
            FROM calls 
            WHERE sentiment IS NOT NULL 
            GROUP BY sentiment
        """)
        sentiments = [{"type": row['type'], "count": row['count']} for row in cursor.fetchall()]
        
        # Equipment performance (mock for now since we need to enhance data extraction)
        equipment_performance = [
            {"type": "Dry Van", "calls": int(total_calls * 0.4)},
            {"type": "Flatbed", "calls": int(total_calls * 0.3)},
            {"type": "Refrigerated", "calls": int(total_calls * 0.2)},
            {"type": "Step Deck", "calls": int(total_calls * 0.1)}
        ]
        
        cursor.close()
        conn.close()
        
        metrics = {
            "summary": {
                "total_calls": total_calls,
                "recent_calls": recent_calls,
                "conversion_rate": round(conversion_rate, 1),
                "average_negotiated_rate": int(avg_rate)
            },
            "classifications": classifications,
            "sentiments": sentiments,
            "equipment_performance": equipment_performance,
            "recent_activity": [],
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "data_source": "happyrobot_database"
        }
        
        return metrics
        
    except Exception as e:
        logger.error(f"Error calculating metrics from database: {e}")
        return await get_fallback_metrics()

async def calculate_metrics_from_happyrobot_api():
    """Calculate metrics directly from HappyRobot API"""
    try:
        calls = await fetch_happyrobot_calls(limit=100)
        
        if not calls:
            return await get_fallback_metrics()
        
        total_calls = len(calls)
        recent_calls = len([c for c in calls if is_recent_call(c.get("created_at", ""))])
        
        # Process calls to get outcomes and sentiments
        outcomes = {}
        sentiments = {}
        
        for call_data in calls[:20]:  # Limit processing for performance
            processed_call = await process_happyrobot_call_data(call_data)
            
            outcome = processed_call.get("call_outcome", "interested")
            sentiment = processed_call.get("sentiment", "neutral")
            
            outcomes[outcome] = outcomes.get(outcome, 0) + 1
            sentiments[sentiment] = sentiments.get(sentiment, 0) + 1
        
        # Convert to expected format
        classifications = [{"type": k, "count": v} for k, v in outcomes.items()]
        sentiment_list = [{"type": k, "count": v} for k, v in sentiments.items()]
        
        # Calculate conversion rate
        booked_count = outcomes.get("booked", 0)
        conversion_rate = (booked_count / total_calls * 100) if total_calls > 0 else 0
        
        equipment_performance = [
            {"type": "Dry Van", "calls": int(total_calls * 0.4)},
            {"type": "Flatbed", "calls": int(total_calls * 0.3)},
            {"type": "Refrigerated", "calls": int(total_calls * 0.2)},
            {"type": "Step Deck", "calls": int(total_calls * 0.1)}
        ]
        
        metrics = {
            "summary": {
                "total_calls": total_calls,
                "recent_calls": recent_calls,
                "conversion_rate": round(conversion_rate, 1),
                "average_negotiated_rate": 2650  # Default since we don't have negotiation data yet
            },
            "classifications": classifications,
            "sentiments": sentiment_list,
            "equipment_performance": equipment_performance,
            "recent_activity": [],
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "data_source": "happyrobot_api"
        }
        
        return metrics
        
    except Exception as e:
        logger.error(f"Error calculating metrics from HappyRobot API: {e}")
        return await get_fallback_metrics()

def is_recent_call(created_at: str) -> bool:
    """Check if call was made in the last 7 days"""
    try:
        if not created_at:
            return False
        
        # Parse the date string (adjust format as needed)
        call_date = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
        week_ago = datetime.now(timezone.utc) - timedelta(days=7)
        
        return call_date >= week_ago
    except Exception:
        return False

async def get_fallback_metrics():
    """Fallback metrics when HappyRobot data is not available"""
    from random import randint
    
    metrics = {
        "summary": {
            "total_calls": randint(15, 35),
            "recent_calls": randint(3, 12),
            "conversion_rate": round(15 + (randint(-3, 8) * 0.1), 1),
            "average_negotiated_rate": randint(2200, 2800)
        },
        "classifications": [
            {"type": "load_inquiry", "count": randint(5, 15)},
            {"type": "rate_negotiation", "count": randint(2, 8)},
            {"type": "not_interested", "count": randint(1, 5)},
            {"type": "booked", "count": randint(1, 4)},
            {"type": "follow_up_needed", "count": randint(1, 6)}
        ],
        "sentiments": [
            {"type": "positive", "count": randint(8, 18)},
            {"type": "neutral", "count": randint(5, 12)},
            {"type": "negative", "count": randint(1, 6)}
        ],
        "equipment_performance": [
            {"type": "Dry Van", "calls": randint(8, 18)},
            {"type": "Flatbed", "calls": randint(4, 12)},
            {"type": "Refrigerated", "calls": randint(3, 10)},
            {"type": "Step Deck", "calls": randint(1, 6)}
        ],
        "recent_activity": [],
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "data_source": "fallback_mock"
    }
    
    return metrics

# Professional Mock Data (fallback)
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

# FMCSA API Integration (unchanged)
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

# Pydantic Models (unchanged)
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

# Background sync task (disabled by default until connection is working)
async def periodic_happyrobot_sync():
    """Periodically sync HappyRobot data"""
    # Test connection first
    connection_test = await test_happyrobot_connection()
    
    if not connection_test.get("success"):
        logger.warning("HappyRobot connection failed, disabling periodic sync")
        logger.info(f"Connection test results: {connection_test}")
        return
    
    logger.info("✅ HappyRobot connection successful, starting periodic sync")
    
    while True:
        try:
            logger.info("🔄 Starting periodic HappyRobot sync...")
            await sync_happyrobot_calls_to_database()
            await asyncio.sleep(300)  # Sync every 5 minutes
        except Exception as e:
            logger.error(f"Error in periodic sync: {e}")
            await asyncio.sleep(60)  # Wait 1 minute before retrying

# Lifespan event handler
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler"""
    # Startup
    logger.info("🚛 Starting Carrier Sales API with HappyRobot Integration...")
    initialize_database()
    
    # Test HappyRobot connection before starting sync task
    if HAPPYROBOT_API_KEY:
        connection_test = await test_happyrobot_connection()
        if connection_test.get("success"):
            asyncio.create_task(periodic_happyrobot_sync())
            logger.info("✅ HappyRobot sync task started")
        else:
            logger.warning("⚠️ HappyRobot connection failed, sync disabled")
            logger.info("Use /test-happyrobot endpoint to debug connection issues")
    
    logger.info("✅ Carrier Sales API started successfully")
    yield
    # Shutdown
    logger.info("🛑 Shutting down Carrier Sales API...")

# Initialize FastAPI app
app = FastAPI(
    title="Carrier Sales API with HappyRobot Integration",
    description="AI-powered freight broker assistant with real HappyRobot call data and FMCSA integration",
    version="1.2.0",
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
    start_time = datetime.now()
    logger.info(f"📥 {request.method} {request.url.path} from {request.client.host}")
    response = await call_next(request)
    process_time = (datetime.now() - start_time).total_seconds()
    logger.info(f"📤 {request.method} {request.url.path} -> {response.status_code} ({process_time:.3f}s)")
    return response

# API Endpoints
@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Carrier Sales API with HappyRobot Integration",
        "version": "1.2.0",
        "environment": ENVIRONMENT,
        "fmcsa_api_available": bool(FMCSA_API_KEY),
        "happyrobot_api_available": bool(HAPPYROBOT_API_KEY),
        "documentation": "/docs",
        "health_check": "/health",
        "debug_endpoints": {
            "test_happyrobot": "/test-happyrobot",
            "test_fmcsa": "/test-fmcsa/{mc_number}"
        }
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
        
        return {
            "status": "healthy",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "environment": ENVIRONMENT,
            "services": {
                "database": database_status,
                "fmcsa_api": "configured" if FMCSA_API_KEY else "mock_only",
                "happyrobot_api": "configured" if HAPPYROBOT_API_KEY else "mock_only"
            },
            "version": "1.2.0",
            "happyrobot_config": {
                "api_key_configured": bool(HAPPYROBOT_API_KEY),
                "base_url": HAPPYROBOT_BASE_URL,
                "key_preview": f"{HAPPYROBOT_API_KEY[:10]}..." if HAPPYROBOT_API_KEY else None
            }
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "degraded",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "environment": ENVIRONMENT,
            "services": {
                "database": "mock",
                "fmcsa_api": "configured" if FMCSA_API_KEY else "mock_only",
                "happyrobot_api": "configured" if HAPPYROBOT_API_KEY else "mock_only"
            },
            "version": "1.2.0",
            "error": str(e)
        }

@app.get("/dashboard-metrics")
async def get_dashboard_metrics():
    """Get real-time dashboard metrics from HappyRobot"""
    try:
        logger.info("📊 Calculating dashboard metrics from HappyRobot data...")
        
        # Get real metrics from HappyRobot
        metrics = await calculate_real_time_metrics_from_happyrobot()
        
        return {
            "success": True,
            "metrics": metrics
        }
        
    except Exception as e:
        logger.error(f"Dashboard metrics error: {e}")
        # Return fallback metrics if everything fails
        fallback_metrics = await get_fallback_metrics()
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
                    fmcsa_data["verified_at"] = datetime.now(timezone.utc).isoformat()
                    
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
            carrier_data["verified_at"] = datetime.now(timezone.utc).isoformat()
            
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

@app.post("/negotiate-rate")
async def negotiate_rate(request: RateNegotiationRequest, api_key: str = Depends(verify_api_key)):
    """Handle rate negotiation for a specific load"""
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
        
        # Negotiation logic
        if proposed_rate >= current_rate * 0.95:
            status = "accepted"
            counter_offer = proposed_rate
            response_message = f"Great! We can accept ${proposed_rate:.2f} for load {request.load_id}."
        elif proposed_rate >= current_rate * 0.90:
            status = "counter_offered"
            counter_offer = current_rate * 0.93
            response_message = f"We're close! How about ${counter_offer:.2f} for load {request.load_id}?"
        else:
            status = "rejected"
            counter_offer = current_rate * 0.90
            response_message = f"Sorry, ${proposed_rate:.2f} is too low. Our best rate for load {request.load_id} is ${counter_offer:.2f}."
        
        call_id = f"CALL_{request.load_id}_{int(datetime.now().timestamp())}"
        
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
                "mc_number": request.get_mc_number()
            }
        }
        
    except Exception as e:
        logger.error(f"Rate negotiation error: {e}")
        raise HTTPException(status_code=500, detail=f"Rate negotiation failed: {str(e)}")

@app.post("/extract-call-data")
async def extract_call_data(request: CallDataExtractionRequest, api_key: str = Depends(verify_api_key)):
    """Extract structured data from call transcripts"""
    try:
        transcript = request.call_transcript.lower()
        
        extracted_data = {
            "carrier_info": {},
            "equipment_needs": [],
            "locations_mentioned": [],
            "rates_discussed": [],
            "follow_up_required": False,
            "sentiment": "neutral",
            "key_phrases": [],
            "action_items": []
        }
        
        # Use the same extraction logic as before
        extracted_data.update(extract_carrier_info_from_text(request.call_transcript))
        extracted_data["sentiment"] = analyze_sentiment(request.call_transcript)
        
        return {
            "success": True,
            "extracted_data": extracted_data,
            "transcript_length": len(request.call_transcript),
            "processing_time": datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        logger.error(f"Call data extraction error: {e}")
        raise HTTPException(status_code=500, detail="Call data extraction failed")

@app.post("/classify-call")
async def classify_call(request: CallClassificationRequest, api_key: str = Depends(verify_api_key)):
    """Classify call intent and outcome"""
    try:
        transcript = request.call_transcript.lower()
        
        classification = {
            "intent": "unknown",
            "outcome": "unknown",
            "priority": "medium",
            "action_required": [],
            "confidence": 0.0,
            "reasoning": []
        }
        
        # Use the same classification logic as before
        classification["outcome"] = classify_call_outcome(request.call_transcript)
        classification["confidence"] = 0.8  # Default confidence
        
        return {
            "success": True,
            "classification": classification,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        logger.error(f"Call classification error: {e}")
        raise HTTPException(status_code=500, detail="Call classification failed")

# Enhanced HappyRobot integration test endpoint
@app.get("/test-happyrobot")
async def test_happyrobot_integration(api_key: str = Depends(verify_api_key)):
    """Enhanced HappyRobot API integration test with detailed debugging"""
    if not HAPPYROBOT_API_KEY:
        return {
            "success": False,
            "error": "No HappyRobot API key configured",
            "message": "Please set HAPPYROBOT_API_KEY environment variable",
            "config_help": {
                "env_var": "HAPPYROBOT_API_KEY",
                "base_url_var": "HAPPYROBOT_BASE_URL",
                "current_base_url": HAPPYROBOT_BASE_URL
            }
        }
    
    try:
        logger.info("🧪 Running comprehensive HappyRobot API test...")
        
        connection_test = await test_happyrobot_connection()
        
        return {
            "success": connection_test.get("success", False),
            "connection_test": connection_test,
            "api_config": {
                "api_key_preview": f"{HAPPYROBOT_API_KEY[:10]}...",
                "base_url": HAPPYROBOT_BASE_URL,
                "key_length": len(HAPPYROBOT_API_KEY)
            },
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        logger.error(f"HappyRobot integration test error: {e}")
        return {
            "success": False,
            "error": str(e),
            "happyrobot_api_status": "failed",
            "timestamp": datetime.now(timezone.utc).isoformat()
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
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        logger.error(f"FMCSA API test error: {e}")
        return {
            "success": False,
            "error": str(e),
            "mc_number": mc_number,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=DEBUG,
        access_log=True,
        log_level="info"
    )