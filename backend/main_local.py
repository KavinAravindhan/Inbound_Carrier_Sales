"""
Carrier Sales API - LOCAL DEVELOPMENT VERSION
Local development with mock data fallback when no database available
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

# Configuration - LOCAL DEVELOPMENT
API_KEY = os.getenv("API_KEY", "secure-api-key-change-this-in-production")
FMCSA_API_KEY = os.getenv("FMCSA_API_KEY", "")
HAPPYROBOT_API_KEY = os.getenv("HAPPYROBOT_API_KEY", "")
HAPPYROBOT_BASE_URL = os.getenv("HAPPYROBOT_BASE_URL", "https://platform.happyrobot.ai/api/v1")
ENVIRONMENT = "development"  # Force development mode
DEBUG = True

# Global variable to track database availability
DATABASE_AVAILABLE = False

# LOCAL MySQL Configuration
MYSQL_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
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

# Global variables for in-memory webhook storage (local development)
webhook_events: List[Dict] = []

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
        "id": "LD002",
        "pickup_location": "Los Angeles, CA",
        "delivery_location": "Phoenix, AZ",
        "rate": 1800.00,
        "pickup_date": "2025-07-23",
        "delivery_date": "2025-07-24"
    }
]

# Enhanced data extraction functions (same as production)
def extract_mc_number(text: str) -> Optional[str]:
    """Extract MC number from text"""
    if not text:
        return None
    
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
    
    company_patterns = [
        r'this is ([A-Za-z0-9\s&\-\',\.]+)(?:\s+(?:trucking|transport|logistics|freight|inc|llc|corp|company))',
        r'calling from ([A-Za-z0-9\s&\-\',\.]+)(?:\s+(?:trucking|transport|logistics|freight))?',
        r'my company is ([A-Za-z0-9\s&\-\',\.]+)',
        r'we are ([A-Za-z0-9\s&\-\',\.]+)(?:\s+(?:trucking|transport))?',
    ]
    
    for pattern in company_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            company = matches[0].strip()
            company = re.sub(r'\s+', ' ', company)
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

def extract_rates(text: str) -> Dict[str, Optional[float]]:
    """Extract rates from text"""
    if not text:
        return {"original_rate": None, "proposed_rate": None}
    
    rate_patterns = [
        r'\$(\d{1,2}[,.]?\d{3,4})',
        r'(\d{1,2}[,.]?\d{3,4})\s*(?:dollars?|bucks?)',
        r'rate\s*(?:of\s*)?(\d{1,2}[,.]?\d{3,4})',
        r'(\d{1,2}[,.]?\d{3,4})\s*(?:for|rate)'
    ]
    
    rates = []
    for pattern in rate_patterns:
        matches = re.findall(pattern, text.lower())
        for match in matches:
            try:
                clean_rate = match.replace(',', '')
                rate = float(clean_rate)
                if 500 <= rate <= 10000:
                    rates.append(rate)
            except ValueError:
                continue
    
    rates = list(set(rates))
    rates.sort()
    
    result = {"original_rate": None, "proposed_rate": None}
    if len(rates) >= 2:
        result["original_rate"] = rates[-1]
        result["proposed_rate"] = rates[-2]
    elif len(rates) == 1:
        result["proposed_rate"] = rates[0]
    
    return result

def determine_final_outcome(text: str) -> str:
    """Determine final outcome from text"""
    if not text:
        return "inquiry_only"
    
    text_lower = text.lower()
    
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
        "final_outcome": determine_final_outcome(text)
    }
    
    rates = extract_rates(text)
    extracted_info.update(rates)
    
    return extracted_info

def enhanced_classify_call_outcome(transcript: str) -> Dict:
    """Enhanced call outcome classification"""
    if not transcript or transcript.strip() == "[]":
        return {
            "response_classification": "inquiry_only",
            "response_reason": "Demo classification - carrier inquired about available loads"
        }
    
    transcript_lower = transcript.lower()
    
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
        return "positive"
    
    transcript_lower = transcript.lower()
    
    positive_indicators = [
        "interested", "good", "great", "excellent", "perfect", "sounds good", 
        "yes", "fantastic", "awesome", "love it", "definitely", "absolutely"
    ]
    
    negative_indicators = [
        "not interested", "too low", "can't do", "no way", "pass", "no", 
        "terrible", "awful", "frustrated", "annoyed", "disappointed"
    ]
    
    positive_count = sum(1 for word in positive_indicators if word in transcript_lower)
    negative_count = sum(1 for word in negative_indicators if word in transcript_lower)
    
    if positive_count > negative_count:
        return "positive"
    elif negative_count > positive_count:
        return "negative"
    else:
        return "neutral"

# Database connection for local development
def get_db_connection():
    """Get local MySQL database connection with fallback"""
    global DATABASE_AVAILABLE
    try:
        logger.info(f"Attempting local database connection to {MYSQL_CONFIG['host']}:{MYSQL_CONFIG['port']}")
        connection = mysql.connector.connect(**MYSQL_CONFIG)
        cursor = connection.cursor()
        cursor.execute("SELECT 1")
        cursor.fetchone()
        cursor.close()
        DATABASE_AVAILABLE = True
        logger.info(f"✅ Local database connection successful")
        return connection
    except mysql.connector.Error as err:
        logger.warning(f"Local database connection failed: {err}")
        logger.info("📦 Running in mock data mode (this is normal for local development)")
        DATABASE_AVAILABLE = False
        return None

# Local metrics for development
async def get_local_metrics():
    """Get metrics for local development"""
    memory_webhook_count = len(webhook_events)
    
    if memory_webhook_count > 0:
        # Calculate from in-memory webhook data
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
            "classifications": classifications if classifications else [{"type": "inquiry_only", "count": memory_webhook_count}],
            "sentiments": sentiments if sentiments else [{"type": "positive", "count": memory_webhook_count}],
            "equipment_performance": equipment_performance if equipment_performance else [{"type": "Dry Van", "calls": memory_webhook_count}],
            "recent_activity": [],
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "data_source": "webhook_memory_real_time"
        }
    else:
        # Return mock data for local development
        from random import randint, uniform
        total_calls = randint(8, 15)
        return {
            "summary": {
                "total_calls": total_calls,
                "recent_calls": randint(3, 8),
                "conversion_rate": round(uniform(12.5, 18.7), 1),
                "average_negotiated_rate": randint(2400, 2950)
            },
            "classifications": [
                {"type": "inquiry_only", "count": randint(3, 8)},
                {"type": "negotiation", "count": randint(2, 5)},
                {"type": "load_booked", "count": randint(1, 3)},
                {"type": "not_interested", "count": randint(0, 2)}
            ],
            "sentiments": [
                {"type": "positive", "count": randint(5, 10)},
                {"type": "neutral", "count": randint(2, 5)},
                {"type": "negative", "count": randint(0, 2)}
            ],
            "equipment_performance": [
                {"type": "Dry Van", "calls": randint(3, 8)},
                {"type": "Flatbed", "calls": randint(1, 4)},
                {"type": "Refrigerated", "calls": randint(1, 3)}
            ],
            "recent_activity": [],
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "data_source": "enhanced_mock_data"
        }

# FMCSA API Integration (same as production)
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

# Pydantic Models (same as production)
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

# Initialize FastAPI app
app = FastAPI(
    title="Enhanced Carrier Sales API - LOCAL DEVELOPMENT",
    description="AI-powered freight broker assistant with HappyRobot integration - Local Development Version",
    version="1.3.0-local",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security (simplified for local development)
security = HTTPBearer()

def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify API key authentication (simplified for local dev)"""
    if credentials.credentials != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return credentials.credentials

# API Endpoints
@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Enhanced Carrier Sales API - LOCAL DEVELOPMENT",
        "version": "1.3.0-local",
        "environment": "development",
        "database_available": DATABASE_AVAILABLE,
        "documentation": "/docs",
        "health_check": "/health",
        "features": [
            "Local development mode",
            "Mock data fallback", 
            "Real FMCSA API integration",
            "HappyRobot webhook simulation"
        ]
    }

@app.get("/health")
async def health_check():
    """Local health check endpoint"""
    try:
        database_status = "connected" if DATABASE_AVAILABLE else "mock"
        
        return {
            "status": "healthy",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "environment": "development",
            "services": {
                "database": database_status,
                "fmcsa_api": "configured" if FMCSA_API_KEY else "mock_only",
                "happyrobot_webhooks": "local_dev"
            },
            "version": "1.3.0-local",
            "integration_approach": "local_development",
            "features": {
                "local_development": True,
                "mock_data_fallback": True,
                "webhook_simulation": True,
                "fmcsa_integration": bool(FMCSA_API_KEY)
            }
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "degraded",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "environment": "development",
            "error": str(e)
        }

@app.get("/dashboard-metrics")
async def get_dashboard_metrics():
    """Get dashboard metrics for local development"""
    try:
        logger.info("📊 Calculating local development metrics...")
        metrics = await get_local_metrics()
        
        return {
            "success": True,
            "metrics": metrics
        }
        
    except Exception as e:
        logger.error(f"Dashboard metrics error: {e}")
        return {
            "success": False,
            "error": str(e)
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
    """Search for available loads based on criteria (using mock data for local dev)"""
    try:
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
        
    except Exception as e:
        logger.error(f"Load search error: {e}")
        raise HTTPException(status_code=500, detail="Load search failed")

@app.post("/negotiate-rate")
async def negotiate_rate(request: RateNegotiationRequest, api_key: str = Depends(verify_api_key)):
    """Handle rate negotiation for a specific load"""
    try:
        # Find load in mock data
        load = None
        for mock_load in MOCK_LOAD_DATA:
            if mock_load["load_id"] == request.load_id:
                load = mock_load
                break
        
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
        
        call_id = f"LOCAL_CALL_{request.load_id}_{int(datetime.now().timestamp())}"
        
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
        logger.info(f"📞 Processing call transcript for data extraction: {len(request.call_transcript)} chars")
        
        # Extract detailed information from transcript
        extracted_info = enhanced_extract_carrier_info_from_text(request.call_transcript)
        
        # Return in format expected by HappyRobot
        response_data = {
            "call_transcript": request.call_transcript,
            "mc_number": extracted_info.get("mc_number") or "",
            "company_name": extracted_info.get("company_name") or "",
            "equipment_type": extracted_info.get("equipment_type") or "",
            "original_rate": str(extracted_info.get("original_rate", 0)),
            "proposed_rate": str(extracted_info.get("proposed_rate", 0)),
            "final_outcome": extracted_info.get("final_outcome", "inquiry_only")
        }
        
        logger.info(f"✅ Successfully extracted call data: MC={response_data['mc_number']}, Company={response_data['company_name']}")
        
        return response_data
        
    except Exception as e:
        logger.error(f"Call data extraction error: {e}")
        raise HTTPException(status_code=500, detail="Call data extraction failed")

@app.post("/classify-call")
async def classify_call(request: CallClassificationRequest, api_key: str = Depends(verify_api_key)):
    """Classify call intent and outcome"""
    try:
        logger.info(f"🔍 Classifying call transcript: {len(request.call_transcript)} chars")
        
        # Get enhanced classification
        classification = enhanced_classify_call_outcome(request.call_transcript)
        
        # Return in format expected by HappyRobot
        response_data = {
            "call_transcript": request.call_transcript,
            "response_classification": classification["response_classification"],
            "response_reason": classification["response_reason"]
        }
        
        logger.info(f"✅ Call classified as: {response_data['response_classification']}")
        
        return response_data
        
    except Exception as e:
        logger.error(f"Call classification error: {e}")
        raise HTTPException(status_code=500, detail="Call classification failed")

# Local webhook simulation endpoint
@app.post("/webhooks/happyrobot/call-completed")
async def happyrobot_call_completed_local(request: Request):
    """LOCAL: Simulate HappyRobot webhook for development"""
    try:
        payload = await request.json()
        logger.info("🎯 Local webhook simulation received")

        # Normalise transcript
        transcript_raw = payload.get("transcript", payload.get("call_transcript", ""))
        if isinstance(transcript_raw, list):
            transcript_raw = " ".join(str(part) for part in transcript_raw)

        # Build call info
        call_info = {
            "happyrobot_call_id": payload.get(
                "call_id", f"LOCAL_{int(datetime.now().timestamp())}"
            ),
            "call_transcript": transcript_raw,
            "call_duration": float(payload.get("duration", payload.get("call_duration", 0))),
            "call_status": payload.get("status", "completed"),
            "carrier_mc": None,
            "company_name": None,
            "equipment_type": None,
            "original_rate": None,
            "proposed_rate": None,
            "final_outcome": "inquiry_only",
            "sentiment": "neutral",
            "created_at": datetime.now()
        }

        # Extract information from transcript
        if call_info["call_transcript"]:
            info = enhanced_extract_carrier_info_from_text(call_info["call_transcript"])
            call_info.update({
                "carrier_mc": info.get("mc_number"),
                "company_name": info.get("company_name"),
                "equipment_type": info.get("equipment_type"),
                "original_rate": info.get("original_rate"),
                "proposed_rate": info.get("proposed_rate"),
                "final_outcome": info.get("final_outcome", "inquiry_only"),
                "sentiment": enhanced_analyze_sentiment(call_info["call_transcript"])
            })

        # Store in memory for local development
        webhook_events.append(call_info)
        
        logger.info(f"✅ Local webhook processed: {call_info['happyrobot_call_id']} -> {call_info['final_outcome']}")

        return {
            "success": True,
            "message": "Local webhook processed successfully",
            "call_id": call_info["happyrobot_call_id"],
            "extracted_data": {
                "mc_number": call_info["carrier_mc"],
                "company_name": call_info["company_name"],
                "equipment_type": call_info["equipment_type"],
                "sentiment": call_info["sentiment"],
                "outcome": call_info["final_outcome"],
            },
            "stored_in": "memory_local_dev",
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"❌ Local webhook processing error: {e}")
        return {
            "success": False, 
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

@app.get("/dashboard/activity")
async def get_dashboard_activity():
    """Get recent activity for local development"""
    try:
        recent_activity = []
        
        # Build activity from in-memory webhook events
        for event in webhook_events[-20:]:
            recent_activity.append({
                "id": event["happyrobot_call_id"],
                "timestamp": event.get("created_at", datetime.now()).isoformat(),
                "carrier": event.get("company_name") or f"MC {event.get('carrier_mc')}" if event.get('carrier_mc') else "Unknown Carrier",
                "outcome": event["final_outcome"],
                "sentiment": event["sentiment"],
                "equipment": event["equipment_type"] or "Not specified",
                "source": "local_memory"
            })
        
        # Sort by timestamp
        recent_activity = sorted(recent_activity, key=lambda x: x["timestamp"], reverse=True)
        
        data_source = "webhook_memory" if len(recent_activity) > 0 else "mock"
    
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

# Serve dashboard
@app.get("/dashboard", response_class=HTMLResponse)
async def get_dashboard():
    """Serve the HTML dashboard"""
    try:
        # Try to read from dashboard directory first
        dashboard_path = Path("../dashboard/index.html")
        if dashboard_path.exists():
            with open(dashboard_path, 'r', encoding='utf-8') as file:
                return HTMLResponse(content=file.read())
        
        # Fallback to current directory
        index_path = Path("index.html")
        if index_path.exists():
            with open(index_path, 'r', encoding='utf-8') as file:
                return HTMLResponse(content=file.read())
        
        # Return basic HTML if no dashboard file found
        return HTMLResponse(content="""
        <h1>Local Development Dashboard</h1>
        <p>Dashboard file not found. Please ensure index.html exists.</p>
        <ul>
            <li><a href="/docs">API Documentation</a></li>
            <li><a href="/health">Health Check</a></li>
            <li><a href="/dashboard-metrics">Metrics API</a></li>
        </ul>
        """)
        
    except Exception as e:
        logger.error(f"Error serving dashboard: {e}")
        return HTMLResponse(content=f"<h1>Error</h1><p>{str(e)}</p>")

# Initialize database (optional for local development)
try:
    get_db_connection()
except Exception as e:
    logger.info("Local database initialization skipped (this is normal)")

if __name__ == "__main__":
    logger.info("🚛 Starting LOCAL DEVELOPMENT server...")
    logger.info("📋 Local endpoints:")
    logger.info("   • API: http://localhost:8000")
    logger.info("   • Docs: http://localhost:8000/docs")
    logger.info("   • Health: http://localhost:8000/health")
    logger.info("   • Dashboard: http://localhost:8000/dashboard")
    logger.info("")
    logger.info("🎯 Test webhook: curl -X POST http://localhost:8000/webhooks/happyrobot/call-completed")
    logger.info("")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        # reload=True,  # Auto-reload for development
        access_log=True,
        log_level="info"
    )