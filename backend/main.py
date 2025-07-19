from fastapi import FastAPI, HTTPException, Depends, Security, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, validator
from typing import List, Optional, Dict, Any
import requests
import json
import os
from datetime import datetime, timedelta
import sqlite3
import logging
from contextlib import contextmanager
import asyncio
import ssl
from urllib.parse import quote

# Enhanced logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('carrier_sales.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Carrier Sales API",
    description="AI-powered inbound carrier sales system for freight matching",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Security
security = HTTPBearer()

# Environment variables with validation
API_KEY = os.getenv("API_KEY", "secure-api-key-change-this-in-production")
FMCSA_API_KEY = os.getenv("FMCSA_API_KEY", "")
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
DEBUG = os.getenv("DEBUG", "false").lower() == "true"

if not API_KEY or API_KEY == "secure-api-key-change-this-in-production":
    logger.warning("Using default API key! Change this in production!")

if not FMCSA_API_KEY:
    logger.warning("No FMCSA API key provided! Using mock data.")

# Security middleware
app.add_middleware(TrustedHostMiddleware, allowed_hosts=["*"] if DEBUG else ["yourdomain.com"])

# Enhanced CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if DEBUG else ["https://yourdomain.com"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = datetime.now()
    
    # Log request
    logger.info(f"{request.method} {request.url} from {request.client.host}")
    
    response = await call_next(request)
    
    # Log response
    process_time = datetime.now() - start_time
    logger.info(f"{request.method} {request.url} -> {response.status_code} ({process_time.total_seconds():.3f}s)")
    
    return response

# Enhanced Data Models with validation
class Load(BaseModel):
    load_id: str
    origin: str
    destination: str
    pickup_datetime: str
    delivery_datetime: str
    equipment_type: str
    loadboard_rate: float
    notes: str
    weight: float
    commodity_type: str
    num_of_pieces: int
    miles: int
    dimensions: str

    @validator('loadboard_rate', 'weight')
    def validate_positive_numbers(cls, v):
        if v <= 0:
            raise ValueError('Must be positive')
        return v

class LoadSearchRequest(BaseModel):
    origin: Optional[str] = None
    destination: Optional[str] = None
    equipment_type: Optional[str] = None
    max_miles: Optional[int] = None

    @validator('max_miles')
    def validate_max_miles(cls, v):
        if v is not None and v <= 0:
            raise ValueError('Max miles must be positive')
        return v

class CarrierVerificationRequest(BaseModel):
    mc_number: str

    @validator('mc_number')
    def validate_mc_number(cls, v):
        # Remove any non-numeric characters and validate
        clean_mc = ''.join(filter(str.isdigit, v))
        if not clean_mc or len(clean_mc) < 4:
            raise ValueError('MC number must contain at least 4 digits')
        return clean_mc

class NegotiationRequest(BaseModel):
    load_id: str
    proposed_rate: float
    mc_number: str

    @validator('proposed_rate')
    def validate_proposed_rate(cls, v):
        if v <= 0:
            raise ValueError('Proposed rate must be positive')
        return v

class CallDataExtraction(BaseModel):
    transcript: str
    call_id: str

class CallClassification(BaseModel):
    transcript: str
    call_id: str

# Enhanced database setup with error handling
@contextmanager
def get_db():
    conn = None
    try:
        conn = sqlite3.connect('data/carrier_sales.db', timeout=30.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        yield conn
    except sqlite3.Error as e:
        logger.error(f"Database error: {e}")
        if conn:
            conn.rollback()
        raise HTTPException(status_code=500, detail="Database error")
    finally:
        if conn:
            conn.close()

def init_db():
    """Initialize database with enhanced schema"""
    try:
        with get_db() as conn:
            # Create loads table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS loads (
                    load_id TEXT PRIMARY KEY,
                    origin TEXT NOT NULL,
                    destination TEXT NOT NULL,
                    pickup_datetime TEXT NOT NULL,
                    delivery_datetime TEXT NOT NULL,
                    equipment_type TEXT NOT NULL,
                    loadboard_rate REAL NOT NULL CHECK(loadboard_rate > 0),
                    notes TEXT,
                    weight REAL NOT NULL CHECK(weight > 0),
                    commodity_type TEXT NOT NULL,
                    num_of_pieces INTEGER NOT NULL CHECK(num_of_pieces > 0),
                    miles INTEGER NOT NULL CHECK(miles > 0),
                    dimensions TEXT NOT NULL,
                    status TEXT DEFAULT 'available' CHECK(status IN ('available', 'booked', 'expired')),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create calls table with enhanced tracking
            conn.execute('''
                CREATE TABLE IF NOT EXISTS calls (
                    call_id TEXT PRIMARY KEY,
                    mc_number TEXT,
                    load_id TEXT,
                    transcript TEXT,
                    classification TEXT,
                    sentiment TEXT,
                    extracted_data TEXT,
                    negotiated_rate REAL,
                    final_outcome TEXT,
                    call_duration INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (load_id) REFERENCES loads (load_id)
                )
            ''')
            
            # Create negotiations table with round tracking
            conn.execute('''
                CREATE TABLE IF NOT EXISTS negotiations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    call_id TEXT NOT NULL,
                    load_id TEXT NOT NULL,
                    mc_number TEXT NOT NULL,
                    proposed_rate REAL NOT NULL CHECK(proposed_rate > 0),
                    counter_offer REAL,
                    round_number INTEGER NOT NULL CHECK(round_number > 0),
                    status TEXT DEFAULT 'pending' CHECK(status IN ('pending', 'accepted', 'rejected', 'countered')),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (call_id) REFERENCES calls (call_id),
                    FOREIGN KEY (load_id) REFERENCES loads (load_id)
                )
            ''')
            
            # Create indexes for performance
            conn.execute('CREATE INDEX IF NOT EXISTS idx_loads_equipment ON loads(equipment_type)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_loads_status ON loads(status)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_calls_mc ON calls(mc_number)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_negotiations_load ON negotiations(load_id)')
            
            conn.commit()
            logger.info("Database initialized successfully")
            
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        raise

# Security dependency with enhanced validation
def verify_api_key(credentials: HTTPAuthorizationCredentials = Security(security)):
    if not credentials or credentials.credentials != API_KEY:
        logger.warning(f"Unauthorized access attempt")
        raise HTTPException(
            status_code=401, 
            detail="Invalid or missing API key",
            headers={"WWW-Authenticate": "Bearer"}
        )
    return credentials

# Enhanced FMCSA API integration
async def verify_carrier_fmcsa(mc_number: str) -> Dict[str, Any]:
    """Verify carrier using real FMCSA API"""
    try:
        if not FMCSA_API_KEY:
            logger.warning("No FMCSA API key - using mock data")
            return {
                "mc_number": mc_number,
                "is_active": True,
                "company_name": f"Mock Carrier {mc_number}",
                "authority_status": "AUTHORIZED",
                "insurance_status": "CURRENT",
                "source": "mock"
            }

        # Real FMCSA API call
        url = f"https://mobile.fmcsa.dot.gov/qc/services/carriers/docket-number/{mc_number}"
        params = {"webKey": FMCSA_API_KEY}
        
        async with asyncio.timeout(10):  # 10 second timeout
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                try:
                    data = response.json()
                    
                    # Parse FMCSA response
                    carrier_data = data.get('content', [])
                    if not carrier_data:
                        return {
                            "mc_number": mc_number,
                            "is_active": False,
                            "error": "Carrier not found in FMCSA database"
                        }
                    
                    carrier = carrier_data[0]  # Get first result
                    
                    return {
                        "mc_number": mc_number,
                        "is_active": carrier.get("allowToOperate", "N") == "Y",
                        "company_name": carrier.get("legalName", "Unknown"),
                        "authority_status": "AUTHORIZED" if carrier.get("allowToOperate") == "Y" else "NOT_AUTHORIZED",
                        "insurance_status": "CURRENT" if carrier.get("insuranceRequired") else "UNKNOWN",
                        "dot_number": carrier.get("dotNumber"),
                        "physical_address": carrier.get("phyStreet", ""),
                        "source": "fmcsa_api"
                    }
                    
                except json.JSONDecodeError:
                    logger.error("Failed to parse FMCSA API response")
                    return {
                        "mc_number": mc_number,
                        "is_active": False,
                        "error": "Invalid response from FMCSA API"
                    }
            else:
                logger.error(f"FMCSA API error: {response.status_code}")
                return {
                    "mc_number": mc_number,
                    "is_active": False,
                    "error": f"FMCSA API returned {response.status_code}"
                }
                
    except asyncio.TimeoutError:
        logger.error("FMCSA API timeout")
        return {
            "mc_number": mc_number,
            "is_active": False,
            "error": "FMCSA API timeout"
        }
    except Exception as e:
        logger.error(f"FMCSA API error: {str(e)}")
        return {
            "mc_number": mc_number,
            "is_active": False,
            "error": "FMCSA API unavailable"
        }

# Enhanced sample data with more realistic loads
def insert_sample_loads():
    sample_loads = [
        {
            "load_id": "LD001",
            "origin": "Chicago, IL",
            "destination": "Atlanta, GA",
            "pickup_datetime": "2025-07-25 08:00:00",
            "delivery_datetime": "2025-07-26 18:00:00",
            "equipment_type": "Dry Van",
            "loadboard_rate": 2500.00,
            "notes": "Fragile electronics, handle with care. Appointment required for delivery.",
            "weight": 25000.0,
            "commodity_type": "Electronics",
            "num_of_pieces": 15,
            "miles": 585,
            "dimensions": "48x8.5x9"
        },
        {
            "load_id": "LD002",
            "origin": "Los Angeles, CA",
            "destination": "Phoenix, AZ",
            "pickup_datetime": "2025-07-24 14:00:00",
            "delivery_datetime": "2025-07-25 10:00:00",
            "equipment_type": "Flatbed",
            "loadboard_rate": 1800.00,
            "notes": "Construction materials, tarps provided. Crane unload available.",
            "weight": 35000.0,
            "commodity_type": "Steel",
            "num_of_pieces": 8,
            "miles": 372,
            "dimensions": "48x8.5x3"
        },
        {
            "load_id": "LD003",
            "origin": "Miami, FL",
            "destination": "New York, NY",
            "pickup_datetime": "2025-07-26 06:00:00",
            "delivery_datetime": "2025-07-28 16:00:00",
            "equipment_type": "Refrigerated",
            "loadboard_rate": 3200.00,
            "notes": "Temperature controlled -10°F. Continuous monitoring required.",
            "weight": 28000.0,
            "commodity_type": "Frozen Foods",
            "num_of_pieces": 120,
            "miles": 1280,
            "dimensions": "48x8.5x8"
        },
        {
            "load_id": "LD004",
            "origin": "Dallas, TX",
            "destination": "Denver, CO",
            "pickup_datetime": "2025-07-27 12:00:00",
            "delivery_datetime": "2025-07-28 20:00:00",
            "equipment_type": "Dry Van",
            "loadboard_rate": 2100.00,
            "notes": "Retail goods, cross-dock facility pickup and delivery.",
            "weight": 22000.0,
            "commodity_type": "Retail Goods",
            "num_of_pieces": 45,
            "miles": 781,
            "dimensions": "48x8.5x9"
        },
        {
            "load_id": "LD005",
            "origin": "Seattle, WA",
            "destination": "San Francisco, CA",
            "pickup_datetime": "2025-07-28 09:00:00",
            "delivery_datetime": "2025-07-29 17:00:00",
            "equipment_type": "Flatbed",
            "loadboard_rate": 1950.00,
            "notes": "Lumber shipment, straps and blocking provided.",
            "weight": 32000.0,
            "commodity_type": "Lumber",
            "num_of_pieces": 12,
            "miles": 808,
            "dimensions": "48x8.5x4"
        }
    ]
    
    try:
        with get_db() as conn:
            for load in sample_loads:
                conn.execute('''
                    INSERT OR REPLACE INTO loads 
                    (load_id, origin, destination, pickup_datetime, delivery_datetime, 
                     equipment_type, loadboard_rate, notes, weight, commodity_type, 
                     num_of_pieces, miles, dimensions)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', tuple(load.values()))
            conn.commit()
            logger.info("Sample loads inserted successfully")
    except Exception as e:
        logger.error(f"Failed to insert sample loads: {e}")

# API Endpoints

@app.on_event("startup")
async def startup_event():
    """Initialize application"""
    logger.info("Starting Carrier Sales API...")
    init_db()
    insert_sample_loads()
    logger.info("Carrier Sales API started successfully")

@app.get("/", tags=["Health"])
async def root():
    """API health check"""
    return {
        "message": "Carrier Sales API",
        "status": "active",
        "version": "1.0.0",
        "environment": ENVIRONMENT,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health", tags=["Health"])
async def health_check():
    """Detailed health check"""
    try:
        # Test database connection
        with get_db() as conn:
            conn.execute("SELECT 1").fetchone()
        
        return {
            "status": "healthy",
            "database": "connected",
            "fmcsa_api": "configured" if FMCSA_API_KEY else "mock_mode",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail="Service unhealthy")

@app.post("/verify-carrier", tags=["Carrier"])
async def verify_carrier(
    request: CarrierVerificationRequest,
    credentials: HTTPAuthorizationCredentials = Security(security)
):
    """Verify carrier using FMCSA API"""
    verify_api_key(credentials)
    
    try:
        logger.info(f"🔍 Verifying carrier MC#{request.mc_number}")
        result = await verify_carrier_fmcsa(request.mc_number)
        
        # Log verification result
        if result.get("is_active"):
            logger.info(f"Carrier MC#{request.mc_number} verified: {result.get('company_name')}")
        else:
            logger.warning(f"Carrier MC#{request.mc_number} verification failed: {result.get('error')}")
        
        return result
        
    except Exception as e:
        logger.error(f"Carrier verification error: {str(e)}")
        raise HTTPException(status_code=500, detail="Carrier verification failed")

@app.post("/search-loads", tags=["Loads"])
async def search_loads(
    request: LoadSearchRequest,
    credentials: HTTPAuthorizationCredentials = Security(security)
):
    """Search for available loads with advanced filtering"""
    verify_api_key(credentials)
    
    try:
        logger.info(f"Searching loads: {request.dict()}")
        
        with get_db() as conn:
            query = "SELECT * FROM loads WHERE status = 'available'"
            params = []
            
            if request.origin:
                query += " AND (origin LIKE ? OR origin LIKE ?)"
                params.extend([f"%{request.origin}%", f"%{request.origin.split(',')[0]}%"])
            
            if request.destination:
                query += " AND (destination LIKE ? OR destination LIKE ?)"
                params.extend([f"%{request.destination}%", f"%{request.destination.split(',')[0]}%"])
                
            if request.equipment_type:
                query += " AND equipment_type = ?"
                params.append(request.equipment_type)
                
            if request.max_miles:
                query += " AND miles <= ?"
                params.append(request.max_miles)
            
            # Order by best matches first
            query += " ORDER BY pickup_datetime ASC, loadboard_rate DESC LIMIT 5"
            
            cursor = conn.execute(query, params)
            loads = [dict(row) for row in cursor.fetchall()]
            
            logger.info(f"Found {len(loads)} matching loads")
            return {"loads": loads, "count": len(loads), "search_criteria": request.dict()}
            
    except Exception as e:
        logger.error(f"Load search error: {str(e)}")
        raise HTTPException(status_code=500, detail="Load search failed")

@app.post("/negotiate-rate", tags=["Negotiation"])
async def negotiate_rate(
    request: NegotiationRequest,
    credentials: HTTPAuthorizationCredentials = Security(security)
):
    """Handle rate negotiation with intelligent counter-offers"""
    verify_api_key(credentials)
    
    try:
        logger.info(f"Rate negotiation: Load {request.load_id}, MC#{request.mc_number}, Proposed: ${request.proposed_rate}")
        
        with get_db() as conn:
            # Get current load
            cursor = conn.execute("SELECT * FROM loads WHERE load_id = ?", (request.load_id,))
            load = cursor.fetchone()
            
            if not load:
                raise HTTPException(status_code=404, detail="Load not found")
            
            current_rate = load['loadboard_rate']
            proposed_rate = request.proposed_rate
            
            # Enhanced negotiation logic
            difference_percent = abs(current_rate - proposed_rate) / current_rate
            
            # Determine negotiation outcome
            if difference_percent <= 0.02:  # Within 2%
                decision = "accepted"
                counter_offer = proposed_rate
                status = "accepted"
            elif difference_percent <= 0.15:  # Within 15%
                # Smart counter-offer calculation
                if proposed_rate < current_rate:
                    # Split the difference, slightly favoring our rate
                    counter_offer = current_rate - (current_rate - proposed_rate) * 0.6
                else:
                    counter_offer = proposed_rate  # Accept if they're offering more
                decision = "counter_offer"
                status = "countered"
            else:
                # Large difference - firm counter
                counter_offer = current_rate * 0.95  # 5% discount maximum
                decision = "counter_offer"
                status = "countered"
            
            # Check negotiation round count
            round_count = conn.execute(
                "SELECT COUNT(*) FROM negotiations WHERE load_id = ? AND mc_number = ?",
                (request.load_id, request.mc_number)
            ).fetchone()[0]
            
            if round_count >= 3:
                decision = "final_offer"
                counter_offer = current_rate * 0.97  # Best final offer
                status = "final"
            
            # Record negotiation
            conn.execute('''
                INSERT INTO negotiations (load_id, mc_number, proposed_rate, counter_offer, round_number, status)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (request.load_id, request.mc_number, proposed_rate, counter_offer, round_count + 1, status))
            conn.commit()
            
            result = {
                "decision": decision,
                "counter_offer": round(counter_offer, 2),
                "original_rate": current_rate,
                "difference_percent": round(difference_percent * 100, 2),
                "round_number": round_count + 1,
                "max_rounds": 3
            }
            
            logger.info(f"Negotiation result: {decision} - Counter: ${counter_offer:.2f}")
            return result
            
    except Exception as e:
        logger.error(f"Negotiation error: {str(e)}")
        raise HTTPException(status_code=500, detail="Rate negotiation failed")

@app.post("/extract-call-data", tags=["AI Processing"])
async def extract_call_data(
    request: CallDataExtraction,
    credentials: HTTPAuthorizationCredentials = Security(security)
):
    """Extract relevant data from call transcript using enhanced NLP"""
    verify_api_key(credentials)
    
    try:
        logger.info(f"Extracting data from call {request.call_id}")
        
        # Enhanced extraction logic with regex patterns
        import re
        
        extracted_data = {
            "mc_number": None,
            "company_name": None,
            "load_interest": None,
            "proposed_rate": None,
            "equipment_preference": None,
            "contact_info": None,
            "origin_preference": None,
            "destination_preference": None
        }
        
        transcript_lower = request.transcript.lower()
        
        # Extract MC number with multiple patterns
        mc_patterns = [
            r'mc[:\s#]*(\d+)',
            r'motor carrier[:\s#]*(\d+)',
            r'docket[:\s#]*(\d+)'
        ]
        for pattern in mc_patterns:
            mc_match = re.search(pattern, transcript_lower)
            if mc_match:
                extracted_data["mc_number"] = mc_match.group(1)
                break
        
        # Extract company name
        company_patterns = [
            r'(?:company|trucking|logistics|transport)[\s\w]*(?:llc|inc|corp|ltd)',
            r'(\w+\s+trucking)',
            r'(\w+\s+logistics)'
        ]
        for pattern in company_patterns:
            company_match = re.search(pattern, transcript_lower)
            if company_match:
                extracted_data["company_name"] = company_match.group(0).title()
                break
        
        # Extract rate mentions with currency handling
        rate_patterns = [
            r'\$?(\d{1,2}[,.]?\d{3})',  # $2,500 or $2500 or 2500
            r'(\d+)\s*dollars?',
            r'rate.*?(\d{1,2}[,.]?\d{3})'
        ]
        for pattern in rate_patterns:
            rate_match = re.search(pattern, transcript_lower)
            if rate_match:
                rate_str = rate_match.group(1).replace(',', '')
                try:
                    extracted_data["proposed_rate"] = float(rate_str)
                    break
                except ValueError:
                    continue
        
        # Equipment type extraction
        equipment_types = {
            'dry van': ['dry van', 'van', 'box truck'],
            'flatbed': ['flatbed', 'flat bed', 'stepdeck'],
            'refrigerated': ['reefer', 'refrigerated', 'cold', 'frozen'],
            'tanker': ['tanker', 'liquid', 'fuel']
        }
        
        for eq_type, keywords in equipment_types.items():
            if any(keyword in transcript_lower for keyword in keywords):
                extracted_data["equipment_preference"] = eq_type
                break
        
        # Location preferences
        state_abbrevs = ['AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA', 
                        'HI', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD',
                        'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ',
                        'NM', 'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC',
                        'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WV', 'WI', 'WY']
        
        # Find state mentions for origin/destination preferences
        for state in state_abbrevs:
            if state.lower() in transcript_lower or f' {state.lower()} ' in transcript_lower:
                if not extracted_data["origin_preference"]:
                    extracted_data["origin_preference"] = state
                elif not extracted_data["destination_preference"]:
                    extracted_data["destination_preference"] = state
                    break
        
        # Store in database
        with get_db() as conn:
            conn.execute('''
                INSERT OR REPLACE INTO calls (call_id, transcript, extracted_data, mc_number)
                VALUES (?, ?, ?, ?)
            ''', (request.call_id, request.transcript, json.dumps(extracted_data), extracted_data.get("mc_number")))
            conn.commit()
        
        logger.info(f"Data extracted successfully from call {request.call_id}")
        return {"extracted_data": extracted_data, "call_id": request.call_id}
        
    except Exception as e:
        logger.error(f"Data extraction error: {str(e)}")
        raise HTTPException(status_code=500, detail="Data extraction failed")

@app.post("/classify-call", tags=["AI Processing"])
async def classify_call(
    request: CallClassification,
    credentials: HTTPAuthorizationCredentials = Security(security)
):
    """Classify call outcome and sentiment with enhanced AI logic"""
    verify_api_key(credentials)
    
    try:
        logger.info(f"Classifying call {request.call_id}")
        
        transcript_lower = request.transcript.lower()
        
        # Enhanced classification with weighted scoring
        classification_keywords = {
            "load_booked": ["book", "take", "accept", "agree", "deal", "yes", "perfect", "great"],
            "negotiation": ["counter", "negotiate", "different rate", "how about", "what about", "can you do"],
            "not_interested": ["not interested", "no thanks", "pass", "can't do", "too low", "no way"],
            "inquiry_only": ["information", "just checking", "what do you have", "tell me about"]
        }
        
        scores = {}
        for category, keywords in classification_keywords.items():
            score = sum(1 for keyword in keywords if keyword in transcript_lower)
            scores[category] = score
        
        # Determine classification
        if scores["load_booked"] > 0 and any(word in transcript_lower for word in ["transfer", "sales", "book"]):
            classification = "load_booked"
        elif scores["negotiation"] > scores["not_interested"]:
            classification = "negotiation"
        elif scores["not_interested"] > 0:
            classification = "not_interested"
        else:
            classification = "inquiry_only"
        
        # Enhanced sentiment analysis
        positive_indicators = ['great', 'good', 'excellent', 'perfect', 'yes', 'sure', 'sounds good', 'thank you']
        negative_indicators = ['bad', 'no', 'terrible', 'awful', 'wrong', 'disappointed', 'frustrated']
        neutral_indicators = ['okay', 'alright', 'maybe', 'possibly', 'perhaps']
        
        positive_score = sum(2 if word in transcript_lower else 0 for word in positive_indicators)
        negative_score = sum(2 if word in transcript_lower else 0 for word in negative_indicators)
        neutral_score = sum(1 if word in transcript_lower else 0 for word in neutral_indicators)
        
        if positive_score > negative_score + 1:
            sentiment = "positive"
            confidence = min(0.9, 0.6 + (positive_score * 0.1))
        elif negative_score > positive_score + 1:
            sentiment = "negative"
            confidence = min(0.9, 0.6 + (negative_score * 0.1))
        else:
            sentiment = "neutral"
            confidence = 0.7
        
        # Store in database
        with get_db() as conn:
            conn.execute('''
                UPDATE calls SET classification = ?, sentiment = ? WHERE call_id = ?
            ''', (classification, sentiment, request.call_id))
            conn.commit()
        
        result = {
            "classification": classification,
            "sentiment": sentiment,
            "confidence": round(confidence, 2),
            "scores": scores,
            "call_id": request.call_id
        }
        
        logger.info(f"Call classified: {classification}/{sentiment} (confidence: {confidence:.2f})")
        return result
        
    except Exception as e:
        logger.error(f"Call classification error: {str(e)}")
        raise HTTPException(status_code=500, detail="Call classification failed")

@app.get("/dashboard-metrics", tags=["Analytics"])
async def get_dashboard_metrics(
    credentials: HTTPAuthorizationCredentials = Security(security)
):
    """Get comprehensive metrics for dashboard"""
    verify_api_key(credentials)
    
    try:
        with get_db() as conn:
            # Total calls
            total_calls = conn.execute("SELECT COUNT(*) FROM calls").fetchone()[0]
            
            # Calls by classification
            classifications = conn.execute('''
                SELECT classification, COUNT(*) as count 
                FROM calls 
                WHERE classification IS NOT NULL 
                GROUP BY classification
            ''').fetchall()
            
            # Sentiment distribution
            sentiments = conn.execute('''
                SELECT sentiment, COUNT(*) as count 
                FROM calls 
                WHERE sentiment IS NOT NULL 
                GROUP BY sentiment
            ''').fetchall()
            
            # Average negotiated rates
            avg_rate = conn.execute('''
                SELECT AVG(counter_offer) 
                FROM negotiations
                WHERE counter_offer IS NOT NULL
            ''').fetchone()[0]
            
            # Recent calls (last 7 days)
            recent_calls = conn.execute('''
                SELECT COUNT(*) 
                FROM calls 
                WHERE created_at >= datetime('now', '-7 days')
            ''').fetchone()[0]
            
            # Conversion metrics
            booked_calls = sum(1 for c in classifications if c[0] == 'load_booked')
            conversion_rate = (booked_calls / max(total_calls, 1)) * 100
            
            # Load metrics
            total_loads = conn.execute("SELECT COUNT(*) FROM loads").fetchone()[0]
            available_loads = conn.execute("SELECT COUNT(*) FROM loads WHERE status = 'available'").fetchone()[0]
            
            # Top performing equipment types
            equipment_performance = conn.execute('''
                SELECT l.equipment_type, COUNT(c.call_id) as call_count
                FROM loads l
                LEFT JOIN calls c ON l.load_id = c.load_id
                GROUP BY l.equipment_type
                ORDER BY call_count DESC
            ''').fetchall()
            
            return {
                "summary": {
                    "total_calls": total_calls,
                    "recent_calls": recent_calls,
                    "conversion_rate": round(conversion_rate, 2),
                    "average_negotiated_rate": round(avg_rate or 0, 2),
                    "total_loads": total_loads,
                    "available_loads": available_loads
                },
                "classifications": [{"type": row[0], "count": row[1]} for row in classifications],
                "sentiments": [{"type": row[0], "count": row[1]} for row in sentiments],
                "equipment_performance": [{"type": row[0], "calls": row[1]} for row in equipment_performance],
                "timestamp": datetime.now().isoformat()
            }
            
    except Exception as e:
        logger.error(f"Dashboard metrics error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve metrics")

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request: Request, exc):
    return JSONResponse(
        status_code=404,
        content={"detail": "Endpoint not found", "path": str(request.url)}
    )

@app.exception_handler(500)
async def internal_error_handler(request: Request, exc):
    logger.error(f"Internal server error: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )

if __name__ == "__main__":
    import uvicorn
    
    # SSL configuration for production
    ssl_context = None
    if ENVIRONMENT == "production":
        try:
            ssl_context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
            ssl_context.load_cert_chain("cert.pem", "key.pem")
            logger.info("SSL enabled for production")
        except FileNotFoundError:
            logger.warning("SSL certificates not found, running without HTTPS")
    
    # Run server
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        ssl_keyfile="key.pem" if ssl_context else None,
        ssl_certfile="cert.pem" if ssl_context else None,
        access_log=True,
        log_level="info"
    )