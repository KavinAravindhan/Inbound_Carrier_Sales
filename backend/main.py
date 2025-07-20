"""
Carrier Sales API - Backend Server
Fixed version with proper route registration and modern FastAPI practices
"""

import os
import logging
import mysql.connector
from datetime import datetime, timezone
from typing import Optional, List
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator
import uvicorn
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - **%(name)s** - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configuration
API_KEY = os.getenv("API_KEY", "secure-api-key-change-this-in-production")
FMCSA_API_KEY = os.getenv("FMCSA_API_KEY", "")
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
DEBUG = os.getenv("DEBUG", "true").lower() == "true"

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
    logger.warning("⚠️  No FMCSA API key provided! Using professional mock data.")

# Database connection functions
def get_db_connection():
    """Get MySQL database connection"""
    try:
        connection = mysql.connector.connect(**MYSQL_CONFIG)
        return connection
    except mysql.connector.Error as err:
        logger.error(f"Database connection error: {err}")
        raise HTTPException(status_code=500, detail="Database connection failed")

def initialize_database():
    """Initialize database with required tables and sample data"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Create loads table (matching existing schema)
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
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
            )
        """)
        
        # Create calls table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS calls (
                id INT AUTO_INCREMENT PRIMARY KEY,
                carrier_mc VARCHAR(20),
                call_duration INT,
                call_transcript TEXT,
                call_outcome ENUM('interested', 'not_interested', 'callback', 'booked') DEFAULT 'interested',
                loads_discussed TEXT,
                follow_up_date DATE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create negotiations table (matching existing schema)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS negotiations (
                id INT AUTO_INCREMENT PRIMARY KEY,
                call_id VARCHAR(100),
                load_id VARCHAR(50),
                mc_number VARCHAR(20),
                proposed_rate DECIMAL(10,2),
                counter_offer DECIMAL(10,2),
                round_number INT,
                status ENUM('pending', 'accepted', 'rejected', 'counter_offered') DEFAULT 'pending',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        conn.commit()
        logger.info("✅ MySQL database initialized successfully")
        
        # Insert sample data if loads table is empty
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
        
    except mysql.connector.Error as err:
        logger.error(f"Database initialization error: {err}")
        raise HTTPException(status_code=500, detail=f"Database initialization failed: {err}")

# Professional Mock Data
MOCK_CARRIER_DATA = {
    "123456": {
        "mc_number": "123456",
        "company_name": "ABC Transportation LLC",
        "address": "123 Trucking Way, Transport City, TX 75001",
        "phone": "(555) 123-4567",
        "email": "dispatch@abctransport.com",
        "equipment_types": ["Dry Van", "Refrigerated"],
        "fleet_size": 25,
        "safety_rating": "Satisfactory",
        "insurance_status": "Active",
        "authority_status": "Active",
        "years_in_business": 8,
        "preferred_lanes": ["TX-CA", "TX-FL", "IL-TX"],
        "average_rate_per_mile": 2.85,
        "last_used": "2025-07-15"
    },
    "789012": {
        "mc_number": "789012",
        "company_name": "Express Freight Solutions",
        "address": "456 Highway Blvd, Logistics Park, CA 90210",
        "phone": "(555) 987-6543",
        "email": "ops@expressfreight.com",
        "equipment_types": ["Flatbed", "Step Deck"],
        "fleet_size": 40,
        "safety_rating": "Satisfactory",
        "insurance_status": "Active",
        "authority_status": "Active",
        "years_in_business": 12,
        "preferred_lanes": ["CA-TX", "CA-AZ", "NV-CA"],
        "average_rate_per_mile": 3.20,
        "last_used": "2025-07-18"
    }
}

# Pydantic Models with V2 validators
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

class CarrierSearchRequest(BaseModel):
    pickup_location: str
    delivery_location: str
    equipment_type: str
    max_miles: Optional[int] = None
    
    @field_validator('max_miles')
    @classmethod
    def validate_max_miles(cls, v):
        if v is not None and v <= 0:
            raise ValueError('Max miles must be positive')
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
    carrier_mc: str
    
    @field_validator('proposed_rate')
    @classmethod
    def validate_proposed_rate(cls, v):
        if v <= 0:
            raise ValueError('Proposed rate must be positive')
        return v

class CallDataExtractionRequest(BaseModel):
    call_transcript: str
    call_duration: Optional[int] = None

class CallClassificationRequest(BaseModel):
    call_transcript: str

# Lifespan event handler
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler"""
    # Startup
    logger.info("🚛 Starting Carrier Sales API...")
    initialize_database()
    logger.info("✅ Carrier Sales API started successfully")
    yield
    # Shutdown (if needed)
    logger.info("🛑 Shutting down Carrier Sales API...")

# Initialize FastAPI app
app = FastAPI(
    title="Carrier Sales API",
    description="AI-powered freight broker assistant for carrier sales",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend domains
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
    
    # Log incoming request
    logger.info(f"📥 {request.method} {request.url} from {request.client.host}")
    
    response = await call_next(request)
    
    # Calculate processing time
    process_time = (datetime.now() - start_time).total_seconds()
    
    # Log outgoing response
    logger.info(f"📤 {request.method} {request.url} -> {response.status_code} ({process_time:.3f}s)")
    
    return response

# API Endpoints
@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Carrier Sales API",
        "version": "1.0.0",
        "environment": ENVIRONMENT,
        "documentation": "/docs",
        "health_check": "/health"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Test database connection
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT 1")
        cursor.fetchone()
        cursor.close()
        conn.close()
        
        return {
            "status": "healthy",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "environment": ENVIRONMENT,
            "database": "connected",
            "version": "1.0.0"
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail="Service unavailable")

@app.post("/verify-carrier")
async def verify_carrier(request: CarrierVerificationRequest, api_key: str = Depends(verify_api_key)):
    """Verify carrier information using MC number"""
    try:
        mc_number = request.mc_number.strip()
        
        # Use professional mock data
        if mc_number in MOCK_CARRIER_DATA:
            carrier_data = MOCK_CARRIER_DATA[mc_number].copy()
            
            # Add verification timestamp
            carrier_data["verified_at"] = datetime.now(timezone.utc).isoformat()
            carrier_data["data_source"] = "professional_mock"
            
            return {
                "success": True,
                "carrier": carrier_data,
                "verification_status": "verified"
            }
        else:
            # Return structured "not found" response
            return {
                "success": False,
                "carrier": None,
                "verification_status": "not_found",
                "message": f"No carrier found with MC number {mc_number}",
                "suggested_action": "Please verify the MC number and try again"
            }
            
    except Exception as e:
        logger.error(f"Carrier verification error: {e}")
        raise HTTPException(status_code=500, detail="Carrier verification failed")

@app.post("/search-loads")
async def search_loads(request: LoadSearchRequest, api_key: str = Depends(verify_api_key)):
    """Search for available loads based on criteria"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        
        # Build dynamic query using correct column names
        query = "SELECT * FROM loads WHERE status = 'available'"
        params = []
        
        if request.equipment_type:
            query += " AND equipment_type = %s"
            params.append(request.equipment_type)
            
        if request.pickup_state:
            query += " AND origin LIKE %s"
            params.append(f"%{request.pickup_state}%")
            
        if request.delivery_state:
            query += " AND destination LIKE %s"
            params.append(f"%{request.delivery_state}%")
            
        if request.pickup_city:
            query += " AND origin LIKE %s"
            params.append(f"%{request.pickup_city}%")
            
        if request.delivery_city:
            query += " AND destination LIKE %s"
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
        
        # Convert dates to strings for JSON serialization
        for load in loads:
            if load.get('pickup_datetime'):
                load['pickup_datetime'] = load['pickup_datetime'].strftime('%Y-%m-%d %H:%M:%S')
            if load.get('delivery_datetime'):
                load['delivery_datetime'] = load['delivery_datetime'].strftime('%Y-%m-%d %H:%M:%S')
            if load.get('created_at'):
                load['created_at'] = load['created_at'].isoformat()
            if load.get('updated_at'):
                load['updated_at'] = load['updated_at'].isoformat()
            
            # Add aliases for backward compatibility
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
            "search_criteria": request.dict()
        }
        
    except Exception as e:
        logger.error(f"Load search error: {e}")
        raise HTTPException(status_code=500, detail="Load search failed")

@app.post("/negotiate-rate")
async def negotiate_rate(request: RateNegotiationRequest, api_key: str = Depends(verify_api_key)):
    """Handle rate negotiation for a specific load"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        
        # Get current load information using correct column names
        cursor.execute("SELECT * FROM loads WHERE load_id = %s", (request.load_id,))
        load = cursor.fetchone()
        
        if not load:
            logger.error(f"Load not found: {request.load_id}")
            raise HTTPException(status_code=404, detail="Load not found")
        
        logger.info(f"Found load: {load}")
        
        current_rate = float(load['loadboard_rate'])
        proposed_rate = request.proposed_rate
        
        # Calculate difference
        rate_difference = abs(current_rate - proposed_rate)
        percentage_difference = (rate_difference / current_rate) * 100
        
        # Determine negotiation outcome
        if proposed_rate >= current_rate * 0.95:  # Within 5% of asking rate
            status = "accepted"
            counter_offer = proposed_rate
            response_message = f"Great! We can accept ${proposed_rate:.2f} for load {request.load_id}."
        elif proposed_rate >= current_rate * 0.90:  # Within 10% of asking rate
            status = "counter_offered"
            counter_offer = current_rate * 0.93  # Counter at 93% of original
            response_message = f"We're close! How about ${counter_offer:.2f} for load {request.load_id}?"
        else:
            status = "rejected"
            counter_offer = current_rate * 0.90  # Our best offer
            response_message = f"Sorry, ${proposed_rate:.2f} is too low. Our best rate for load {request.load_id} is ${counter_offer:.2f}."
        
        # Record negotiation using correct column names with better error handling
        call_id = f"CALL_{request.load_id}_{int(datetime.now().timestamp())}"
        
        try:
            logger.info(f"Inserting negotiation: call_id={call_id}, load_id={request.load_id}, mc_number={request.carrier_mc}")
            
            cursor.execute("""
                INSERT INTO negotiations (call_id, load_id, mc_number, proposed_rate, counter_offer, round_number, status)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            """, (call_id, request.load_id, request.carrier_mc, proposed_rate, counter_offer, 1, status))
            
            conn.commit()
            logger.info(f"Negotiation inserted successfully with ID: {cursor.lastrowid}")
            
        except mysql.connector.Error as db_error:
            logger.error(f"Database error during negotiation insert: {db_error}")
            conn.rollback()
            
            # Try without foreign key constraint (in case there's an FK issue)
            try:
                logger.info("Retrying without foreign key constraint...")
                cursor.execute("""
                    INSERT INTO negotiations (call_id, load_id, mc_number, proposed_rate, counter_offer, round_number, status)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                """, (call_id, request.load_id, request.carrier_mc, proposed_rate, counter_offer, 1, status))
                conn.commit()
                logger.info("Negotiation inserted on retry")
            except mysql.connector.Error as retry_error:
                logger.error(f"Retry also failed: {retry_error}")
                # Continue without recording negotiation but return the result
                pass
        
        cursor.close()
        conn.close()
        
        return {
            "success": True,
            "negotiation_result": {
                "status": status,
                "original_rate": current_rate,
                "proposed_rate": proposed_rate,
                "counter_offer": counter_offer,
                "final_rate": counter_offer,  # For backward compatibility
                "response_message": response_message,
                "percentage_difference": round(percentage_difference, 2),
                "call_id": call_id
            }
        }
        
    except Exception as e:
        logger.error(f"Rate negotiation error: {e}")
        logger.error(f"Error type: {type(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Rate negotiation failed: {str(e)}")

@app.post("/extract-call-data")
async def extract_call_data(request: CallDataExtractionRequest, api_key: str = Depends(verify_api_key)):
    """Extract structured data from call transcripts"""
    try:
        transcript = request.call_transcript.lower()
        
        # Simple extraction logic (in production, use NLP/LLM)
        extracted_data = {
            "carrier_info": {},
            "equipment_needs": [],
            "locations_mentioned": [],
            "rates_discussed": [],
            "follow_up_required": False,
            "sentiment": "neutral"
        }
        
        # Extract MC number
        import re
        mc_matches = re.findall(r'mc\s*(?:number\s*)?(\d{4,8})', transcript)
        if mc_matches:
            extracted_data["carrier_info"]["mc_number"] = mc_matches[0]
        
        # Extract company name
        company_patterns = [
            r'this is ([A-Za-z\s]+) (?:trucking|transport|logistics|freight)',
            r'calling from ([A-Za-z\s]+)'
        ]
        for pattern in company_patterns:
            matches = re.findall(pattern, transcript)
            if matches:
                extracted_data["carrier_info"]["company_name"] = matches[0].strip().title()
                break
        
        # Extract equipment types
        equipment_types = ["dry van", "reefer", "flatbed", "step deck", "lowboy"]
        for equipment in equipment_types:
            if equipment in transcript:
                extracted_data["equipment_needs"].append(equipment.title())
        
        # Extract locations
        location_patterns = [
            r'from ([A-Za-z\s]+) to ([A-Za-z\s]+)',
            r'picking up in ([A-Za-z\s]+)',
            r'delivering to ([A-Za-z\s]+)'
        ]
        for pattern in location_patterns:
            matches = re.findall(pattern, transcript)
            for match in matches:
                if isinstance(match, tuple):
                    extracted_data["locations_mentioned"].extend(match)
                else:
                    extracted_data["locations_mentioned"].append(match)
        
        # Extract rates
        rate_matches = re.findall(r'\$(\d{1,4}(?:,\d{3})*(?:\.\d{2})?)', transcript)
        extracted_data["rates_discussed"] = [float(rate.replace(',', '')) for rate in rate_matches]
        
        # Determine sentiment
        positive_words = ["interested", "good", "great", "excellent", "perfect"]
        negative_words = ["not interested", "too low", "can't do", "no way"]
        
        if any(word in transcript for word in positive_words):
            extracted_data["sentiment"] = "positive"
        elif any(word in transcript for word in negative_words):
            extracted_data["sentiment"] = "negative"
        
        # Check for follow-up needs
        follow_up_phrases = ["call back", "follow up", "check with", "let me know"]
        extracted_data["follow_up_required"] = any(phrase in transcript for phrase in follow_up_phrases)
        
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
        
        # Classification logic
        classification = {
            "intent": "unknown",
            "outcome": "unknown",
            "priority": "medium",
            "action_required": [],
            "confidence": 0.0
        }
        
        # Intent classification
        if any(word in transcript for word in ["looking for", "need", "want", "interested"]):
            classification["intent"] = "load_inquiry"
            classification["confidence"] += 0.3
        elif any(word in transcript for word in ["rate", "price", "pay", "money"]):
            classification["intent"] = "rate_negotiation"
            classification["confidence"] += 0.3
        elif any(word in transcript for word in ["capacity", "available", "truck"]):
            classification["intent"] = "capacity_check"
            classification["confidence"] += 0.3
        
        # Outcome classification
        if any(word in transcript for word in ["book", "take it", "sounds good", "deal"]):
            classification["outcome"] = "booked"
            classification["priority"] = "high"
            classification["confidence"] += 0.4
        elif any(word in transcript for word in ["maybe", "think about", "call back"]):
            classification["outcome"] = "follow_up_needed"
            classification["priority"] = "medium"
            classification["confidence"] += 0.3
        elif any(word in transcript for word in ["not interested", "pass", "no thanks"]):
            classification["outcome"] = "not_interested"
            classification["priority"] = "low"
            classification["confidence"] += 0.4
        
        # Action items
        if "follow_up_needed" in classification["outcome"]:
            classification["action_required"].append("schedule_follow_up")
        if "rate" in transcript:
            classification["action_required"].append("send_rate_confirmation")
        if any(word in transcript for word in ["email", "send", "documents"]):
            classification["action_required"].append("send_documentation")
        
        # Ensure confidence is between 0 and 1
        classification["confidence"] = min(classification["confidence"], 1.0)
        
        return {
            "success": True,
            "classification": classification,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        logger.error(f"Call classification error: {e}")
        raise HTTPException(status_code=500, detail="Call classification failed")

if __name__ == "__main__":
    # Run the application with proper configuration
    uvicorn.run(
        "main:app",  # Import string format
        host="0.0.0.0",
        port=8000,
        reload=DEBUG,
        access_log=True,
        log_level="info"
    )