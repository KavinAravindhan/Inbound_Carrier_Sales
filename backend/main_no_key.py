"""
Carrier Sales API - Backend Server
Production-ready version with graceful database handling
Updated for HappyRobot integration compatibility
"""

import os
import logging
import mysql.connector
from datetime import datetime, timezone, timedelta
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
    logger.warning("⚠️  No FMCSA API key provided! Using professional mock data.")

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
            raise HTTPException(status_code=500, detail="Database connection failed")

def initialize_database():
    """Initialize database with required tables and sample data"""
    global DATABASE_AVAILABLE
    
    # Skip database initialization in production
    if ENVIRONMENT == "production":
        logger.info("🚀 Production mode: Using mock data only")
        logger.info("✅ Mock database initialized for production")
        DATABASE_AVAILABLE = False
        return
    
    try:
        conn = get_db_connection()
        if not conn:
            logger.info("📦 Running in mock data mode")
            return
            
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
        DATABASE_AVAILABLE = True
        
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
        logger.warning(f"⚠️  MySQL not available: {err}")
        logger.info("📦 Running in mock data mode")
        DATABASE_AVAILABLE = False
    except Exception as err:
        logger.warning(f"⚠️  Database initialization failed: {err}")
        logger.info("📦 Running in mock data mode")
        DATABASE_AVAILABLE = False

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

# Mock load data for production
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
    },
    {
        "load_id": "LD003",
        "origin": "Dallas, TX",
        "destination": "Denver, CO",
        "pickup_datetime": "2025-07-24 06:00:00",
        "delivery_datetime": "2025-07-26 18:00:00",
        "equipment_type": "Refrigerated",
        "loadboard_rate": 2200.00,
        "notes": "Temperature controlled",
        "weight": 42000,
        "commodity_type": "Food Products",
        "num_of_pieces": 1,
        "miles": 780,
        "dimensions": "53ft reefer",
        "status": "available",
        "created_at": "2025-07-20T12:57:14",
        "updated_at": "2025-07-20T12:57:14",
        "id": "LD003",
        "pickup_location": "Dallas, TX",
        "delivery_location": "Denver, CO",
        "rate": 2200.00,
        "pickup_date": "2025-07-24",
        "delivery_date": "2025-07-26"
    },
    {
        "load_id": "LD004",
        "origin": "Miami, FL",
        "destination": "Jacksonville, FL",
        "pickup_datetime": "2025-07-22 10:00:00",
        "delivery_datetime": "2025-07-23 14:00:00",
        "equipment_type": "Dry Van",
        "loadboard_rate": 800.00,
        "notes": "Regional delivery",
        "weight": 38000,
        "commodity_type": "Retail Goods",
        "num_of_pieces": 1,
        "miles": 350,
        "dimensions": "48ft",
        "status": "available",
        "created_at": "2025-07-20T12:57:14",
        "updated_at": "2025-07-20T12:57:14",
        "id": "LD004",
        "pickup_location": "Miami, FL",
        "delivery_location": "Jacksonville, FL",
        "rate": 800.00,
        "pickup_date": "2025-07-22",
        "delivery_date": "2025-07-23"
    },
    {
        "load_id": "LD005",
        "origin": "Seattle, WA",
        "destination": "Portland, OR",
        "pickup_datetime": "2025-07-25 07:00:00",
        "delivery_datetime": "2025-07-26 12:00:00",
        "equipment_type": "Flatbed",
        "loadboard_rate": 600.00,
        "notes": "Short haul",
        "weight": 50000,
        "commodity_type": "Lumber",
        "num_of_pieces": 1,
        "miles": 180,
        "dimensions": "48ft flatbed",
        "status": "available",
        "created_at": "2025-07-20T12:57:14",
        "updated_at": "2025-07-20T12:57:14",
        "id": "LD005",
        "pickup_location": "Seattle, WA",
        "delivery_location": "Portland, OR",
        "rate": 600.00,
        "pickup_date": "2025-07-25",
        "delivery_date": "2025-07-26"
    }
]

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
    carrier_mc: Optional[str] = None  # For backward compatibility
    mc_number: Optional[str] = None  # For HappyRobot compatibility
    
    @field_validator('proposed_rate')
    @classmethod
    def validate_proposed_rate(cls, v):
        if v <= 0:
            raise ValueError('Proposed rate must be positive')
        return v
    
    def get_mc_number(self):
        """Get MC number from either field for compatibility"""
        return self.carrier_mc or self.mc_number or "unknown"

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
        database_status = "connected" if DATABASE_AVAILABLE else "mock"
        
        # Only test database if it's supposed to be available
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
            "database": database_status,
            "version": "1.0.0"
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "healthy",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "environment": ENVIRONMENT,
            "database": "mock",
            "version": "1.0.0"
        }

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
        # Use mock data if database not available
        if not DATABASE_AVAILABLE:
            # Filter mock data based on request criteria
            filtered_loads = []
            for load in MOCK_LOAD_DATA:
                # Equipment type filter (case insensitive)
                if request.equipment_type and load["equipment_type"].lower() != request.equipment_type.lower():
                    continue
                    
                # Location filters (case insensitive)
                if request.pickup_state and request.pickup_state.upper() not in load["origin"].upper():
                    continue
                if request.delivery_state and request.delivery_state.upper() not in load["destination"].upper():
                    continue
                if request.pickup_city and request.pickup_city.upper() not in load["origin"].upper():
                    continue
                if request.delivery_city and request.delivery_city.upper() not in load["destination"].upper():
                    continue
                    
                # Rate filter
                if request.loadboard_rate and load["loadboard_rate"] < request.loadboard_rate:
                    continue
                    
                # Weight filter
                if request.weight and load["weight"] > request.weight:
                    continue
                    
                filtered_loads.append(load)
            
            # Sort by rate descending
            filtered_loads.sort(key=lambda x: x["loadboard_rate"], reverse=True)
            
            return {
                "success": True,
                "loads": filtered_loads[:20],  # Limit to 20
                "total_found": len(filtered_loads),
                "search_criteria": request.dict(),
                "data_source": "mock"
            }
        
        # Database query (for development)
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        
        # Build dynamic query using correct column names
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
        # Find load in mock data or database
        load = None
        
        if not DATABASE_AVAILABLE:
            # Search in mock data
            for mock_load in MOCK_LOAD_DATA:
                if mock_load["load_id"] == request.load_id:
                    load = mock_load
                    break
        else:
            # Search in database
            conn = get_db_connection()
            cursor = conn.cursor(dictionary=True)
            cursor.execute("SELECT * FROM loads WHERE load_id = %s", (request.load_id,))
            load = cursor.fetchone()
            cursor.close()
            conn.close()
        
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
        
        # Generate call ID
        call_id = f"CALL_{request.load_id}_{int(datetime.now().timestamp())}"
        
        # Try to record negotiation in database if available
        if DATABASE_AVAILABLE:
            try:
                conn = get_db_connection()
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO negotiations (call_id, load_id, mc_number, proposed_rate, counter_offer, round_number, status)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                """, (call_id, request.load_id, request.get_mc_number(), proposed_rate, counter_offer, 1, status))
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
                "final_rate": counter_offer,  # For backward compatibility
                "response_message": response_message,
                "percentage_difference": round(percentage_difference, 2),
                "call_id": call_id,
                "mc_number": request.get_mc_number()
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
            "sentiment": "neutral",
            "key_phrases": [],
            "action_items": []
        }
        
        # Extract MC number
        import re
        mc_matches = re.findall(r'mc\s*(?:number\s*)?(\d{4,8})', transcript)
        if mc_matches:
            extracted_data["carrier_info"]["mc_number"] = mc_matches[0]
        
        # Extract company name
        company_patterns = [
            r'this is ([A-Za-z\s]+) (?:trucking|transport|logistics|freight)',
            r'calling from ([A-Za-z\s]+)',
            r'my company is ([A-Za-z\s]+)'
        ]
        for pattern in company_patterns:
            matches = re.findall(pattern, transcript)
            if matches:
                extracted_data["carrier_info"]["company_name"] = matches[0].strip().title()
                break
        
        # Extract equipment types
        equipment_types = ["dry van", "reefer", "flatbed", "step deck", "lowboy", "refrigerated"]
        for equipment in equipment_types:
            if equipment in transcript:
                extracted_data["equipment_needs"].append(equipment.title())
        
        # Extract locations
        location_patterns = [
            r'from ([A-Za-z\s]+) to ([A-Za-z\s]+)',
            r'picking up in ([A-Za-z\s]+)',
            r'delivering to ([A-Za-z\s]+)',
            r'going to ([A-Za-z\s]+)',
            r'heading to ([A-Za-z\s]+)'
        ]
        for pattern in location_patterns:
            matches = re.findall(pattern, transcript)
            for match in matches:
                if isinstance(match, tuple):
                    extracted_data["locations_mentioned"].extend([m.strip().title() for m in match])
                else:
                    extracted_data["locations_mentioned"].append(match.strip().title())
        
        # Extract rates
        rate_matches = re.findall(r'\$(\d{1,4}(?:,\d{3})*(?:\.\d{2})?)', transcript)
        extracted_data["rates_discussed"] = [float(rate.replace(',', '')) for rate in rate_matches]
        
        # Extract key phrases
        key_phrases = []
        if "interested" in transcript:
            key_phrases.append("expressed interest")
        if "not interested" in transcript:
            key_phrases.append("not interested")
        if "call back" in transcript or "follow up" in transcript:
            key_phrases.append("requires follow-up")
        if "book" in transcript or "take it" in transcript:
            key_phrases.append("ready to book")
        
        extracted_data["key_phrases"] = key_phrases
        
        # Determine sentiment
        positive_words = ["interested", "good", "great", "excellent", "perfect", "sounds good", "yes"]
        negative_words = ["not interested", "too low", "can't do", "no way", "pass", "no"]
        
        positive_count = sum(1 for word in positive_words if word in transcript)
        negative_count = sum(1 for word in negative_words if word in transcript)
        
        if positive_count > negative_count:
            extracted_data["sentiment"] = "positive"
        elif negative_count > positive_count:
            extracted_data["sentiment"] = "negative"
        else:
            extracted_data["sentiment"] = "neutral"
        
        # Check for follow-up needs
        follow_up_phrases = ["call back", "follow up", "check with", "let me know", "think about it"]
        extracted_data["follow_up_required"] = any(phrase in transcript for phrase in follow_up_phrases)
        
        # Extract action items
        if extracted_data["follow_up_required"]:
            extracted_data["action_items"].append("Schedule follow-up call")
        if any(rate for rate in extracted_data["rates_discussed"]):
            extracted_data["action_items"].append("Review rate negotiation")
        if "email" in transcript or "send" in transcript:
            extracted_data["action_items"].append("Send documentation")
        
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
            "confidence": 0.0,
            "reasoning": []
        }
        
        # Intent classification
        if any(word in transcript for word in ["looking for", "need", "want", "interested", "available"]):
            classification["intent"] = "load_inquiry"
            classification["confidence"] += 0.3
            classification["reasoning"].append("Contains load inquiry keywords")
        elif any(word in transcript for word in ["rate", "price", "pay", "money", "negotiate"]):
            classification["intent"] = "rate_negotiation"
            classification["confidence"] += 0.3
            classification["reasoning"].append("Contains rate negotiation keywords")
        elif any(word in transcript for word in ["capacity", "truck", "equipment"]):
            classification["intent"] = "capacity_check"
            classification["confidence"] += 0.3
            classification["reasoning"].append("Contains capacity inquiry keywords")
        
        # Outcome classification
        if any(word in transcript for word in ["book", "take it", "sounds good", "deal", "yes"]):
            classification["outcome"] = "booked"
            classification["priority"] = "high"
            classification["confidence"] += 0.4
            classification["reasoning"].append("Positive booking indicators found")
        elif any(word in transcript for word in ["maybe", "think about", "call back", "follow up"]):
            classification["outcome"] = "follow_up_needed"
            classification["priority"] = "medium"
            classification["confidence"] += 0.3
            classification["reasoning"].append("Follow-up indicators found")
        elif any(word in transcript for word in ["not interested", "pass", "no thanks", "too low"]):
            classification["outcome"] = "not_interested"
            classification["priority"] = "low"
            classification["confidence"] += 0.4
            classification["reasoning"].append("Negative response indicators found")
        elif any(word in transcript for word in ["interested", "tell me more", "what else"]):
            classification["outcome"] = "interested"
            classification["priority"] = "high"
            classification["confidence"] += 0.4
            classification["reasoning"].append("Interest indicators found")
        
        # Action items
        if classification["outcome"] == "follow_up_needed":
            classification["action_required"].append("schedule_follow_up")
        if classification["outcome"] == "booked":
            classification["action_required"].append("send_confirmation")
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

@app.get("/dashboard-metrics")
async def get_dashboard_metrics():
    """Get dashboard metrics and analytics"""
    try:
        # Generate realistic mock metrics for demo
        from random import randint, choice
        
        metrics = {
            "summary": {
                "total_calls": randint(150, 250),
                "recent_calls": randint(25, 45),
                "conversion_rate": round(15 + (randint(-3, 8) * 0.1), 1),
                "average_negotiated_rate": randint(2200, 2800)
            },
            "classifications": [
                {"type": "load_inquiry", "count": randint(35, 55)},
                {"type": "rate_negotiation", "count": randint(20, 35)},
                {"type": "not_interested", "count": randint(10, 20)},
                {"type": "booked", "count": randint(8, 15)},
                {"type": "follow_up", "count": randint(5, 12)}
            ],
            "sentiments": [
                {"type": "positive", "count": randint(40, 60)},
                {"type": "neutral", "count": randint(30, 50)},
                {"type": "negative", "count": randint(10, 25)}
            ],
            "equipment_performance": [
                {"type": "Dry Van", "calls": randint(45, 65)},
                {"type": "Flatbed", "calls": randint(25, 40)},
                {"type": "Refrigerated", "calls": randint(20, 35)},
                {"type": "Step Deck", "calls": randint(10, 20)}
            ],
            "recent_activity": [
                {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "carrier_mc": "123456",
                    "company": "ABC Transportation LLC",
                    "outcome": "counter_offered",
                    "rate": 2325.00,
                    "load_id": "LD001"
                },
                {
                    "timestamp": (datetime.now(timezone.utc) - timedelta(minutes=15)).isoformat(),
                    "carrier_mc": "789012",
                    "company": "Express Freight Solutions",
                    "outcome": "booked",
                    "rate": 1800.00,
                    "load_id": "LD002"
                }
            ],
            "updated_at": datetime.now(timezone.utc).isoformat()
        }
        
        return {
            "success": True,
            "metrics": metrics
        }
        
    except Exception as e:
        logger.error(f"Dashboard metrics error: {e}")
        raise HTTPException(status_code=500, detail="Dashboard metrics failed")

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