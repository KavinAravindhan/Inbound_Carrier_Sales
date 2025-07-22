# Carrier Sales API with HappyRobot Integration

### HappyRobot Technical Challenge

A comprehensive inbound carrier sales system built with FastAPI, MySQL, and real-time HappyRobot webhook integration. This solution handles carrier verification, load matching, rate negotiation, and provides a live analytics dashboard.

---

## Project Structure

```
inbound_carrier_sales/
├── backend/                          # FastAPI application
│   ├── main.py                       # Production deployment version
│   ├── main_local.py                 # Local development version
│   └── requirements.txt              # Python dependencies
├── carrier-sales-db/                 # Database deployment
│   └── fly.toml                      # Database app configuration
├── dashboard/                        # Frontend dashboard
│   ├── index.html                    # Live dashboard
│   └── index_mock.html               # Mock data dashboard
├── Dockerfile                        # Container configuration
├── docker-compose.yml                # Multi-container setup
├── fly.toml                          # Main app configuration
├── setup.sh                          # Automated setup script
├── complete-setup-test.sh            # Production testing
├── .env.example                      # Environment template
└── README.md                         # This file
```

---

## Quick Start

### **Option 1: Automated Setup (Recommended)**

```bash
# Clone the repository
git clone https://github.com/Aravindhan/Inbound_Carrier_Sales
cd Inbound_Carrier_Sales

# Make setup script executable
chmod +x setup.sh

# Choose your setup method:
./setup.sh local      # Local development
./setup.sh docker     # Docker container
./setup.sh deploy     # Production deployment
./setup.sh test both  # Test all environments
```

### **Option 2: Manual Setup**

Follow the detailed instructions in the [Manual Setup](https://github.com/KavinAravindhan/Inbound_Carrier_Sales/tree/main?tab=readme-ov-file#manual-setup) section below.

---

## Prerequisites

### **Required**
- **Python 3.8+** with pip3
- **MySQL** for local database
- **curl** for API testing
- **Git** for version control

### **Optional (but recommended)**
- **Docker & Docker Compose** for containerization
- **Fly CLI** for production deployment ([Install Guide](https://fly.io/docs/hands-on/install-flyctl/))
- **jq** for JSON processing in tests

### **API Keys Required**
- **FMCSA API Key**: Obtain from [FMCSA API Portal](https://mobile.fmcsa.dot.gov/developer)
- **HappyRobot API Key**: Get from your HappyRobot platform account

---

## Manual Setup

## Environment Configuration

1. **Create environment file:**
   ```bash
   cp .env.example .env
   ```

2. **Update `.env` with your API keys:**
   ```bash
   # API Security
   API_KEY=secure-api-key-change-this-in-production
   
   # FMCSA API (Get from FMCSA portal)
   FMCSA_API_KEY=your-actual-fmcsa-api-key
   
   # HappyRobot API (Get from HappyRobot platform)
   HAPPYROBOT_API_KEY=your-actual-happyrobot-api-key
   HAPPYROBOT_BASE_URL=https://platform.happyrobot.ai/api/v1/
   
   # Application Settings
   ENVIRONMENT=development  # or production
   DEBUG=true              # or false for production
   
   # Database Configuration
   DATABASE_URL=mysql+aiomysql://<DB_USER>:<DB_PASSWORD>@<DB_HOST>:3306/<DB_NAME>
   # Local MySQL
   DATABASE_URL=mysql://<MYSQL_USER>:<MYSQL_PASSWORD>@<MYSQL_HOST>:3306/<MYSQL_DATABASE>
   ```

---

## Local Development

### **Method 1: Python Virtual Environment**

```bash
# 1. Set up Python environment
python3 -m venv .venv_carrier_sales
source .venv_carrier_sales/bin/activate

# 2. Install dependencies
pip install -r backend/requirements.txt

# 3. Run local server
cd backend
python main_local.py

# 4. Access the application
# API: http://localhost:8000
# Dashboard: http://localhost:8000/dashboard
# API Docs: http://localhost:8000/docs
```

### **Method 2: Docker Development**

```bash
# 1. Build and run with Docker Compose
docker-compose up -d

# 2. View logs
docker-compose logs -f

# 3. Stop services
docker-compose down
```

### **Testing Local Setup**

```bash
# Test local deployment
./complete-setup-test.sh local

# Or use the setup script
./setup.sh test local
```

---

## Docker Deployment

### **Single Container**

```bash
# Build image
docker build -t inbound_carrier_sales .

# Run container
docker run -d \
  --name carrier-sales-api \
  -p 8000:8000 \
  --env-file .env \
  carrier-sales-api
```

### **Multi-Container with Database**

```bash
# Start all services
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs carrier-sales-api
```

---

## Production Deployment (Fly.io)

### **Prerequisites**
1. **Install Fly CLI**: [Installation Guide](https://fly.io/docs/hands-on/install-flyctl/)
2. **Login to Fly.io**: `fly auth login`
3. **Have your API keys ready**

### **Automated Deployment**

```bash
# Deploy everything automatically
./setup.sh deploy
```

### **Manual Deployment Steps**

#### **Step 1: Deploy Database**

```bash
# Navigate to database configuration
cd carrier-sales-db

# Create database app (if not exists)
fly apps create carrier-sales-db --org personal

# Create persistent volume
fly vol create mysqldata -a carrier-sales-db --size 10 --region bos

# Deploy database
fly deploy -a carrier-sales-db

# Wait for initialization (2-3 minutes)
cd ..
```

#### **Step 2: Deploy Main Application**

```bash
# Create main app (if not exists)
fly apps create carrier-sales --org personal

# Set production secrets
fly secrets set -a carrier-sales \
  API_KEY="secure-api-key-change-this-in-production" \
  FMCSA_API_KEY="your-fmcsa-api-key" \
  HAPPYROBOT_API_KEY="your-happyrobot-api-key" \
  HAPPYROBOT_BASE_URL="https://platform.happyrobot.ai/api/v1/" \
  ENVIRONMENT="production" \
  DEBUG="false" \
  DB_HOST=<DB_HOST> \
  DB_USER=<DB_USER> \
  DB_PASS=<DB_PASSWORD> \
  DB_NAME=<DB_NAME> \
  DATABASE_URL=mysql+aiomysql://<DB_USER>:<DB_PASSWORD>@<DB_HOST>:3306/<DB_NAME>"

# Deploy main application
fly deploy -a carrier-sales
```

#### **Step 3: Verify Deployment**

```bash
# Check application status
fly status -a carrier-sales

# Test health endpoint
curl https://carrier-sales.fly.dev/health

# Run comprehensive tests
./setup.sh test production
```

---

## Testing

### **Comprehensive Test Suite**

```bash
# Test local environment
./complete-setup-test.sh local

# Test production deployment
./complete-setup-test.sh deployed

# Test both environments
./complete-setup-test.sh both
```

### **Individual API Testing**

```bash
# Health check
curl https://carrier-sales.fly.dev/health

# Carrier verification
curl -X POST "https://carrier-sales.fly.dev/verify-carrier" \
  -H "Authorization: Bearer secure-api-key-change-this-in-production" \
  -H "Content-Type: application/json" \
  -d '{"mc_number": "123456"}'

# Test webhook endpoint
curl -X POST "https://carrier-sales.fly.dev/webhooks/happyrobot/call-completed" \
  -H "Authorization: Bearer secure-api-key-change-this-in-production" \
  -H "Content-Type: application/json" \
  -d '{
    "call_id": "TEST_123",
    "transcript": "Hello, this is ABC Transportation, MC 123456, looking for loads",
    "duration": 30,
    "status": "completed"
  }'
```

---

## HappyRobot Integration

### **Webhook Configuration**

Configure your HappyRobot campaign with these webhook URLs:

| Webhook Type | URL | Method |
|--------------|-----|---------|
| **Call Completed** | `https://carrier-sales.fly.dev/webhooks/happyrobot/call-completed` | POST |
| **Carrier Verification** | `https://carrier-sales.fly.dev/verify-carrier` | POST |
| **Load Search** | `https://carrier-sales.fly.dev/search-loads` | POST |
| **Rate Negotiation** | `https://carrier-sales.fly.dev/negotiate-rate` | POST |
| **Call Classification** | `https://carrier-sales.fly.dev/classify-call` | POST |
| **Data Extraction** | `https://carrier-sales.fly.dev/extract-call-data` | POST |

### **Authentication**
- **Header**: `Authorization: Bearer secure-api-key-change-this-in-production`
- **Content-Type**: `application/json`

### **Workflow Setup**

1. **Create Inbound Voice Agent** in HappyRobot
2. **Configure Workflow** with the function calls above
3. **Set Webhook URLs** for each endpoint
4. **Use Web Call Feature** for testing (no phone number needed)

---

## Troubleshooting

### **Common Issues**

#### **Database Connection Issues**
```bash
# Check database status
fly status -a carrier-sales-db

# View database logs
fly logs -a carrier-sales-db

# Restart database
fly machine restart -a carrier-sales-db
```

#### **Application Not Responding**
```bash
# Check application logs
fly logs -a carrier-sales

# Check application status
fly status -a carrier-sales

# Restart application
fly machine restart -a carrier-sales
```

#### **Webhook Issues**
```bash
# Test webhook endpoint directly
curl -X POST "https://carrier-sales.fly.dev/webhooks/debug"

# Check webhook configuration
curl "https://carrier-sales.fly.dev/webhooks/debug"

# View recent webhook activity
curl "https://carrier-sales.fly.dev/dashboard/activity"
```

### **Debug Commands**

```bash
# Local debugging
python backend/main_local.py --debug

# Docker debugging
docker-compose logs -f inbound_carrier_sales

# Production debugging
fly logs -a carrier-sales --follow
```

### **Health Checks**

| Endpoint | Local | Production |
|----------|-------|------------|
| **Health** | http://localhost:8000/health | https://carrier-sales.fly.dev/health |
| **Metrics** | http://localhost:8000/dashboard-metrics | https://carrier-sales.fly.dev/dashboard-metrics |
| **Webhook Debug** | http://localhost:8000/webhooks/debug | https://carrier-sales.fly.dev/webhooks/debug |

---

## API Reference

### **Core Endpoints**

#### **Health Check**
```http
GET /health
```
Returns system health and integration status.

#### **Carrier Verification**
```http
POST /verify-carrier
Authorization: Bearer {API_KEY}
Content-Type: application/json

{
  "mc_number": "123456"
}
```

#### **Load Search**
```http
POST /search-loads
Authorization: Bearer {API_KEY}
Content-Type: application/json

{
  "equipment_type": "Dry Van",
  "pickup_city": "Chicago",
  "delivery_city": "Atlanta"
}
```

#### **Rate Negotiation**
```http
POST /negotiate-rate
Authorization: Bearer {API_KEY}
Content-Type: application/json

{
  "load_id": "LD001",
  "proposed_rate": 2400.00,
  "mc_number": "123456"
}
```

### **HappyRobot Webhooks**

#### **Call Completed**
```http
POST /webhooks/happyrobot/call-completed
Authorization: Bearer {API_KEY}
Content-Type: application/json

{
  "call_id": "HR_123456789",
  "transcript": "Call transcript...",
  "duration": 120,
  "status": "completed"
}
```

#### **Call Data Extraction**
```http
POST /extract-call-data
Authorization: Bearer {API_KEY}
Content-Type: application/json

{
  "call_transcript": "Hi, this is ABC Transportation, MC 123456...",
  "call_duration": 85
}
```

#### **Call Classification**
```http
POST /classify-call
Authorization: Bearer {API_KEY}
Content-Type: application/json

{
  "call_transcript": "We'll take that load for $2500..."
}
```

---

## Maintenance

### **Database Maintenance**

```bash
# Backup database
fly ssh console -a carrier-sales-db
mysqldump -u carrier_user -psupersecret carrier_db > backup.sql

# Restore database
mysql -u <DB_USER> -p <DB_PASSWORD> <DB_NAME> < backup.sql
```

### **Application Updates**

```bash
# Deploy new version
fly deploy -a carrier-sales

# Rollback if needed
fly releases -a carrier-sales
fly rollback -a carrier-sales v123
```

### **Monitoring**

```bash
# View metrics
fly status -a carrier-sales
fly metrics -a carrier-sales

# Monitor logs
fly logs -a carrier-sales --follow
```

---

## Development

### **Local Development Setup**

```bash
# 1. Clone and setup
git clone https://github.com/KavinAravindhan/Inbound_Carrier_Sales
cd inbound_carrier_sales
cp .env.example .env

# 2. Update .env with your API keys

# 3. Run setup
./setup.sh local

# 4. Start developing
# Edit files in backend/ directory
# Changes auto-reload in development mode
```

### **Adding New Features**

1. **Update backend/main.py** for new endpoints
2. **Add tests** to complete-setup-test.sh
3. **Update dashboard** in dashboard/index.html
4. **Test locally** with `./setup.sh test local`
5. **Deploy** with `./setup.sh deploy`

---

## Additional Resources

### **Documentation Links**
- **FastAPI Documentation**: https://fastapi.tiangolo.com/
- **HappyRobot Platform**: https://platform.happyrobot.ai/
- **Fly.io Deployment**: https://fly.io/docs/
- **FMCSA API**: https://mobile.fmcsa.dot.gov/developer

### **Support**
- **Technical Issues**: Check troubleshooting section above
- **API Questions**: Refer to `/docs` endpoint
- **Deployment Issues**: Check Fly.io status and logs

*Built with ❤️ for the HappyRobot Technical Challenge*
