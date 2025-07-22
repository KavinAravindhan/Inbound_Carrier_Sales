#!/bin/bash

# Carrier Sales API Setup Script
# Supports local development, Docker, and Fly.io production deployment
# For HappyRobot Technical Challenge

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
PROJECT_NAME="carrier-sales-api"
DB_APP_NAME="carrier-sales-db"
MAIN_APP_NAME="carrier-sales-kavin"

echo -e "${BLUE}=================================================${NC}"
echo -e "${BLUE}ENHANCED CARRIER SALES API SETUP${NC}"
echo -e "${BLUE}HappyRobot Technical Challenge Solution${NC}"
echo -e "${BLUE}=================================================${NC}"
echo ""

print_header() {
    echo -e "${PURPLE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${PURPLE}$1${NC}"
    echo -e "${PURPLE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_info() {
    echo -e "${CYAN} $1${NC}"
}

# Function to check prerequisites
check_prerequisites() {
    print_header "CHECKING PREREQUISITES"
    
    local missing_tools=()
    
    # Check Python
    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
        print_success "Python 3 found: $PYTHON_VERSION"
    else
        missing_tools+=("python3")
        print_error "Python 3 is required"
    fi
    
    # Check pip
    if command -v pip3 &> /dev/null; then
        print_success "pip3 found"
    else
        missing_tools+=("pip3")
        print_error "pip3 is required"
    fi
    
    # Check Docker (optional)
    if command -v docker &> /dev/null; then
        print_success "Docker found"
        DOCKER_AVAILABLE=true
    else
        print_warning "Docker not found (optional - needed for containerization)"
        DOCKER_AVAILABLE=false
    fi
    
    # Check fly CLI (optional)
    if command -v fly &> /dev/null; then
        FLY_VERSION=$(fly version | head -1)
        print_success "Fly CLI found: $FLY_VERSION"
        FLY_AVAILABLE=true
    else
        print_warning "Fly CLI not found (optional - needed for production deployment)"
        print_info "Install from: https://fly.io/docs/hands-on/install-flyctl/"
        FLY_AVAILABLE=false
    fi
    
    # Check curl
    if command -v curl &> /dev/null; then
        print_success "curl found"
    else
        missing_tools+=("curl")
        print_error "curl is required for testing"
    fi
    
    # Check jq (optional but recommended)
    if command -v jq &> /dev/null; then
        print_success "jq found (for JSON processing)"
        JQ_AVAILABLE=true
    else
        print_warning "jq not found (optional but recommended for testing)"
        print_info "Install: sudo apt-get install jq (Linux) or brew install jq (Mac)"
        JQ_AVAILABLE=false
    fi
    
    if [ ${#missing_tools[@]} -ne 0 ]; then
        print_error "Missing required tools: ${missing_tools[*]}"
        echo ""
        echo "Please install the missing tools and run this script again."
        exit 1
    fi
    
    print_success "All required prerequisites found!"
    echo ""
}

# Function to set up environment
setup_environment() {
    print_header "ENVIRONMENT SETUP"
    
    # Check if .env exists
    if [ ! -f ".env" ]; then
        if [ -f ".env.example" ]; then
            print_info "Creating .env from .env.example..."
            cp .env.example .env
            print_warning "Please update .env with your actual API keys before proceeding"
            echo ""
            echo "Required API keys:"
            echo "  • FMCSA_API_KEY: Get from FMCSA API portal"
            echo "  • HAPPYROBOT_API_KEY: Get from HappyRobot platform"
            echo ""
            read -p "Press Enter when you've updated .env with your API keys..."
        else
            print_error ".env.example not found!"
            echo "Creating basic .env file..."
            cat > .env << 'EOF'
# API Security
API_KEY=secure-api-key-change-this-in-production

# FMCSA API (Get from FMCSA API portal)
FMCSA_API_KEY=your-fmcsa-api-key-here

# HappyRobot API Configuration (Get from HappyRobot platform)
HAPPYROBOT_API_KEY=your-happyrobot-api-key-here
HAPPYROBOT_BASE_URL=https://platform.happyrobot.ai/api/v1/

# Application Settings
ENVIRONMENT=development
DEBUG=true

# Database Configuration
DB_HOST=localhost
DB_USER=carrier_user
DB_PASS=supersecret
DB_NAME=carrier_db
DB_PORT=3306

# Full database URL
DATABASE_URL=mysql+aiomysql://carrier_user:supersecret@localhost:3306/carrier_db

# Deployment URLs
LOCAL_API_URL=http://localhost:8000
DEPLOYED_API_URL=https://carrier-sales-kavin.fly.dev
EOF
            print_warning "Created basic .env file. Please update with your actual API keys!"
            exit 1
        fi
    else
        print_success ".env file found"
    fi
    
    # Load environment variables
    source .env
    echo ""
}

# Function to set up Python environment
setup_python_env() {
    print_header "PYTHON ENVIRONMENT SETUP"
    
    # Check if virtual environment exists
    if [ ! -d ".venv_carrier_sales" ]; then
        print_info "Creating Python virtual environment..."
        python3 -m venv .venv_carrier_sales
    fi
    
    print_info "Activating virtual environment..."
    source .venv_carrier_sales/bin/activate
    
    print_info "Upgrading pip..."
    pip install --upgrade pip
    
    print_info "Installing Python dependencies..."
    if [ -f "backend/requirements.txt" ]; then
        pip install -r backend/requirements.txt
    elif [ -f "requirements.txt" ]; then
        pip install -r requirements.txt
    else
        print_error "requirements.txt not found!"
        exit 1
    fi
    
    print_success "Python environment setup complete"
    echo ""
}

# Function to run local development
run_local() {
    print_header "LOCAL DEVELOPMENT SETUP"
    
    setup_python_env
    
    print_info "Starting local development server..."
    echo ""
    echo "The API will be available at: http://localhost:8000"
    echo "  • API Documentation: http://localhost:8000/docs"
    echo "  • Health Check: http://localhost:8000/health"
    echo "  • Dashboard: http://localhost:8000/dashboard"
    echo ""
    echo "Press Ctrl+C to stop the server"
    echo ""
    
    # Check if main_local.py exists, otherwise use main.py
    if [ -f "backend/main_local.py" ]; then
        cd backend && python main_local.py
    elif [ -f "backend/main.py" ]; then
        cd backend && python main.py
    else
        print_error "No main.py file found in backend directory!"
        exit 1
    fi
}

# Function to run with Docker
run_docker() {
    print_header "DOCKER SETUP"
    
    if [ "$DOCKER_AVAILABLE" != true ]; then
        print_error "Docker is not installed!"
        exit 1
    fi
    
    print_info "Building Docker images..."
    
    # Build and run with docker-compose
    if [ -f "docker-compose.yml" ]; then
        print_info "Using docker-compose..."
        docker-compose down 2>/dev/null || true
        docker-compose build
        docker-compose up -d
        
        print_success "Services started with Docker Compose"
        echo ""
        echo "Services available:"
        echo "  • API: http://localhost:8000"
        echo "  • Database: localhost:3306"
        echo ""
        echo "To stop: docker-compose down"
        echo "To view logs: docker-compose logs -f"
        
    else
        # Fallback to basic Docker build
        print_info "Building Docker image..."
        docker build -t carrier-sales-api .
        
        print_info "Running Docker container..."
        docker run -d \
            --name carrier-sales-api \
            -p 8000:8000 \
            --env-file .env \
            carrier-sales-api
        
        print_success "Docker container started"
        echo "API available at: http://localhost:8000"
        echo "To stop: docker stop carrier-sales-api"
    fi
    
    # Wait for service to be ready
    print_info "Waiting for service to be ready..."
    sleep 5
    
    # Test the service
    if curl -f http://localhost:8000/health >/dev/null 2>&1; then
        print_success "Service is responding!"
    else
        print_warning "Service may still be starting up..."
    fi
    
    echo ""
}

# Function to deploy to Fly.io
deploy_production() {
    print_header "PRODUCTION DEPLOYMENT TO FLY.IO"
    
    if [ "$FLY_AVAILABLE" != true ]; then
        print_error "Fly CLI is not installed!"
        echo "Install from: https://fly.io/docs/hands-on/install-flyctl/"
        exit 1
    fi
    
    # Check if logged in
    if ! fly auth whoami >/dev/null 2>&1; then
        print_info "Please log in to Fly.io..."
        fly auth login
    fi
    
    # Deploy database first
    print_info "Deploying MySQL database..."
    
    if [ -d "carrier-sales-db" ]; then
        cd carrier-sales-db
        
        # Check if database app exists
        if ! fly apps list | grep -q "$DB_APP_NAME"; then
            print_info "Creating database app..."
            fly apps create $DB_APP_NAME --org personal
            
            # Create volume for database
            print_info "Creating database volume..."
            fly vol create mysqldata -a $DB_APP_NAME --size 10 --region bos
        fi
        
        print_info "Deploying database..."
        fly deploy -a $DB_APP_NAME
        
        cd ..
    else
        print_warning "carrier-sales-db directory not found, skipping database deployment"
    fi
    
    # Wait for database to be ready
    print_info "Waiting for database to initialize (this may take 2-3 minutes)..."
    sleep 30
    
    # Deploy main application
    print_info "Deploying main application..."
    
    # Set production secrets
    print_info "Setting production environment variables..."
    fly secrets set -a $MAIN_APP_NAME \
        API_KEY="${API_KEY:-secure-api-key-change-this-in-production}" \
        FMCSA_API_KEY="${FMCSA_API_KEY}" \
        HAPPYROBOT_API_KEY="${HAPPYROBOT_API_KEY}" \
        HAPPYROBOT_BASE_URL="${HAPPYROBOT_BASE_URL:-https://platform.happyrobot.ai/api/v1/}" \
        ENVIRONMENT="production" \
        DEBUG="false" \
        DB_HOST="carrier-sales-db.internal" \
        DB_USER="carrier_user" \
        DB_PASS="supersecret" \
        DB_NAME="carrier_db" \
        DB_PORT="3306" \
        DATABASE_URL="mysql+aiomysql://carrier_user:supersecret@carrier-sales-db.internal:3306/carrier_db" \
        2>/dev/null || print_warning "Some secrets may have failed to set"
    
    # Deploy main app
    if ! fly apps list | grep -q "$MAIN_APP_NAME"; then
        print_info "Creating main app..."
        fly apps create $MAIN_APP_NAME --org personal
    fi
    
    print_info "Deploying main application..."
    fly deploy -a $MAIN_APP_NAME
    
    # Wait for deployment
    print_info "Waiting for deployment to complete..."
    sleep 20
    
    # Test deployment
    print_info "Testing production deployment..."
    PROD_URL="https://$MAIN_APP_NAME.fly.dev"
    
    if curl -f "$PROD_URL/health" >/dev/null 2>&1; then
        print_success "Production deployment successful!"
        echo ""
        echo "Production URLs:"
        echo "  • API: $PROD_URL"
        echo "  • Health: $PROD_URL/health"
        echo "  • Dashboard: $PROD_URL/dashboard"
        echo "  • API Docs: $PROD_URL/docs"
        echo ""
        echo "HappyRobot Webhook URL:"
        echo "  $PROD_URL/webhooks/happyrobot/call-completed"
        echo ""
    else
        print_warning "Deployment completed but service may still be starting..."
        echo "Check status with: fly status -a $MAIN_APP_NAME"
        echo "View logs with: fly logs -a $MAIN_APP_NAME"
    fi
    
    echo ""
}

# Function to run tests
run_tests() {
    print_header "RUNNING TESTS"
    
    local test_env="$1"
    
    if [ -f "complete-setup-test.sh" ]; then
        chmod +x complete-setup-test.sh
        
        case "$test_env" in
            "local")
                if [ -f "complete-setup-test-local-db.sh" ]; then
                    chmod +x complete-setup-test-local-db.sh
                    ./complete-setup-test-local-db.sh
                else
                    ./complete-setup-test.sh local
                fi
                ;;
            "production")
                ./complete-setup-test.sh deployed
                ;;
            "both")
                ./complete-setup-test.sh both
                ;;
            *)
                print_info "Testing deployed version..."
                ./complete-setup-test.sh deployed
                ;;
        esac
    else
        print_warning "Test script not found, running basic health check..."
        
        if [ "$test_env" = "local" ]; then
            curl -f http://localhost:8000/health || print_error "Local health check failed"
        else
            curl -f "https://$MAIN_APP_NAME.fly.dev/health" || print_error "Production health check failed"
        fi
    fi
    
    echo ""
}

# Function to show help
show_help() {
    echo "Enhanced Carrier Sales API Setup Script"
    echo ""
    echo "Usage: $0 [COMMAND] [OPTIONS]"
    echo ""
    echo "Commands:"
    echo "  local           Run locally with Python virtual environment"
    echo "  docker          Build and run with Docker"
    echo "  deploy          Deploy to Fly.io production"
    echo "  test [env]      Run tests (env: local, production, both)"
    echo "  clean           Clean up local environment"
    echo "  help            Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 local                    # Run locally for development"
    echo "  $0 docker                   # Run with Docker"
    echo "  $0 deploy                   # Deploy to production"
    echo "  $0 test local               # Test local deployment"
    echo "  $0 test production          # Test production deployment"
    echo "  $0 test both                # Test both environments"
    echo ""
    echo "Prerequisites:"
    echo "  • Python 3.8+"
    echo "  • pip3"
    echo "  • curl"
    echo "  • Docker (optional, for containerization)"
    echo "  • Fly CLI (optional, for production deployment)"
    echo "  • jq (optional, for JSON processing)"
    echo ""
}

# Function to clean up
clean_environment() {
    print_header "🧹 CLEANING UP ENVIRONMENT"
    
    print_info "Stopping Docker containers..."
    docker-compose down 2>/dev/null || true
    docker stop carrier-sales-api 2>/dev/null || true
    docker rm carrier-sales-api 2>/dev/null || true
    
    print_info "Removing Python virtual environment..."
    rm -rf .venv_carrier_sales
    
    print_info "Removing Docker images..."
    docker rmi carrier-sales-api 2>/dev/null || true
    
    print_success "Cleanup complete!"
    echo ""
}

# Main script logic
main() {
    case "${1:-help}" in
        "local")
            check_prerequisites
            setup_environment
            run_local
            ;;
        "docker")
            check_prerequisites
            setup_environment
            run_docker
            ;;
        "deploy")
            check_prerequisites
            setup_environment
            deploy_production
            ;;
        "test")
            check_prerequisites
            run_tests "${2:-production}"
            ;;
        "clean")
            clean_environment
            ;;
        "help"|"--help"|"-h")
            show_help
            ;;
        *)
            echo "Unknown command: $1"
            echo ""
            show_help
            exit 1
            ;;
    esac
}

# Run main function with all arguments
main "$@"