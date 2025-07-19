#!/bin/bash

# Carrier Sales AI - Setup Script
echo "Setting up Carrier Sales AI environment..."

# Create virtual environment
echo "Creating Python virtual environment (.venv_happyrobot)..."
python3 -m venv .venv_happyrobot

# Activate virtual environment
echo "Activating virtual environment..."
source .venv_happyrobot/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "Installing Python dependencies..."
pip install -r backend/requirements.txt

# Create data directory
echo "Creating data directory..."
mkdir -p data

# Set up environment file
echo "Creating .env file..."
cat > .env << 'EOF'
# API Security
API_KEY=secure-api-key-change-this-in-production

# FMCSA API (get from https://mobile.fmcsa.dot.gov/QCDevsite/)
FMCSA_API_KEY=your-fmcsa-webkey-here

# Database
DATABASE_URL=sqlite:///./data/carrier_sales.db

# Environment
ENVIRONMENT=development
DEBUG=true

# Deployment URLs (update when deployed)
API_BASE_URL=http://localhost:8000
DASHBOARD_URL=http://localhost:3000
EOF

echo "Setup complete!"
echo ""
echo "To start development:"
echo "1. source .venv_happyrobot/bin/activate"
echo "2. cd backend && python main.py"
echo "3. Visit http://localhost:8000 for API"
echo "4. Visit http://localhost:3000 for dashboard"
echo ""
echo "📋 Next steps:"
echo "1. Get FMCSA API key from: https://mobile.fmcsa.dot.gov/QCDevsite/"
echo "2. Update .env with your FMCSA_API_KEY"
echo "3. Update HappyRobot webhook URLs when deployed"