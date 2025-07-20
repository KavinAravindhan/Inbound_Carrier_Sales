#!/bin/bash

# Complete Carrier Sales API Setup Test
# Tests all endpoints and prepares for deployment

echo "🚛 ================================="
echo "🚛 CARRIER SALES API SETUP TEST"
echo "🚛 ================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Test configuration
API_BASE_URL="http://localhost:8000"
API_KEY="secure-api-key-change-this-in-production"

echo "📋 Testing API at: $API_BASE_URL"
echo "🔑 Using API Key: ${API_KEY:0:20}..."
echo ""

# Function to test endpoint
test_endpoint() {
    local endpoint=$1
    local method=$2
    local data=$3
    local description=$4
    
    echo -n "🧪 Testing $description... "
    
    if [ "$method" = "GET" ]; then
        response=$(curl -s -w "%{http_code}" "$API_BASE_URL$endpoint")
    else
        response=$(curl -s -w "%{http_code}" -X "$method" \
            -H "Authorization: Bearer $API_KEY" \
            -H "Content-Type: application/json" \
            -d "$data" \
            "$API_BASE_URL$endpoint")
    fi
    
    http_code="${response: -3}"
    response_body="${response%???}"
    
    if [ "$http_code" = "200" ]; then
        echo -e "${GREEN}✅ PASS${NC}"
        return 0
    else
        echo -e "${RED}❌ FAIL (HTTP $http_code)${NC}"
        echo "   Response: $response_body"
        return 1
    fi
}

# Test counter
TOTAL_TESTS=0
PASSED_TESTS=0

echo "🔍 ================================="
echo "🔍 CORE API TESTS"
echo "🔍 ================================="
echo ""

# Test 1: Health Check
TOTAL_TESTS=$((TOTAL_TESTS + 1))
if test_endpoint "/health" "GET" "" "Health Check"; then
    PASSED_TESTS=$((PASSED_TESTS + 1))
fi

# Test 2: Root Endpoint
TOTAL_TESTS=$((TOTAL_TESTS + 1))
if test_endpoint "/" "GET" "" "Root Endpoint"; then
    PASSED_TESTS=$((PASSED_TESTS + 1))
fi

echo ""
echo "🚛 ================================="
echo "🚛 CARRIER OPERATIONS TESTS"
echo "🚛 ================================="
echo ""

# Test 3: Carrier Verification (Valid MC)
TOTAL_TESTS=$((TOTAL_TESTS + 1))
if test_endpoint "/verify-carrier" "POST" '{"mc_number": "123456"}' "Carrier Verification (Valid MC)"; then
    PASSED_TESTS=$((PASSED_TESTS + 1))
fi

# Test 4: Carrier Verification (Invalid MC)
TOTAL_TESTS=$((TOTAL_TESTS + 1))
if test_endpoint "/verify-carrier" "POST" '{"mc_number": "999999"}' "Carrier Verification (Invalid MC)"; then
    PASSED_TESTS=$((PASSED_TESTS + 1))
fi

echo ""
echo "📦 ================================="
echo "📦 LOAD OPERATIONS TESTS"
echo "📦 ================================="
echo ""

# Test 5: Load Search (Dry Van)
TOTAL_TESTS=$((TOTAL_TESTS + 1))
if test_endpoint "/search-loads" "POST" '{"equipment_type": "Dry Van"}' "Load Search (Dry Van)"; then
    PASSED_TESTS=$((PASSED_TESTS + 1))
fi

# Test 6: Load Search (Flatbed)
TOTAL_TESTS=$((TOTAL_TESTS + 1))
if test_endpoint "/search-loads" "POST" '{"equipment_type": "Flatbed"}' "Load Search (Flatbed)"; then
    PASSED_TESTS=$((PASSED_TESTS + 1))
fi

# Test 7: Load Search (With Filters)
TOTAL_TESTS=$((TOTAL_TESTS + 1))
if test_endpoint "/search-loads" "POST" '{"equipment_type": "Dry Van", "pickup_state": "IL", "delivery_state": "GA"}' "Load Search (With Filters)"; then
    PASSED_TESTS=$((PASSED_TESTS + 1))
fi

echo ""
echo "💰 ================================="
echo "💰 NEGOTIATION TESTS"
echo "💰 ================================="
echo ""

# Test 8: Rate Negotiation (Accept)
TOTAL_TESTS=$((TOTAL_TESTS + 1))
if test_endpoint "/negotiate-rate" "POST" '{"load_id": "LD001", "proposed_rate": 2450.00, "carrier_mc": "123456"}' "Rate Negotiation (Should Accept)"; then
    PASSED_TESTS=$((PASSED_TESTS + 1))
fi

# Test 9: Rate Negotiation (Counter-offer)
TOTAL_TESTS=$((TOTAL_TESTS + 1))
if test_endpoint "/negotiate-rate" "POST" '{"load_id": "LD001", "proposed_rate": 2300.00, "carrier_mc": "123456"}' "Rate Negotiation (Should Counter)"; then
    PASSED_TESTS=$((PASSED_TESTS + 1))
fi

# Test 10: Rate Negotiation (Reject)
TOTAL_TESTS=$((TOTAL_TESTS + 1))
if test_endpoint "/negotiate-rate" "POST" '{"load_id": "LD001", "proposed_rate": 1500.00, "carrier_mc": "123456"}' "Rate Negotiation (Should Reject)"; then
    PASSED_TESTS=$((PASSED_TESTS + 1))
fi

echo ""
echo "📞 ================================="
echo "📞 CALL PROCESSING TESTS"
echo "📞 ================================="
echo ""

# Test 11: Call Data Extraction
TOTAL_TESTS=$((TOTAL_TESTS + 1))
call_transcript='{"call_transcript": "Hi, this is ABC Transportation, MC number 123456. Looking for dry van loads from Chicago to Atlanta for $2400."}'
if test_endpoint "/extract-call-data" "POST" "$call_transcript" "Call Data Extraction"; then
    PASSED_TESTS=$((PASSED_TESTS + 1))
fi

# Test 12: Call Classification
TOTAL_TESTS=$((TOTAL_TESTS + 1))
classify_transcript='{"call_transcript": "Hi, looking for loads. Can you do $2200 for that Chicago to Atlanta run?"}'
if test_endpoint "/classify-call" "POST" "$classify_transcript" "Call Classification"; then
    PASSED_TESTS=$((PASSED_TESTS + 1))
fi

echo ""
echo "📊 ================================="
echo "📊 TEST RESULTS SUMMARY"
echo "📊 ================================="
echo ""

if [ $PASSED_TESTS -eq $TOTAL_TESTS ]; then
    echo -e "${GREEN}🎉 ALL TESTS PASSED! ($PASSED_TESTS/$TOTAL_TESTS)${NC}"
    echo ""
    echo -e "${GREEN}✅ Your Carrier Sales API is ready for deployment!${NC}"
    echo ""
    echo "🚀 Next Steps:"
    echo "   1. Deploy to Fly.io with: fly deploy"
    echo "   2. Configure HappyRobot webhooks"
    echo "   3. Run live demo tests"
    echo ""
    echo "📋 API Documentation: $API_BASE_URL/docs"
    echo "🔍 Health Check: $API_BASE_URL/health"
    echo ""
    exit 0
else
    echo -e "${RED}❌ SOME TESTS FAILED ($PASSED_TESTS/$TOTAL_TESTS passed)${NC}"
    echo ""
    echo -e "${YELLOW}⚠️  Please fix failing tests before deployment${NC}"
    echo ""
    exit 1
fi