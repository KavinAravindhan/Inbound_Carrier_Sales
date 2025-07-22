#!/bin/bash

echo "🔍 Quick Debug: Data Source Detection Issue"
echo "==========================================="

API_BASE_URL="https://carrier-sales-kavin.fly.dev"
API_KEY="secure-api-key-change-this-in-production"

echo ""
echo "1. 📊 Checking current dashboard metrics..."
METRICS_RESPONSE=$(curl -s "${API_BASE_URL}/dashboard-metrics")
echo "Current data source: $(echo $METRICS_RESPONSE | jq -r '.metrics.data_source')"
echo "Total calls: $(echo $METRICS_RESPONSE | jq -r '.metrics.summary.total_calls')"
echo ""

echo "2. 🎯 Checking webhook activity..."
ACTIVITY_RESPONSE=$(curl -s "${API_BASE_URL}/dashboard/activity")
echo "Activity count: $(echo $ACTIVITY_RESPONSE | jq -r '.activity | length')"
echo "Activity data source: $(echo $ACTIVITY_RESPONSE | jq -r '.data_source')"
echo "Database available: $(echo $ACTIVITY_RESPONSE | jq -r '.database_available')"
echo ""

echo "3. 🏥 Checking health status..."
HEALTH_RESPONSE=$(curl -s "${API_BASE_URL}/health")
echo "Database status: $(echo $HEALTH_RESPONSE | jq -r '.services.database')"
echo "Webhook status: $(echo $HEALTH_RESPONSE | jq -r '.services.happyrobot_webhooks')"
echo ""

echo "4. 🧪 Sending a fresh test webhook to force update..."
TEST_WEBHOOK='{
  "call_id": "DEBUG_WEBHOOK_'$(date +%s)'",
  "transcript": "This is a debug webhook from ABC Transportation, MC 123456. We need dry van loads from Chicago to Atlanta.",
  "duration": 30,
  "status": "completed"
}'

WEBHOOK_RESPONSE=$(curl -s -X POST "${API_BASE_URL}/webhooks/happyrobot/call-completed" \
  -H "Authorization: Bearer ${API_KEY}" \
  -H "Content-Type: application/json" \
  -d "$TEST_WEBHOOK")

echo "Webhook response success: $(echo $WEBHOOK_RESPONSE | jq -r '.success')"
echo "Webhook stored in: $(echo $WEBHOOK_RESPONSE | jq -r '.stored_in // "unknown"')"
echo ""

echo "5. ⏳ Waiting 3 seconds for processing..."
sleep 3

echo ""
echo "6. 📊 Checking metrics again after webhook..."
METRICS_RESPONSE2=$(curl -s "${API_BASE_URL}/dashboard-metrics")
echo "New data source: $(echo $METRICS_RESPONSE2 | jq -r '.metrics.data_source')"
echo "New total calls: $(echo $METRICS_RESPONSE2 | jq -r '.metrics.summary.total_calls')"
echo ""

echo "7. 🎯 Checking activity again..."
ACTIVITY_RESPONSE2=$(curl -s "${API_BASE_URL}/dashboard/activity")
echo "New activity count: $(echo $ACTIVITY_RESPONSE2 | jq -r '.activity | length')"
echo "Latest call: $(echo $ACTIVITY_RESPONSE2 | jq -r '.activity[0].id // "none"')"
echo ""

# Analysis
echo "🔍 ANALYSIS:"
echo "============"

CURRENT_SOURCE=$(echo $METRICS_RESPONSE2 | jq -r '.metrics.data_source')
ACTIVITY_COUNT=$(echo $ACTIVITY_RESPONSE2 | jq -r '.activity | length')
DB_STATUS=$(echo $HEALTH_RESPONSE | jq -r '.services.database')

if [[ "$CURRENT_SOURCE" == *"webhook"* ]]; then
    echo "✅ SUCCESS: Data source is now showing webhook data: $CURRENT_SOURCE"
    echo "   Your dashboard should now show the green 'LIVE DATA' banner!"
elif [[ "$ACTIVITY_COUNT" -gt "0" ]]; then
    echo "❌ ISSUE: Webhooks are being received ($ACTIVITY_COUNT calls) but data source is still: $CURRENT_SOURCE"
    echo ""
    echo "🔧 FIX REQUIRED:"
    echo "   The get_real_time_metrics() function needs to be updated."
    echo "   Replace the function in main.py with the fixed version provided."
    echo "   Then redeploy: fly deploy -a carrier-sales-kavin"
else
    echo "❌ ISSUE: No webhook activity detected"
    echo "   Check webhook endpoint and processing logic"
fi

echo ""
echo "🚀 NEXT STEPS:"
if [[ "$CURRENT_SOURCE" != *"webhook"* ]] && [[ "$ACTIVITY_COUNT" -gt "0" ]]; then
    echo "   1. Update main.py with the fixed get_real_time_metrics() function"
    echo "   2. Deploy: fly deploy -a carrier-sales-kavin"  
    echo "   3. Refresh your dashboard to see the green 'LIVE DATA' banner"
    echo "   4. Your webhook integration is working - just need to fix the display logic"
else
    echo "   1. Check fly logs -a carrier-sales-kavin for any errors"
    echo "   2. Verify database connection is stable"
    echo "   3. Test individual webhook endpoint manually"
fi
echo ""