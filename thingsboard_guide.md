# ThingsBoard Integration Guide

Complete guide for integrating the Visitor Counting System with ThingsBoard IoT platform.

## Overview

The system posts visitor counts to ThingsBoard every 60 seconds via HTTP API, enabling:
- Real-time dashboards
- Historical data analysis
- Alerts and notifications
- Data export and reporting

## Quick Setup

### 1. ThingsBoard Installation

#### Option A: Cloud (Easiest)
```bash
# Use ThingsBoard Cloud
https://thingsboard.cloud
# or
https://demo.thingsboard.io
```

#### Option B: Docker (Recommended)
```bash
# Pull and run ThingsBoard
docker run -it -p 8080:9090 -p 1883:1883 -p 5683:5683/udp \
  --name thingsboard thingsboard/tb-postgres

# Access at: http://localhost:8080
# Default credentials: tenant@thingsboard.org / tenant
```

#### Option C: Local Installation
```bash
# Ubuntu/Debian
wget https://github.com/thingsboard/thingsboard/releases/download/v3.6/thingsboard-3.6.deb
sudo dpkg -i thingsboard-3.6.deb
sudo service thingsboard start
```

### 2. Create Device

1. Login to ThingsBoard
2. Go to **Devices** → **+ Add Device**
3. Enter device details:
   - Name: `Camera_Entrance_1`
   - Device Profile: `default`
4. Click **Add**

### 3. Get Access Token

1. Click on your new device
2. Click **Copy access token**
3. Save it (e.g., `dfxoifFCHa7zjWSf7AOT`)

### 4. Configure System

Add to your `camera_config.json`:
```json
{
  "cameras": {
    "rtsp://admin:12345@192.168.5.227/71": {
      "thingsboard_url": "http://localhost:8080/api/v1/dfxoifFCHa7zjWSf7AOT/telemetry"
    }
  }
}
```

Or enter when prompted during interactive setup.

## Data Format

### Posted Every Minute
```json
{
  "line_1": 15,
  "line_2": 8,
  "timestamp": 1703001234567
}
```

### Key Naming Convention
- `line_1`: First counting line
- `line_2`: Second counting line
- etc.

You can customize keys in config:
```json
{
  "lines": [
    {
      "points": [[100, 200], [500, 200]],
      "direction": "NS",
      "thingsboard_key": "entrance_count"
    }
  ]
}
```

## Testing Connection

### Using curl
```bash
# Windows
curl -X POST http://localhost:8080/api/v1/YOUR_TOKEN/telemetry ^
  -H "Content-Type: application/json" ^
  -d "{\"test_count\":1}"

# Linux/Mac
curl -X POST http://localhost:8080/api/v1/YOUR_TOKEN/telemetry \
  -H "Content-Type: application/json" \
  -d '{"test_count":1}'
```

### Using Python
```python
import requests
import json

url = "http://localhost:8080/api/v1/YOUR_TOKEN/telemetry"
data = {"test_count": 1}

response = requests.post(url, json=data)
print(f"Status: {response.status_code}")
```

## Creating Dashboards

### Basic Counter Dashboard

1. Go to **Dashboards** → **+ Add Dashboard**
2. Name it: `Visitor Counting`
3. Add widgets:

#### Widget 1: Current Counters
```
Type: Digital Gauge
Data Key: line_1
Title: Entrance Count
```

#### Widget 2: Hourly Chart
```
Type: Time Series Chart
Data Keys: line_1, line_2
Aggregation: SUM
Time Window: Last 24 hours
Interval: 1 hour
```

#### Widget 3: Total Today
```
Type: Cards
Data Key: line_1
Aggregation: SUM
Time Window: Today
```

### Advanced Dashboard Example

```json
{
  "widgets": [
    {
      "type": "timeseries",
      "title": "Visitor Traffic",
      "dataKeys": [
        {"name": "line_1", "label": "Entrance"},
        {"name": "line_2", "label": "Exit"}
      ],
      "aggregation": "SUM",
      "interval": 3600000
    },
    {
      "type": "gauge",
      "title": "Current Flow",
      "dataKey": "line_1",
      "min": 0,
      "max": 100
    }
  ]
}
```

## Setting Up Alerts

### Example: High Traffic Alert

1. Go to **Rule Chains** → **Root Rule Chain**
2. Add **Script** node:

```javascript
// Alert if count > 50 in one minute
if (msg.line_1 > 50) {
  return {
    msg: msg,
    metadata: metadata,
    msgType: "High Traffic Alert"
  };
}
return {msg: msg, metadata: metadata, msgType: "OK"};
```

3. Add **Send Email** node
4. Configure email template:

```
Subject: High Visitor Traffic Alert

Camera: ${deviceName}
Count: ${line_1}
Time: ${timestamp}

Threshold exceeded: 50 visitors per minute
```

## Data Aggregation Examples

### Hourly Totals
```sql
SELECT 
  SUM(line_1) as entrance_total,
  SUM(line_2) as exit_total,
  DATE_TRUNC('hour', ts) as hour
FROM ts_kv
WHERE key IN ('line_1', 'line_2')
  AND ts >= NOW() - INTERVAL '24 hours'
GROUP BY hour
ORDER BY hour
```

### Peak Hours
```sql
SELECT 
  EXTRACT(HOUR FROM ts) as hour,
  AVG(line_1) as avg_traffic
FROM ts_kv
WHERE key = 'line_1'
  AND ts >= NOW() - INTERVAL '7 days'
GROUP BY hour
ORDER BY avg_traffic DESC
LIMIT 5
```

### Daily Comparison
```sql
SELECT 
  DATE(ts) as date,
  SUM(line_1) as total
FROM ts_kv
WHERE key = 'line_1'
  AND ts >= NOW() - INTERVAL '30 days'
GROUP BY date
ORDER BY date
```

## REST API Examples

### Get Latest Telemetry
```bash
curl -X GET "http://localhost:8080/api/plugins/telemetry/DEVICE/YOUR_DEVICE_ID/values/timeseries?keys=line_1,line_2" \
  -H "X-Authorization: Bearer YOUR_JWT_TOKEN"
```

### Get Historical Data
```bash
curl -X GET "http://localhost:8080/api/plugins/telemetry/DEVICE/YOUR_DEVICE_ID/values/timeseries?keys=line_1&startTs=1703001234000&endTs=1703087634000" \
  -H "X-Authorization: Bearer YOUR_JWT_TOKEN"
```

### Delete Telemetry
```bash
curl -X DELETE "http://localhost:8080/api/plugins/telemetry/DEVICE/YOUR_DEVICE_ID/timeseries/delete?keys=line_1&deleteAllDataForKeys=false&startTs=1703001234000&endTs=1703087634000" \
  -H "X-Authorization: Bearer YOUR_JWT_TOKEN"
```

## Multiple Cameras Setup

### Device Hierarchy
```
Root
├── Building_A
│   ├── Camera_Entrance
│   ├── Camera_Exit
│   └── Camera_Lobby
└── Building_B
    ├── Camera_Front
    └── Camera_Back
```

### Configuration Example
```json
{
  "cameras": {
    "rtsp://...camera1": {
      "thingsboard_url": "http://localhost:8080/api/v1/TOKEN_1/telemetry",
      "lines": [
        {"thingsboard_key": "building_a_entrance", ...}
      ]
    },
    "rtsp://...camera2": {
      "thingsboard_url": "http://localhost:8080/api/v1/TOKEN_2/telemetry",
      "lines": [
        {"thingsboard_key": "building_a_exit", ...}
      ]
    }
  }
}
```

### Aggregated Dashboard
Create a dashboard that shows:
- Total across all buildings
- Per-building totals
- Per-camera counts
- Comparative analysis

## Advanced Features

### Custom Attributes
Post additional metadata:
```python
# In main.py, modify post_to_thingsboard():
attributes = {
    "camera_location": "Main Entrance",
    "camera_model": "Hikvision DS-2CD2",
    "last_calibration": "2025-12-01"
}

requests.post(
    f"{base_url}/attributes",
    json=attributes
)
```

### Calculated Telemetry
```javascript
// In ThingsBoard Rule Chain
var entrance = msg.line_1;
var exit = msg.line_2;
var occupancy = metadata.occupancy || 0;

occupancy = occupancy + entrance - exit;
metadata.occupancy = occupancy;

msg.current_occupancy = occupancy;
return {msg: msg, metadata: metadata, msgType: msgType};
```

### Data Export

#### CSV Export
```bash
# Export last 24 hours
curl -X GET "http://localhost:8080/api/plugins/telemetry/DEVICE/YOUR_DEVICE_ID/values/timeseries?keys=line_1,line_2&startTs=START&endTs=END" \
  -H "X-Authorization: Bearer TOKEN" \
  > export.json

# Convert to CSV with Python
python -c "
import json, csv
data = json.load(open('export.json'))
with open('export.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['timestamp', 'line_1', 'line_2'])
    for row in data['line_1']:
        writer.writerow([row['ts'], row['value'], 0])
"
```

## Troubleshooting

### Connection Issues

**Error: Connection Refused**
```bash
# Check ThingsBoard is running
docker ps | grep thingsboard
# or
sudo service thingsboard status

# Check port
netstat -an | grep 8080
```

**Error: 401 Unauthorized**
- Verify access token is correct
- Check device exists and is active
- Token might have been regenerated

**Error: Timeout**
- Check firewall rules
- Verify network connectivity
- Check ThingsBoard server load

### Data Not Appearing

1. **Check logs in application**
```bash
python main.py 2>&1 | grep -i thingsboard
```

2. **Verify telemetry in ThingsBoard**
   - Go to Device → Latest Telemetry
   - Check if data keys appear
   - Verify timestamps are current

3. **Test with curl**
```bash
curl -v -X POST http://localhost:8080/api/v1/YOUR_TOKEN/telemetry \
  -H "Content-Type: application/json" \
  -d '{"test":1}'
```

### Performance Issues

**High Latency**
- Use local ThingsBoard instance
- Reduce posting frequency (modify `post_interval`)
- Use batch posting for multiple lines

**Data Loss**
- Check network stability
- Implement retry logic (already included)
- Monitor ThingsBoard queue size

## Security Best Practices

### Production Deployment

1. **Use HTTPS**
```python
thingsboard_url = "https://your-domain.com/api/v1/TOKEN/telemetry"
```

2. **Restrict Device Access**
   - Use unique tokens per device
   - Rotate tokens regularly
   - Implement IP whitelisting in ThingsBoard

3. **Network Security**
   - Use VPN for remote cameras
   - Firewall ThingsBoard ports
   - Enable SSL/TLS

4. **Monitor Access**
   - Enable audit logs in ThingsBoard
   - Set up alerts for unusual activity
   - Regular security reviews

## Sample Integration Script

```python
# standalone_test.py
import requests
import time

THINGSBOARD_URL = "http://localhost:8080/api/v1/YOUR_TOKEN/telemetry"

def post_data(line_1_count, line_2_count):
    data = {
        "line_1": line_1_count,
        "line_2": line_2_count,
        "timestamp": int(time.time() * 1000)
    }
    
    try:
        response = requests.post(THINGSBOARD_URL, json=data, timeout=5)
        if response.status_code == 200:
            print(f"✓ Posted: {data}")
        else:
            print(f"✗ Error {response.status_code}: {response.text}")
    except Exception as e:
        print(f"✗ Exception: {e}")

# Test posting
if __name__ == "__main__":
    for i in range(10):
        post_data(i, i * 2)
        time.sleep(1)
```

## Resources

- [ThingsBoard Documentation](https://thingsboard.io/docs/)
- [REST API Reference](https://thingsboard.io/docs/reference/rest-api/)
- [Rule Engine Guide](https://thingsboard.io/docs/user-guide/rule-engine-2-0/overview/)
- [Dashboard Development](https://thingsboard.io/docs/user-guide/dashboards/)

## Summary

The integration provides:
- ✅ Real-time data posting every minute
- ✅ Automatic retry on failures  
- ✅ Zero-value posting for complete data
- ✅ Multiple cameras support
- ✅ Custom key naming
- ✅ Production-ready error handling

For support, check the main README.md or application logs.
