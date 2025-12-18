# Multi-Camera Visitor Counting System

Industry-standard visitor counting system using YOLOv11n and ByteTrack for real-time people detection and tracking across multiple RTSP camera streams.

## Features

✅ **Multi-Camera Support** - Process multiple RTSP streams simultaneously

✅ **Interactive Setup** - Easy ROI and counting line configuration

✅ **Persistent Configuration** - Settings saved and reused

✅ **Advanced Tracking** - YOLOv11n with ByteTrack for accurate people counting

✅ **GPU Accelerated** - Optimized for NVIDIA RTX

✅ **Flexible Counting Lines** - Draw curved/irregular lines, set custom directions

✅ **ThingsBoard Integration** - Real-time telemetry posting

✅ **Robust Error Handling** - Auto-recovery from stream interruptions

✅ **Frame Rate Control** - Process at specified FPS while avoiding buffer buildup

✅ **Efficient Resource Management** - Share model instances across cameras

## Quick Start

### Windows
```bash
# Download all files and run:
quick_start.bat
```

### Linux/Mac
```bash
# Download all files and run:
chmod +x quick_start.sh
./quick_start.sh
```

## Manual Installation

### 1. Prerequisites
- Python 3.12
- NVIDIA GPU (RTX 5060 Ti or compatible)
- CUDA 12.1 or higher
- RTSP-enabled cameras

### 2. Create Virtual Environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### 3. Install PyTorch with CUDA
```bash
# For CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 4. Install Dependencies
```bash
pip install ultralytics opencv-python opencv-contrib-python numpy requests lap
```

### 5. Verify GPU
```bash
python -c "import torch; print(torch.cuda.is_available())"
# Should print: True
```

## Project Structure

```
visitor_counter/
├── main.py                 # Main application
├── camera_config.json      # Auto-generated camera configurations
├── rtsp_urls.txt          # Your camera URLs
├── bytetrack.yaml         # Tracker configuration
├── quick_start.bat        # Windows setup script
├── quick_start.sh         # Linux/Mac setup script
├── requirements.txt       # Python dependencies
└── venv/                  # Virtual environment
```

## Configuration

### RTSP URLs (`rtsp_urls.txt`)
```
# Camera 1
rtsp://admin:12345@192.168.5.227/71

# Camera 2
rtsp://admin:12345@192.168.5.228/71
```

### Settings (`camera_config.json`)
```json
{
  "settings": {
    "process_fps": 10,          // Frames to process per second
    "resize_width": 640,        // Processing frame width
    "resize_height": 480        // Processing frame height
  }
}
```

## First Run - Interactive Setup

### Step 1: Define ROI (Region of Interest)
1. Click on the video frame to add points
2. Create a polygon around the area to monitor (minimum 3 points)
3. Press **ENTER** when done
4. Press **C** to clear and restart

### Step 2: Draw Counting Lines
1. Click to add points along the desired counting line
2. Lines can be curved or irregular (not just straight)
3. Press **ENTER** when the line is complete

### Step 3: Set Direction
After drawing a line, set the counting direction:
- Press **N** for NS (North/Downward) - counts objects moving down
- Press **S** for SN (South/Upward) - counts objects moving up
- Press **W** for WE (West/Rightward) - counts objects moving right
- Press **E** for EW (East/Leftward) - counts objects moving left

### Step 4: Add More Lines (Optional)
- Add multiple lines per camera for bidirectional counting
- Each line has its own independent counter
- Press **Q** when all lines are configured

### Step 5: ThingsBoard URL
Enter your ThingsBoard telemetry endpoint:
```
http://localhost:8080/api/v1/YOUR_DEVICE_TOKEN/telemetry
```

## Running the System

```bash
python main.py
```

The system will:
1. Load camera configurations
2. Prompt for setup if any camera needs configuration
3. Load YOLOv11n model to GPU
4. Start processing all cameras in parallel
5. Post counts to ThingsBoard every minute
6. Display real-time video with overlays

## Keyboard Controls

During operation:
- **Q** - Close camera window
- **Ctrl+C** - Stop entire system

During setup:
- **Left Click** - Add point
- **ENTER** - Confirm current step
- **C** - Clear current drawing
- **N/S/W/E** - Set direction
- **Q** - Finish setup

## Understanding the Display

### Visual Elements
- **Green Polygon** - Region of Interest (ROI)
- **Red Line** - Counting line (odd-numbered)
- **Blue Line** - Counting line (even-numbered)
- **Arrow** - Direction of counting
- **Counter Text** - `L1:15` means Line 1 has counted 15 people

### What Gets Counted
- Only persons detected within the ROI
- Only crossings in the specified direction
- Each person counted once per crossing

## ThingsBoard Integration

### Data Format
Posted every 60 seconds:
```json
{
  "line_1": 15,
  "line_2": 8
}
```

### Count Reset
- Counts are sent every minute
- Minute counter resets after posting
- Cumulative display counter continues
- Zero counts are sent (allows downstream logic)

### Testing Connection
```bash
curl -X POST http://localhost:8080/api/v1/YOUR_TOKEN/telemetry \
  -H "Content-Type: application/json" \
  -d '{"line_1":5}'
```

## Advanced Configuration

### Adjust Processing Speed
Edit `camera_config.json`:
```json
{
  "settings": {
    "process_fps": 5,    // Lower = faster, less accurate
                         // Higher = slower, more accurate
    "resize_width": 640,
    "resize_height": 480
  }
}
```

### Multiple Models for Many Cameras
The system automatically:
- Loads 1 model for up to 5 cameras
- Loads 2 models for 6-10 cameras
- And so on...

To change cameras per model, edit `main.py`:
```python
self.cameras_per_model = 3  # Default is 5
```

### Frame Rate vs Processing Rate
- **RTSP Stream FPS**: 30 FPS (camera output)
- **Processing FPS**: 10 FPS (default, configurable)
- System reads all frames (prevents buffering)
- Only processes every N-th frame based on `process_fps`

## Reconfiguring Cameras

### Reset Single Camera
1. Stop the system
2. Edit `camera_config.json`
3. Delete the camera's entry
4. Save and restart

### Reset All Cameras
Delete `camera_config.json` and restart

## Performance Tuning

### For RTX 5060 Ti

**High Accuracy Mode**
- process_fps: 15
- resize: 640×480
- Cameras: 5-7

**Balanced Mode (Default)**
- process_fps: 10
- resize: 640×480
- Cameras: 8-10

**High Throughput Mode**
- process_fps: 5
- resize: 416×416
- Cameras: 10-15

### Monitoring GPU Usage
```bash
# In another terminal
nvidia-smi -l 1
```

Look for:
- GPU Utilization: Should be 60-90%
- Memory Usage: Should be stable
- Temperature: Should be < 85°C

## Troubleshooting

### CUDA Not Available
```bash
# Check driver
nvidia-smi

# Reinstall PyTorch
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Verify
python -c "import torch; print(torch.cuda.is_available())"
```

### RTSP Connection Failed
1. Test with VLC: `Media > Open Network Stream`
2. Verify credentials and IP address
3. Check network connectivity: `ping 192.168.5.227`
4. Ensure firewall allows RTSP (port 554)
5. Check camera is not at max connection limit

### Model Not Loading
```bash
# Clear cache
rm -rf ~/.cache/ultralytics

# Re-run (will re-download)
python main.py
```

### Low FPS / Lag
1. Reduce `process_fps` to 5
2. Reduce resolution to 416×416
3. Close other GPU applications
4. Check GPU usage with `nvidia-smi`

### High False Positives
1. Draw tighter ROI
2. Increase `track_high_thresh` in `bytetrack.yaml`
3. Position counting lines farther from ROI edges

### High False Negatives
1. Increase `process_fps`
2. Lower `track_high_thresh` in `bytetrack.yaml`
3. Ensure good camera angle and lighting

### ThingsBoard Not Receiving Data
1. Test URL with curl command
2. Check ThingsBoard device token
3. Verify network connectivity
4. Check ThingsBoard logs
5. Ensure URL starts with `http://` or `https://`

## Logging

Console logs include:
- Camera connection status
- Detection and tracking info
- Line crossing events
- ThingsBoard post results
- Error messages with details

## Best Practices

### Camera Positioning
- Mount at 2-3 meters height
- Angle 30-45 degrees downward
- Avoid direct sunlight/backlight
- Ensure minimum 480p resolution

### ROI Definition
- Include only walking paths
- Exclude stationary areas
- Leave margins around edges
- Test with actual traffic

### Counting Lines
- Place perpendicular to traffic flow
- Avoid placing near decision points
- Use multiple lines for bidirectional counting
- Test both directions before deployment

### System Deployment
- Use wired network for RTSP streams
- Ensure UPS for continuous operation
- Monitor logs regularly
- Set up alerts for connection failures

## Architecture

### Threading Model
```
Main Thread
├── Camera 1 Thread → Model Instance 1
├── Camera 2 Thread → Model Instance 1
├── Camera 3 Thread → Model Instance 1
├── Camera 4 Thread → Model Instance 1
├── Camera 5 Thread → Model Instance 1
├── Camera 6 Thread → Model Instance 2
└── ...
```

### Processing Pipeline
```
RTSP Stream → Frame Read → Resize → ROI Mask → 
YOLO Detection → ByteTrack → Line Cross Detection →
Count Update → Display → ThingsBoard Post
```

### Data Flow
1. Continuous frame reading (prevents buffering)
2. FPS-controlled processing
3. Per-object tracking with history
4. Geometric line crossing detection
5. Directional filtering
6. Minute-based aggregation
7. HTTP POST to ThingsBoard

## Security

### Credential Management
- Store credentials in `rtsp_urls.txt` (gitignored)
- Use strong camera passwords
- Restrict network access to cameras
- Use VPN for remote access

### .gitignore
```
venv/
*.pyc
__pycache__/
camera_config.json
rtsp_urls.txt
logs/
*.log
```

## Support & Contributions

### Getting Help
1. Check logs for error details
2. Review troubleshooting section
3. Test individual components
4. Verify hardware compatibility

### Known Limitations
- Works with RTSP streams only
- Requires NVIDIA GPU with CUDA
- Python 3.12 required for best compatibility
- Counting accuracy depends on camera quality and positioning

## License

This is an industry-standard implementation for educational and commercial use.

## Credits

- **YOLOv11**: Ultralytics
- **ByteTrack**: ByteTrack Paper
- **OpenCV**: Computer Vision operations
- **PyTorch**: Deep Learning framework

---

**Version**: 1.0  
**Last Updated**: December 2025  
**Tested On**: RTX 5060 Ti, Python 3.12, CUDA 12.1
