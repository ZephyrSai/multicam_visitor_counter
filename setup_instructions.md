# Visitor Counting System - Setup Instructions

## System Requirements
- **GPU**: NVIDIA RTX 5060 Ti (or compatible CUDA-enabled GPU)
- **Python**: 3.12
- **OS**: Windows/Linux
- **CUDA**: 12.1 or higher recommended

## Step 1: Create Virtual Environment

### Windows
```bash
# Navigate to your project directory
cd your_project_directory

# Create virtual environment
python -m venv venv

# Activate virtual environment
venv\Scripts\activate
```

### Linux/Mac
```bash
# Navigate to your project directory
cd your_project_directory

# Create virtual environment
python3.12 -m venv venv

# Activate virtual environment
source venv/bin/activate
```

## Step 2: Install CUDA-enabled PyTorch

**IMPORTANT**: Install PyTorch with CUDA support BEFORE installing other packages.

### For CUDA 12.1
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### For CUDA 11.8 (if you have older drivers)
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Verify GPU Installation
```bash
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
```

Expected output:
```
CUDA Available: True
GPU: NVIDIA GeForce RTX 5060 Ti
```

## Step 3: Install Ultralytics and Dependencies

```bash
# Install ultralytics (includes YOLOv11)
pip install ultralytics

# Install other required packages
pip install opencv-python opencv-contrib-python
pip install numpy
pip install requests

# Optional but recommended
pip install lap  # Improves tracking performance
```

## Step 4: Project Structure

Create the following structure:
```
visitor_counter/
├── venv/                          # Virtual environment
├── main.py                        # Main application (from artifact)
├── camera_config.json             # Auto-generated config file
├── bytetrack.yaml                 # ByteTrack configuration
├── rtsp_urls.txt                  # Your camera URLs
└── logs/                          # Optional: for logging
```

## Step 5: Create ByteTrack Configuration

Create `bytetrack.yaml`:
```yaml
# ByteTrack configuration for Ultralytics
tracker_type: bytetrack
track_high_thresh: 0.5      # High threshold for track initialization
track_low_thresh: 0.1       # Low threshold for track continuation
new_track_thresh: 0.6       # Threshold for new track
track_buffer: 30            # Frames to keep lost tracks
match_thresh: 0.8           # Matching threshold
frame_rate: 30              # Expected frame rate
```

## Step 6: Create RTSP URLs File

Create `rtsp_urls.txt` with your camera URLs (one per line):
```
rtsp://admin:12345@192.168.5.227/71
rtsp://admin:12345@192.168.5.228/71
rtsp://admin:12345@192.168.5.229/71
```

## Step 7: Update Main Script

Modify the bottom of `main.py` to load URLs from file:

```python
if __name__ == "__main__":
    # Load RTSP URLs from file
    rtsp_urls = []
    if Path('rtsp_urls.txt').exists():
        with open('rtsp_urls.txt', 'r') as f:
            rtsp_urls = [line.strip() for line in f if line.strip() and not line.startswith('#')]
    
    if not rtsp_urls:
        print("No RTSP URLs found in rtsp_urls.txt")
        exit(1)
    
    # Create and run system
    system = VisitorCountingSystem()
    system.run(rtsp_urls)
```

## Step 8: First Run - Interactive Setup

On first run, the system will prompt you to configure each camera:

### ROI (Region of Interest) Setup
1. Click to add points (minimum 3) to define the area to monitor
2. Press **ENTER** when done
3. Press **C** to clear and restart

### Counting Line Setup
1. Click to add points (minimum 2) to draw the counting line
2. The line can be curved/irregular - not just straight
3. Press **ENTER** when done with the line
4. Set direction:
   - **N** = North/Downward (NS)
   - **S** = South/Upward (SN)  
   - **W** = West/Leftward (WE)
   - **E** = East/Rightward (EW)
5. You can add multiple lines per camera
6. Press **Q** when all lines are configured

### ThingsBoard URL Setup
After visual setup, you'll be prompted for ThingsBoard URL:
```
Example: http://localhost:8080/api/v1/YOUR_DEVICE_TOKEN/telemetry
```

## Step 9: Run the System

```bash
# Activate virtual environment (if not already active)
# Windows: venv\Scripts\activate
# Linux/Mac: source venv/bin/activate

# Run the application
python main.py
```

## Configuration File Format

After setup, `camera_config.json` will look like:
```json
{
  "cameras": {
    "rtsp://admin:12345@192.168.5.227/71": {
      "roi": [[100, 150], [500, 150], [500, 400], [100, 400]],
      "lines": [
        {
          "points": [[200, 250], [300, 260], [400, 250]],
          "direction": "NS",
          "thingsboard_key": "line_1"
        },
        {
          "points": [[200, 280], [300, 290], [400, 280]],
          "direction": "SN",
          "thingsboard_key": "line_2"
        }
      ],
      "thingsboard_url": "http://localhost:8080/api/v1/YOUR_TOKEN/telemetry"
    }
  },
  "settings": {
    "process_fps": 10,
    "resize_width": 640,
    "resize_height": 480
  }
}
```

## Reconfiguring a Camera

To reset a camera's configuration:
1. Stop the application
2. Open `camera_config.json`
3. Delete the camera's entry
4. Save the file
5. Restart the application - it will prompt for setup again

## Adjusting Settings

Edit `camera_config.json` to change:
- `process_fps`: Number of frames to process per second (default: 10)
- `resize_width`: Frame width for processing (default: 640)
- `resize_height`: Frame height for processing (default: 480)

## ThingsBoard Integration

### Data Format
The system posts to ThingsBoard every minute:
```json
{
  "line_1": 15,
  "line_2": 8
}
```

### Sample ThingsBoard URL Structure
```
http://<thingsboard-host>:<port>/api/v1/<device-access-token>/telemetry
```

Example:
```
http://localhost:8080/api/v1/dfxoifFCHa7zjWSf7AOT/telemetry
```

### Testing ThingsBoard Connection
```bash
curl -v -X POST http://localhost:8080/api/v1/YOUR_TOKEN/telemetry \
  --header "Content-Type:application/json" \
  --data '{"line_1":5}'
```

## Troubleshooting

### CUDA Not Available
```bash
# Check NVIDIA driver
nvidia-smi

# Reinstall PyTorch with correct CUDA version
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### RTSP Connection Issues
- Verify camera URL with VLC media player
- Check network connectivity
- Ensure credentials are correct
- Check firewall settings

### Model Download
First run will download YOLOv11n model (~6MB). This is automatic.

### Performance Issues
- Reduce `process_fps` in config (try 5 FPS)
- Reduce `resize_width` and `resize_height`
- Ensure GPU is being used: check with `nvidia-smi`

### Memory Issues
If processing many cameras:
- Reduce number of cameras per model instance (default: 5)
- In code, change `self.cameras_per_model = 3`

## Keyboard Shortcuts

During operation:
- **Q**: Quit camera view
- **Ctrl+C**: Stop entire system

## Logs

Check console output for:
- Connection status
- Crossing detections
- ThingsBoard posting status
- Error messages

## Performance Optimization

### For RTX 5060 Ti:
- Default settings should work well
- Can handle 5-10 cameras simultaneously at 10 FPS
- Adjust `process_fps` based on accuracy vs performance needs

### Recommended Settings by Scenario:
- **High Accuracy**: process_fps=15, resize=640x480
- **Balanced**: process_fps=10, resize=640x480 (default)
- **High Throughput**: process_fps=5, resize=416x416

## Security Notes

- Store RTSP credentials securely
- Don't commit `rtsp_urls.txt` or `camera_config.json` with credentials to version control
- Use `.gitignore`:
```
venv/
*.pyc
__pycache__/
camera_config.json
rtsp_urls.txt
logs/
```

## Support

For issues:
1. Check logs in console
2. Verify GPU usage with `nvidia-smi`
3. Test RTSP stream with VLC
4. Ensure all dependencies installed correctly
