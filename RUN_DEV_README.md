# Development Server Manager

A unified Python script to run both the Footsteps backend (Rust) and frontend (Python) processes with integrated logging, automatic restart on failure, graceful shutdown, and automatic Python virtual environment setup.

## Features

- **Dual Process Management**: Runs both backend (Rust) and frontend (Python) simultaneously
- **Automatic Virtual Environment**: Creates and manages Python venv automatically on first run
- **Dependency Management**: Installs dependencies from `requirements.txt` automatically
- **Debug Mode**: Optional `--debug` flag to build backend with debug logging
- **Video Support**: Optional `--video` argument to pass video files or camera indices to backend
- **Unified Logging**: Color-coded, timestamped output for both processes
- **Automatic Restart**: Automatically restarts processes if they crash (up to 5 attempts by default)
- **Graceful Shutdown**: Properly terminates both processes when interrupted (Ctrl+C)
- **Process Monitoring**: Tracks process status and reports exit codes
- **Error Handling**: Separate error stream handling and reporting

## Usage

### Basic Usage
```bash
python3 run_dev.py
```

### With Debug Mode
Enables debug logging in the backend (builds with `--features debug`):
```bash
python3 run_dev.py --debug
```

### With Video File
Pass a video file or camera index to the backend:
```bash
python3 run_dev.py --video people.mp4
python3 run_dev.py --video 0              # Camera 0
python3 run_dev.py --video 1              # Camera 1
```

### With Debug Mode and Video
Combine options:
```bash
python3 run_dev.py --debug --video people.mp4
```

### Display Help
```bash
python3 run_dev.py --help
```

## Automatic Setup

The first time you run any of these commands, they will:

1. **Create virtual environment** at `./venv/` (if it doesn't exist)
2. **Upgrade pip** in the virtual environment
3. **Install dependencies** from `requirements.txt`
4. **Start both processes** with unified logging

Subsequent runs will:
- Reuse the existing virtual environment  
- Reinstall/upgrade dependencies if `requirements.txt` has changed
- Start the processes immediately

## Dependencies

The project installs dependencies specified in `requirements.txt`:

```
opencv-python>=4.5.0
numpy>=1.20.0
pillow>=8.0.0
```

You can add more Python dependencies to `requirements.txt` and they will be installed on the next run.

## Virtual Environment Management

### Location
The virtual environment is created at `./venv/` in the project root.

### Python Executable
- **Linux/macOS**: `./venv/bin/python`
- **Windows**: `./venv/Scripts/python.exe`

### Removing Virtual Environment
To start fresh with a clean environment:
```bash
rm -rf venv
```

## Command-Line Options

### `--debug`
Enables debug mode for the backend:
```bash
python3 run_dev.py --debug
```

This builds the backend with `--features debug`, which includes:
- Detailed timing information
- Model information logging
- Progress bars for operations

### `--video VIDEO_PATH`
Specifies a video file or camera index for the backend:
```bash
python3 run_dev.py --video people.mp4
python3 run_dev.py --video 0
```

The video path is passed as an argument to the backend. Examples:
- `people.mp4` - Video file (relative or absolute path)
- `0` - Webcam (camera index)
- `/path/to/video.mp4` - Absolute path to video file

## Output Colors

- **BACKEND** (Blue): Rust backend process logs
- **FRONTEND** (Green): Python frontend process logs
- **ERROR** (Red): Error messages
- **WARNING** (Yellow): Warning messages

## Configuration

To customize the backend or frontend commands directly in the code, edit the `main()` function in `run_dev.py`:

```python
# Backend is built dynamically based on --debug flag
# Default:
backend_cmd = "cargo build --release && cargo run --release"

# With --debug flag:
backend_cmd = "cargo build --release --features debug && cargo run --release --features debug"

# Frontend command (runs from footsteps_projection_mapping directory)
frontend_cmd = [venv_python, "test_demo.py"]
```

### Backend Options

The backend command is automatically generated based on command-line flags:

- **Default build**:
  ```
  cargo build --release && cargo run --release
  ```

- **Debug build** (`--debug`):
  ```
  cargo build --release --features debug && cargo run --release --features debug
  ```

- **With video file** (`--video`):
  ```
  cargo build --release && cargo run --release -- /path/to/video.mp4
  ```

- **With both debug and video**:
  ```
  cargo build --release --features debug && cargo run --release --features debug -- /path/to/video.mp4
  ```

## Process Control

### Stop All Processes
Press `Ctrl+C` to gracefully shut down both processes.

The manager will:
1. Send SIGTERM to both processes
2. Wait up to 5 seconds for graceful shutdown
3. Force kill (SIGKILL) any remaining processes
4. Print shutdown confirmation

### Restart Behavior

If a process crashes:
1. The manager detects the exit
2. Logs the exit code and reason
3. Waits 2 seconds before restart
4. Attempts to restart (up to 5 times)
5. If max attempts exceeded, stops the manager

To change restart behavior, edit the ProcessManager initialization:
```python
manager = ProcessManagerShell(
    backend_cmd,
    frontend_cmd,
    max_restart_attempts=5,      # Change maximum attempts
    restart_delay=2                # Change delay in seconds
)
```

## Requirements

- Python 3.6+
- Rust toolchain (for backend)
- OpenCV (for backend)
- LLVM (for backend on some systems)
- macOS with CoreML available

See the main [README.md](../README.md) for full system requirements.

## Troubleshooting

### "Python 3 is not installed"
Install Python 3 via Homebrew:
```bash
brew install python3
```

### Backend Won't Start
Ensure you have all backend dependencies:
```bash
brew install opencv llvm
```

### Frontend Module Not Found
Ensure you're running from the project root:
```bash
cd /path/to/footsteps
python3 run_dev.py
```

Or update the `frontend_cmd` path in `run_dev.py` to use absolute paths.

### Stuck Processes
If processes don't exit cleanly, you can manually kill them:
```bash
pkill -f "cargo run"
pkill -f test_demo.py
```

## Advanced Usage

### Customizing Backend Build
Modify the backend_cmd to add flags or different build profiles:

```python
# Build only, don't run
backend_cmd = "cargo build --release"

# With debug logging
backend_cmd = "cargo build --release --features debug && cargo run --release --features debug"

# With specific options
backend_cmd = "cargo build --release && cargo run --release -- -m models/custom_model.mlpackage"
```

### Running Different Frontend
Replace the frontend command with any executable:

```python
# Run a web server
frontend_cmd = [sys.executable, "-m", "http.server", "8000"]

# Run another process entirely
frontend_cmd = ["~/myapp/start.sh"]
```

### Integration with CI/CD
The script returns 0 on success and 1 on failure, suitable for CI/CD:
```bash
if python3 run_dev.py; then
    echo "dev server ran successfully"
else
    echo "dev server failed"
    exit 1
fi
```

## License

This development tool is part of the Footsteps project. See LICENSE for details.
