#!/usr/bin/env python3
"""
Footsteps Development Server Manager
=====================================
Manages both backend (Rust) and frontend (Python) processes with:
- Combined logging with timestamps
- Automatic restart on crash
- Graceful shutdown
- Color-coded process output
- Virtual environment management
- Debug mode and video file support
"""

import subprocess
import sys
import threading
import time
import signal
import os
import argparse
from datetime import datetime
from pathlib import Path
from collections import deque
import queue
import venv

# ANSI color codes
class Color:
    RESET = '\033[0m'
    BOLD = '\033[1m'
    DIM = '\033[2m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'


def setup_virtual_environment(project_root, venv_name="venv"):
    """
    Initialize and setup virtual environment with dependencies.
    
    Args:
        project_root: Path to project root
        venv_name: Name of virtual environment directory
        
    Returns:
        Path to Python executable in venv
    """
    venv_path = Path(project_root) / venv_name
    python_exec = venv_path / "bin" / "python"
    pip_exec = venv_path / "bin" / "pip"
    
    # Create venv if it doesn't exist
    if not venv_path.exists():
        print(f"{Color.YELLOW}Creating virtual environment at {venv_path}...{Color.RESET}")
        venv.create(venv_path, with_pip=True)
        print(f"{Color.GREEN}Virtual environment created{Color.RESET}")
    
    # Install/upgrade pip
    print(f"{Color.YELLOW}Upgrading pip...{Color.RESET}")
    subprocess.run(
        [str(python_exec), "-m", "pip", "install", "--upgrade", "pip"],
        capture_output=False
    )
    
    # Install requirements
    requirements_file = Path(project_root) / "footsteps_projection_mapping" / "requirements.txt"
    if requirements_file.exists():
        print(f"{Color.YELLOW}Installing dependencies from {requirements_file.name}...{Color.RESET}")
        result = subprocess.run(
            [str(pip_exec), "install", "-r", str(requirements_file)],
            capture_output=False
        )
        if result.returncode != 0:
            print(f"{Color.RED}Failed to install dependencies{Color.RESET}")
            sys.exit(1)
        print(f"{Color.GREEN}Dependencies installed{Color.RESET}")
    else:
        print(f"{Color.YELLOW}No requirements.txt found at {requirements_file}{Color.RESET}")
    
    return str(python_exec)


class ProcessManager:
    def __init__(self, backend_cmd, frontend_cmd, max_restart_attempts=5, restart_delay=2):
        """
        Initialize the process manager.
        
        Args:
            backend_cmd: List of command arguments for backend (e.g., ['cargo', 'run', '--release'])
            frontend_cmd: List of command arguments for frontend (e.g., ['python', 'script.py'])
            max_restart_attempts: Maximum number of restart attempts before giving up
            restart_delay: Delay in seconds before restarting a failed process
        """
        self.backend_cmd = backend_cmd
        self.frontend_cmd = frontend_cmd
        self.max_restart_attempts = max_restart_attempts
        self.restart_delay = restart_delay
        
        self.backend_process = None
        self.frontend_process = None
        self.backend_restart_count = 0
        self.frontend_restart_count = 0
        self.running = True
        
        # Log queues for thread-safe logging
        self.log_queue = queue.Queue()
        self.backend_monitor_thread = None
        self.frontend_monitor_thread = None
        
    def log(self, process_name, message, level="INFO"):
        """Thread-safe logging with timestamps and colors."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Color code by process
        if process_name == "BACKEND":
            color = Color.BLUE
        elif process_name == "FRONTEND":
            color = Color.GREEN
        elif level == "ERROR":
            color = Color.RED
        elif level == "WARNING":
            color = Color.YELLOW
        else:
            color = Color.WHITE
            
        log_msg = f"{Color.DIM}{timestamp}{Color.RESET} {color}[{process_name:8}]{Color.RESET} {message}"
        self.log_queue.put(log_msg)
        
    def print_logs(self):
        """Print all queued logs."""
        while not self.log_queue.empty():
            try:
                print(self.log_queue.get_nowait())
            except queue.Empty:
                break
                
    def read_output(self, process, process_name):
        """Read output from a process and log it."""
        try:
            while self.running and process.poll() is None:
                line = process.stdout.readline()
                if line:
                    line = line.decode('utf-8', errors='replace').strip()
                    if line:
                        self.log(process_name, line)
                self.print_logs()
        except Exception as e:
            self.log(process_name, f"Error reading output: {e}", "ERROR")
            
    def start_backend(self):
        """Start the backend process."""
        if self.backend_restart_count >= self.max_restart_attempts:
            self.log("BACKEND", 
                    f"Max restart attempts ({self.max_restart_attempts}) reached. Stopping.",
                    "ERROR")
            self.running = False
            return False
            
        try:
            self.log("BACKEND", f"Starting: {' '.join(self.backend_cmd)}")
            self.backend_process = subprocess.Popen(
                self.backend_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=False,
                bufsize=1
            )
            self.log("BACKEND", f"Started with PID {self.backend_process.pid}")
            self.backend_restart_count += 1
            
            # Start monitoring thread
            self.backend_monitor_thread = threading.Thread(
                target=self.monitor_backend, daemon=True
            )
            self.backend_monitor_thread.start()
            return True
        except Exception as e:
            self.log("BACKEND", f"Failed to start: {e}", "ERROR")
            return False
            
    def start_frontend(self):
        """Start the frontend process."""
        if self.frontend_restart_count >= self.max_restart_attempts:
            self.log("FRONTEND", 
                    f"Max restart attempts ({self.max_restart_attempts}) reached. Stopping.",
                    "ERROR")
            self.running = False
            return False
            
        try:
            self.log("FRONTEND", f"Starting: {' '.join(self.frontend_cmd)}")
            self.frontend_process = subprocess.Popen(
                self.frontend_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=False,
                bufsize=1,
                cwd=os.path.join(os.getcwd(), "footsteps_projection_mapping")
            )
            self.log("FRONTEND", f"Started with PID {self.frontend_process.pid}")
            self.frontend_restart_count += 1
            
            # Start monitoring thread
            self.frontend_monitor_thread = threading.Thread(
                target=self.monitor_frontend, daemon=True
            )
            self.frontend_monitor_thread.start()
            return True
        except Exception as e:
            self.log("FRONTEND", f"Failed to start: {e}", "ERROR")
            return False
            
    def monitor_backend(self):
        """Monitor backend process and handle restarts."""
        while self.running:
            try:
                # Read output
                self.read_output(self.backend_process, "BACKEND")
                
                # Check if process is still running
                if not self.running:
                    break
                    
                returncode = self.backend_process.poll()
                if returncode is not None:
                    self.log("BACKEND", 
                            f"Process exited with code {returncode}", 
                            "WARNING")
                    
                    if self.running:
                        self.log("BACKEND", 
                                f"Restarting in {self.restart_delay}s... "
                                f"(attempt {self.backend_restart_count}/{self.max_restart_attempts})")
                        time.sleep(self.restart_delay)
                        
                        if not self.start_backend():
                            break
                else:
                    time.sleep(0.1)
            except Exception as e:
                self.log("BACKEND", f"Monitor error: {e}", "ERROR")
                
    def monitor_frontend(self):
        """Monitor frontend process and handle restarts."""
        while self.running:
            try:
                # Read output
                self.read_output(self.frontend_process, "FRONTEND")
                
                # Check if process is still running
                if not self.running:
                    break
                    
                returncode = self.frontend_process.poll()
                if returncode is not None:
                    self.log("FRONTEND", 
                            f"Process exited with code {returncode}", 
                            "WARNING")
                    
                    if self.running:
                        self.log("FRONTEND", 
                                f"Restarting in {self.restart_delay}s... "
                                f"(attempt {self.frontend_restart_count}/{self.max_restart_attempts})")
                        time.sleep(self.restart_delay)
                        
                        if not self.start_frontend():
                            break
                else:
                    time.sleep(0.1)
            except Exception as e:
                self.log("FRONTEND", f"Monitor error: {e}", "ERROR")
                
    def run(self):
        """Main run loop."""
        self.log("MAIN", f"{Color.BOLD}=== Footsteps Development Server ==={Color.RESET}")
        self.log("MAIN", "Starting both backend and frontend processes...")
        
        # Start both processes
        backend_started = self.start_backend()
        time.sleep(1)  # Stagger startup
        frontend_started = self.start_frontend()
        
        if not backend_started or not frontend_started:
            self.log("MAIN", "Failed to start processes", "ERROR")
            self.running = False
            return False
            
        self.log("MAIN", "Both processes started. Press Ctrl+C to stop.")
        
        # Main loop - just print logs
        try:
            while self.running:
                self.print_logs()
                time.sleep(0.1)
        except KeyboardInterrupt:
            self.log("MAIN", "Keyboard interrupt received")
            
        return True
        
    def shutdown(self):
        """Gracefully shutdown both processes."""
        self.log("MAIN", "Shutting down processes...")
        self.running = False
        
        # Terminate processes
        if self.backend_process and self.backend_process.poll() is None:
            self.log("BACKEND", "Terminating...")
            self.backend_process.terminate()
            try:
                self.backend_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.log("BACKEND", "Force killing after timeout", "WARNING")
                self.backend_process.kill()
                
        if self.frontend_process and self.frontend_process.poll() is None:
            self.log("FRONTEND", "Terminating...")
            self.frontend_process.terminate()
            try:
                self.frontend_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.log("FRONTEND", "Force killing after timeout", "WARNING")
                self.frontend_process.kill()
                
        self.log("MAIN", "All processes stopped")


def main():
    """Main entry point."""
    # Determine the project root
    script_dir = Path(__file__).parent.resolve()
    
    # Define backend and frontend commands
    # Backend: Cargo build and run
    backend_cmd = ["cargo", "build", "--release", "&&", "cargo", "run", "--release"]
    # Use shell=True to handle the && operator
    backend_cmd_shell = "cargo build --release && cargo run --release"
    
    # Frontend: Python script (edit as needed for your actual frontend)
    # For now, we'll use the projection_mapping test demo
    frontend_cmd = [sys.executable, "footsteps_projection_mapping/test_demo.py"]
    
    # Change to project directory
    os.chdir(script_dir)
    
    # Create manager with shell=True for backend
    manager = ProcessManager(
        backend_cmd_shell,
        frontend_cmd,
        max_restart_attempts=5,
        restart_delay=2
    )
    
    # Handle signals
    def signal_handler(sig, frame):
        manager.shutdown()
        sys.exit(0)
        
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Need to use shell=True for backend, so we'll modify the run method
        manager.run_with_shell()
    except Exception as e:
        print(f"Error: {e}")
        manager.shutdown()
        sys.exit(1)


class ProcessManagerShell(ProcessManager):
    """Extended ProcessManager that supports shell commands."""
    
    def run_with_shell(self):
        """Run with shell support for backend."""
        self.log("MAIN", f"{Color.BOLD}=== Footsteps Development Server ==={Color.RESET}")
        self.log("MAIN", "Starting both backend and frontend processes...")
        
        # Start backend with shell=True
        if self.backend_restart_count >= self.max_restart_attempts:
            self.log("BACKEND", 
                    f"Max restart attempts ({self.max_restart_attempts}) reached. Stopping.",
                    "ERROR")
            self.running = False
            return False
            
        try:
            self.log("BACKEND", f"Starting: {self.backend_cmd}")
            self.backend_process = subprocess.Popen(
                self.backend_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=False,
                bufsize=1,
                shell=True
            )
            self.log("BACKEND", f"Started with PID {self.backend_process.pid}")
            self.backend_restart_count += 1
            
            # Start monitoring thread
            self.backend_monitor_thread = threading.Thread(
                target=self.monitor_backend, daemon=True
            )
            self.backend_monitor_thread.start()
        except Exception as e:
            self.log("BACKEND", f"Failed to start: {e}", "ERROR")
            return False
            
        time.sleep(1)
        
        # Start frontend
        frontend_started = self.start_frontend()
        
        self.log("MAIN", "Both processes started. Press Ctrl+C to stop.")
        
        # Main loop
        try:
            while self.running:
                self.print_logs()
                time.sleep(0.1)
        except KeyboardInterrupt:
            self.log("MAIN", "Keyboard interrupt received")
            
        return True


def main():
    """Main entry point."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Footsteps Development Server Manager",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 run_dev.py                              # Run with defaults
  python3 run_dev.py --debug                      # Enable debug logging
  python3 run_dev.py --video people.mp4           # Use video file
  python3 run_dev.py --debug --video people.mp4  # Debug with video
        """
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode (builds with --features debug)"
    )
    parser.add_argument(
        "--video",
        type=str,
        default=None,
        help="Path to video file for backend (or camera index)"
    )
    args = parser.parse_args()
    
    script_dir = Path(__file__).parent.resolve()
    os.chdir(script_dir)
    
    # Setup virtual environment
    print(f"\n{Color.BOLD}=== Setting up Python Virtual Environment ==={Color.RESET}")
    venv_python = setup_virtual_environment(script_dir)
    print(f"{Color.GREEN}Virtual environment ready: {venv_python}{Color.RESET}\n")
    
    # Build backend command with optional debug features
    if args.debug:
        print(f"{Color.YELLOW}Debug mode enabled{Color.RESET}")
        backend_build = "cargo build --release --features debug"
        backend_run = "cargo run --release --features debug"
    else:
        backend_build = "cargo build --release"
        backend_run = "cargo run --release"
    
    # Add video file argument if provided
    if args.video:
        backend_run = f"{backend_run} -- {args.video}"
        print(f"{Color.YELLOW}Video/camera input: {args.video}{Color.RESET}")

    # Keep UDP enabled by passing -u as the final backend argument.
    backend_run = f"{backend_run} -u"
    
    backend_cmd = f"{backend_build} && {backend_run}"
    
    # Frontend command (runs from footsteps_projection_mapping directory)
    frontend_cmd = [venv_python, "test_demo.py"]
    
    # Create manager
    manager = ProcessManagerShell(
        backend_cmd,
        frontend_cmd,
        max_restart_attempts=5,
        restart_delay=2
    )
    
    # Handle signals
    def signal_handler(sig, frame):
        manager.shutdown()
        sys.exit(0)
        
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        manager.run_with_shell()
    except Exception as e:
        print(f"Error: {e}")
        manager.shutdown()
        sys.exit(1)
    finally:
        manager.shutdown()


if __name__ == "__main__":
    main()
