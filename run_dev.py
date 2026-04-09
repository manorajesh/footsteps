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

import asyncio
import os
import sys
import argparse
import subprocess
import venv
from pathlib import Path
from datetime import datetime

class Color:
    RESET = '\033[0m'
    DIM = '\033[2m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    BOLD = '\033[1m'

def setup_virtual_environment(project_root: Path) -> str:
    """Initialize virtual environment and install dependencies."""
    venv_path = project_root / "venv"
    python_exec = venv_path / "bin" / "python"
    pip_exec = venv_path / "bin" / "pip"
    
    if not venv_path.exists():
        print(f"{Color.YELLOW}Creating virtual environment at {venv_path}...{Color.RESET}")
        venv.create(venv_path, with_pip=True)
    
    print(f"{Color.YELLOW}Installing/Upgrading requirements...{Color.RESET}")
    subprocess.run([str(python_exec), "-m", "pip", "install", "--upgrade", "pip", "--quiet"], check=True)
    
    req_file = project_root / "footsteps_projection_mapping" / "requirements.txt"
    if req_file.exists():
        subprocess.run([str(pip_exec), "install", "-r", str(req_file), "--quiet"], check=True)
    
    return str(python_exec)

async def read_stream(stream, name, color):
    """Read lines from async stream and print with timestamps and colors."""
    while True:
        line = await stream.readline()
        if not line:
            break
        text = line.decode(errors='replace').rstrip()
        ts = datetime.now().strftime("%H:%M:%S")
        print(f"{Color.DIM}{ts}{Color.RESET} {color}[{name:8}]{Color.RESET} {text}")

async def run_process(cmd, name, color, cwd=None, is_shell=False):
    """Run a process, log its output, and restart it if it exits."""
    while True:
        proc = None
        try:
            if is_shell:
                proc = await asyncio.create_subprocess_shell(
                    cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT, cwd=cwd
                )
            else:
                proc = await asyncio.create_subprocess_exec(
                    *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT, cwd=cwd
                )
            
            await read_stream(proc.stdout, name, color)
            await proc.wait()
            
            print(f"{Color.YELLOW}[{name:8}] Process exited with code {proc.returncode}, restarting in 2s...{Color.RESET}")
            await asyncio.sleep(2)
            
        except asyncio.CancelledError:
            if proc and proc.returncode is None:
                proc.terminate()
                try:
                    await asyncio.wait_for(proc.wait(), timeout=5.0)
                except asyncio.TimeoutError:
                    proc.kill()
            raise

async def main_async():
    parser = argparse.ArgumentParser(description="Footsteps Development Server")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode (--features debug)")
    parser.add_argument("--video", type=str, help="Path to video file for backend (or camera index)")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    os.chdir(project_root)

    print(f"\n{Color.BOLD}=== Setting up Python Virtual Environment ==={Color.RESET}")
    python_exec = setup_virtual_environment(project_root)
    print(f"{Color.GREEN}Virtual environment ready.{Color.RESET}\n")

    print(f"{Color.YELLOW}Building backend...{Color.RESET}")
    build_cmd = ["cargo", "build", "--release"]
    run_cmd = ["cargo", "run", "--release"]
    
    if args.debug:
        build_cmd.append("--features=debug")
        run_cmd.append("--features=debug")
        
    subprocess.run(build_cmd, check=True)
    
    run_cmd.append("--")
    if args.video:
        run_cmd.append(args.video)
    run_cmd.append("-u")
    
    frontend_cmd = [python_exec, "test_demo.py"]
    frontend_cwd = project_root / "footsteps_projection_mapping"

    print(f"{Color.GREEN}Starting processes... Press Ctrl+C to stop.{Color.RESET}")
    
    # Run the backend using shell to allow argument expansion comfortably if needed, or just exec.
    # The original used `cargo built && cargo run`, but we handle build synchronously first above.
    tasks = [
        asyncio.create_task(run_process(run_cmd, "BACKEND", Color.BLUE, cwd=project_root, is_shell=False)),
        asyncio.create_task(run_process(frontend_cmd, "FRONTEND", Color.GREEN, cwd=frontend_cwd, is_shell=False))
    ]
    
    try:
        await asyncio.gather(*tasks)
    except asyncio.CancelledError:
        print(f"\n{Color.YELLOW}Graceful shutdown requested...{Color.RESET}")

def main():
    try:
        asyncio.run(main_async())
    except KeyboardInterrupt:
        print(f"\n{Color.YELLOW}Shutting down...{Color.RESET}")
    except subprocess.CalledProcessError as e:
        print(f"\n{Color.RED}Build failed: {e}{Color.RESET}")
        sys.exit(1)

if __name__ == "__main__":
    main()
