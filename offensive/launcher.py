#!/usr/bin/env python3
"""
AEGIS‑C Cold War Offensive Launcher
====================================

Quick launcher for the offensive toolkit dashboard and CLI tools
"""

import subprocess
import sys
import os
import argparse
import asyncio
from pathlib import Path

def install_dependencies():
    """Install required dependencies"""
    print("📦 Installing offensive toolkit dependencies...")
    
    try:
        subprocess.run([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ], check=True)
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False

def start_dashboard():
    """Launch the Streamlit dashboard"""
    print("🚀 Starting Cold War Offensive Dashboard...")
    
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "dashboard.py", 
            "--server.port", "8502", "--server.address", "0.0.0.0"
        ], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to start dashboard: {e}")
        return False
    except KeyboardInterrupt:
        print("\n🛑 Dashboard stopped by user")
        return True

def run_cli_campaign():
    """Run a command-line campaign"""
    print("🎯 Running CLI Cold War Campaign...")
    
    try:
        # Import and run the campaign
        from coldwar_toolkit import main as campaign_main
        asyncio.run(campaign_main())
        return True
    except Exception as e:
        print(f"❌ Campaign failed: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="AEGIS‑C Cold War Offensive Launcher")
    parser.add_argument("--mode", choices=["dashboard", "cli", "install"], 
                       default="dashboard", help="Launch mode")
    parser.add_argument("--port", type=int, default=8502, help="Dashboard port")
    
    args = parser.parse_args()
    
    print("⚔️  AEGIS‑C Cold War Offensive Toolkit")
    print("🎯 For authorized red-team testing only")
    print("=" * 50)
    
    # Change to offensive directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    if args.mode == "install":
        success = install_dependencies()
        sys.exit(0 if success else 1)
    
    elif args.mode == "cli":
        success = run_cli_campaign()
        sys.exit(0 if success else 1)
    
    elif args.mode == "dashboard":
        # Check if dependencies are installed
        try:
            import streamlit
            import aiohttp
            import plotly
        except ImportError:
            print("❌ Dependencies not found. Installing...")
            if not install_dependencies():
                sys.exit(1)
        
        print(f"🌐 Dashboard will be available at: http://localhost:{args.port}")
        print("🛑 Press Ctrl+C to stop the dashboard")
        print()
        
        start_dashboard()

if __name__ == "__main__":
    main()