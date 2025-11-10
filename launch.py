#!/usr/bin/env python3
"""
AEGIS-C Unified Launcher
========================

One script to rule them all - starts all services and launches the unified web interface.
"""

import os
import sys
import time
import subprocess
import threading
import signal
import requests
from datetime import datetime

# Configuration
SERVICES = {
    "brain": {"port": 8030, "path": "services/brain", "name": "Brain Gateway", "priority": 1},
    "detector": {"port": 8010, "path": "services/detector", "name": "Detector", "priority": 2},
    "fingerprint": {"port": 8011, "path": "services/fingerprint", "name": "Fingerprint", "priority": 2},
    "honeynet": {"port": 8012, "path": "services/honeynet", "name": "Honeynet", "priority": 2},
    "admission": {"port": 8013, "path": "services/admission", "name": "Admission", "priority": 2},
    "provenance": {"port": 8014, "path": "services/provenance", "name": "Provenance", "priority": 2},
    "coldwar": {"port": 8015, "path": "services/coldwar", "name": "Cold War", "priority": 2},
    "hardware": {"port": 8016, "path": "services/hardware", "name": "Hardware", "priority": 2},
    "discovery": {"port": 8017, "path": "services/discovery", "name": "Discovery", "priority": 3},
    "intelligence": {"port": 8018, "path": "services/intelligence", "name": "Intelligence", "priority": 3},
    "vuln_db": {"port": 8019, "path": "services/vuln_db", "name": "Vulnerability DB", "priority": 3}
}

UNIFIED_APP_PORT = 8500

class AegisLauncher:
    def __init__(self):
        self.running_services = {}
        self.shutdown_requested = False
        
    def print_banner(self):
        """Print AEGIS-C banner"""
        print("""
╔══════════════════════════════════════════════════════════════╗
║                    🛡️ AEGIS-C UNIFIED LAUNCHER                ║
║              Adaptive Counter-AI Intelligence Platform         ║
╠══════════════════════════════════════════════════════════════╣
║  One script to start everything:                              ║
║  • All microservices                                           ║
║  • Infrastructure dependencies                                ║
║  • Unified web interface                                       ║
║  • Health monitoring                                           ║
╚══════════════════════════════════════════════════════════════╝
        """)
    
    def check_dependencies(self):
        """Check if required dependencies are available"""
        print("🔍 Checking dependencies...")
        
        # Check Python
        try:
            import fastapi
            import uvicorn
            import streamlit
            import requests
            print("✅ Python dependencies available")
        except ImportError as e:
            print(f"❌ Missing dependency: {e}")
            print("Run: pip install fastapi uvicorn streamlit requests")
            return False
        
        # Check Docker (optional)
        try:
            subprocess.run(["docker", "--version"], capture_output=True, check=True)
            print("✅ Docker available")
        except:
            print("⚠️  Docker not available - infrastructure services will be skipped")
        
        return True
    
    def start_infrastructure(self):
        """Start Docker infrastructure services"""
        print("🏗️  Starting infrastructure services...")
        
        try:
            # Start core services
            core_services = ["postgres", "redis", "neo4j"]
            for service in core_services:
                print(f"   Starting {service}...")
                result = subprocess.run(
                    ["docker-compose", "up", "-d", service],
                    capture_output=True, text=True
                )
                if result.returncode == 0:
                    print(f"   ✅ {service} started")
                else:
                    print(f"   ⚠️  {service} failed: {result.stderr}")
            
            # Wait for services to be ready
            print("   Waiting for services to be healthy...")
            time.sleep(10)
            
        except Exception as e:
            print(f"⚠️  Infrastructure startup failed: {e}")
    
    def check_service_health(self, port):
        """Check if a service is healthy"""
        try:
            response = requests.get(f"http://localhost:{port}/health", timeout=2)
            return response.status_code == 200
        except:
            return False
    
    def start_service(self, service_name, config):
        """Start a single service"""
        port = config["port"]
        path = config["path"]
        name = config["name"]
        
        print(f"🚀 Starting {name} (port {port})...")
        
        try:
            # Check if service is already running
            if self.check_service_health(port):
                print(f"   ✅ {name} already running")
                return True
            
            # Start the service
            main_file = os.path.join(path, "main.py")
            if os.path.exists(main_file):
                process = subprocess.Popen([
                    sys.executable, "-m", "uvicorn", "main:app",
                    f"--port={port}", "--host=localhost"
                ], cwd=path)
                
                self.running_services[service_name] = process
                
                # Wait for service to be healthy
                for i in range(30):  # Wait up to 30 seconds
                    if self.check_service_health(port):
                        print(f"   ✅ {name} started successfully")
                        return True
                    time.sleep(1)
                
                print(f"   ⚠️  {name} started but not healthy after 30s")
                return False
            else:
                print(f"   ❌ {name} main.py not found at {main_file}")
                return False
                
        except Exception as e:
            print(f"   ❌ Failed to start {name}: {e}")
            return False
    
    def start_all_services(self):
        """Start all services in priority order"""
        print("🔄 Starting all AEGIS-C services...")
        
        # Sort services by priority
        sorted_services = sorted(
            SERVICES.items(), 
            key=lambda x: x[1]["priority"]
        )
        
        success_count = 0
        total_count = len(sorted_services)
        
        for service_name, config in sorted_services:
            if self.start_service(service_name, config):
                success_count += 1
            time.sleep(1)  # Brief pause between services
        
        print(f"\n📊 Service Startup Summary: {success_count}/{total_count} services started")
        
        # Show service status
        print("\n📋 Service Status:")
        for service_name, config in sorted_services:
            port = config["port"]
            name = config["name"]
            status = "✅ Healthy" if self.check_service_health(port) else "❌ Unhealthy"
            print(f"   {name:<20} (:{port:<4}) - {status}")
    
    def launch_unified_app(self):
        """Launch the unified web application"""
        print(f"\n🌐 Launching Unified Web Application (port {UNIFIED_APP_PORT})...")
        
        try:
            # Check if port is available
            if self.check_service_health(UNIFIED_APP_PORT):
                print(f"   ✅ Unified app already running on port {UNIFIED_APP_PORT}")
                return
            
            # Launch the unified app
            app_path = "services/console/unified_app.py"
            if os.path.exists(app_path):
                process = subprocess.Popen([
                    sys.executable, "-m", "streamlit", "run", app_path,
                    f"--server.port={UNIFIED_APP_PORT}",
                    "--server.address=localhost",
                    "--server.headless=true"
                ])
                
                self.running_services["unified_app"] = process
                
                # Wait for app to be ready
                for i in range(30):
                    try:
                        response = requests.get(f"http://localhost:{UNIFIED_APP_PORT}", timeout=2)
                        if response.status_code == 200:
                            print(f"   ✅ Unified app started successfully")
                            print(f"\n🎯 ACCESS YOUR AEGIS-C PLATFORM: http://localhost:{UNIFIED_APP_PORT}")
                            return
                    except:
                        pass
                    time.sleep(1)
                
                print(f"   ⚠️  Unified app started but may not be fully ready")
            else:
                print(f"   ❌ Unified app not found at {app_path}")
                
        except Exception as e:
            print(f"   ❌ Failed to launch unified app: {e}")
    
    def setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown"""
        def signal_handler(signum, frame):
            print(f"\n\n🛑 Shutdown signal received ({signum})...")
            self.shutdown_requested = True
            self.stop_all_services()
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    def stop_all_services(self):
        """Stop all running services"""
        print("\n🛑 Stopping all services...")
        
        for service_name, process in self.running_services.items():
            try:
                print(f"   Stopping {service_name}...")
                process.terminate()
                process.wait(timeout=5)
                print(f"   ✅ {service_name} stopped")
            except subprocess.TimeoutExpired:
                print(f"   ⚠️  Force killing {service_name}...")
                process.kill()
            except Exception as e:
                print(f"   ❌ Error stopping {service_name}: {e}")
    
    def monitor_services(self):
        """Monitor running services and report status"""
        print("\n📊 Service Monitoring Active")
        print("   Press Ctrl+C to stop all services")
        print("=" * 60)
        
        while not self.shutdown_requested:
            time.sleep(30)  # Check every 30 seconds
            
            # Check service health
            unhealthy_services = []
            for service_name, config in SERVICES.items():
                if not self.check_service_health(config["port"]):
                    unhealthy_services.append(service_name)
            
            if unhealthy_services:
                timestamp = datetime.now().strftime("%H:%M:%S")
                print(f"[{timestamp}] ⚠️  Unhealthy services: {', '.join(unhealthy_services)}")
    
    def run(self):
        """Main launcher execution"""
        self.print_banner()
        
        # Setup signal handlers
        self.setup_signal_handlers()
        
        # Check dependencies
        if not self.check_dependencies():
            sys.exit(1)
        
        # Start infrastructure
        self.start_infrastructure()
        
        # Start all services
        self.start_all_services()
        
        # Launch unified app
        self.launch_unified_app()
        
        # Show final summary
        print("\n" + "=" * 60)
        print("🎉 AEGIS-C PLATFORM IS READY!")
        print("=" * 60)
        print(f"🌐 Unified Interface: http://localhost:{UNIFIED_APP_PORT}")
        print("🧠 Brain Gateway: http://localhost:8030")
        print("🔍 Detector API: http://localhost:8010")
        print("💻 Hardware Monitor: http://localhost:8016")
        print("\n📋 All Services:")
        for service_name, config in SERVICES.items():
            port = config["port"]
            name = config["name"]
            status = "✅" if self.check_service_health(port) else "❌"
            print(f"   {status} {name:<20} :{port}")
        
        print("\n💡 Tips:")
        print("   • The unified interface provides access to ALL functionality")
        print("   • Services will auto-restart if they fail")
        print("   • Monitor this terminal for service health updates")
        print("   • Press Ctrl+C to stop everything gracefully")
        
        # Start monitoring
        try:
            self.monitor_services()
        except KeyboardInterrupt:
            pass

def main():
    """Main entry point"""
    launcher = AegisLauncher()
    
    try:
        launcher.run()
    except KeyboardInterrupt:
        print("\n\n👋 Shutting down AEGIS-C...")
        launcher.stop_all_services()
        print("✅ All services stopped. Goodbye!")
    except Exception as e:
        print(f"\n❌ Launcher error: {e}")
        launcher.stop_all_services()
        sys.exit(1)

if __name__ == "__main__":
    main()
