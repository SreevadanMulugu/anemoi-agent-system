#!/usr/bin/env python3
"""
One-command startup script for Anemoi Semi-Centralized Multi-Agent System
Automatically handles Ollama installation, model pulling, and system startup.
"""

import asyncio
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

def check_ollama_installed():
    """Check if Ollama is installed."""
    try:
        result = subprocess.run(["ollama", "--version"], 
                              capture_output=True, text=True, timeout=5)
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False

def install_ollama():
    """Install Ollama automatically."""
    system = platform.system().lower()
    
    print("Installing Ollama...")
    
    if system == "windows":
        # Download and run Ollama installer for Windows
        import urllib.request
        url = "https://ollama.com/download/windows"
        print("Please download and install Ollama from: https://ollama.com/download/windows")
        print("After installation, restart this script.")
        return False
    
    elif system == "darwin":  # macOS
        try:
            subprocess.run(["brew", "install", "ollama"], check=True)
            return True
        except subprocess.CalledProcessError:
            print("Please install Ollama manually: https://ollama.com/download/mac")
            return False
    
    elif system == "linux":
        try:
            # Install Ollama on Linux
            subprocess.run([
                "curl -fsSL https://ollama.com/install.sh | sh"
            ], shell=True, check=True)
            return True
        except subprocess.CalledProcessError:
            print("Please install Ollama manually: https://ollama.com/download/linux")
            return False
    
    else:
        print(f"Unsupported system: {system}")
        print("Please install Ollama manually: https://ollama.com/download")
        return False

def start_ollama_service():
    """Start Ollama service."""
    print("Starting Ollama service...")
    try:
        # Start Ollama in background
        subprocess.Popen(["ollama", "serve"], 
                        stdout=subprocess.DEVNULL, 
                        stderr=subprocess.DEVNULL)
        time.sleep(3)  # Wait for service to start
        return True
    except Exception as e:
        print(f"Failed to start Ollama: {e}")
        return False

async def ensure_models():
    """Ensure required models are available."""
    import requests
    
    models = {
        "openbmb/minicpm4.1": "Meta-planner model",
        "gpt-oss-20b": "Executor model"
    }
    
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=10)
        if response.status_code == 200:
            available_models = {model["name"] for model in response.json().get("models", [])}
            
            for model_name, description in models.items():
                if model_name not in available_models:
                    print(f"Pulling {model_name} ({description})...")
                    process = await asyncio.create_subprocess_exec(
                        "ollama", "pull", model_name,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE
                    )
                    stdout, stderr = await process.communicate()
                    
                    if process.returncode == 0:
                        print(f"✓ Successfully pulled {model_name}")
                    else:
                        print(f"✗ Failed to pull {model_name}: {stderr.decode()}")
                        return False
                else:
                    print(f"✓ {model_name} already available")
            
            return True
    except Exception as e:
        print(f"Error checking models: {e}")
        return False

def create_requirements():
    """Create requirements.txt if it doesn't exist."""
    requirements = """requests>=2.31.0
asyncio
"""
    
    req_file = Path("requirements.txt")
    if not req_file.exists():
        with open(req_file, "w") as f:
            f.write(requirements)
        print("Created requirements.txt")

async def main():
    """Main startup function."""
    print("🚀 Anemoi Semi-Centralized Multi-Agent System")
    print("=" * 50)
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ required")
        sys.exit(1)
    
    # Create requirements file
    create_requirements()
    
    # Check if Ollama is installed
    if not check_ollama_installed():
        print("❌ Ollama not found")
        if not install_ollama():
            print("Please install Ollama manually and restart this script.")
            sys.exit(1)
    
    # Start Ollama service
    if not start_ollama_service():
        print("❌ Failed to start Ollama service")
        sys.exit(1)
    
    # Ensure models are available
    print("\n📦 Checking required models...")
    if not await ensure_models():
        print("❌ Failed to ensure required models")
        sys.exit(1)
    
    # Start the Anemoi system
    print("\n🤖 Starting Anemoi system...")
    
    # Import and run the main system
    try:
        from client.anemoi_agent import AnemoiSystem
        import argparse
        
        parser = argparse.ArgumentParser(description="Anemoi System")
        parser.add_argument("-q", "--query", type=str, help="Query to process")
        parser.add_argument("--interactive", action="store_true", help="Interactive mode")
        args = parser.parse_args()
        
        system = AnemoiSystem()
        await system.initialize()
        
        if args.query:
            result = await system.process_query(args.query)
            print(f"\n🎯 Result: {result}")
        elif args.interactive:
            print("\n💬 Interactive Mode - Type 'exit' to quit")
            while True:
                query = input("\n❓ Query: ").strip()
                if query.lower() in ["exit", "quit", "q"]:
                    break
                result = await system.process_query(query)
                print(f"\n🎯 Result: {result}")
        else:
            print("\n💡 Usage examples:")
            print("  python start_anemoi.py -q 'What is 2+2?'")
            print("  python start_anemoi.py --interactive")
        
        await system.shutdown()
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure you're in the correct directory with client/anemoi_agent.py")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())

