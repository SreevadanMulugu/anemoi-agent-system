#!/usr/bin/env python3
"""
Google Colab setup script for Anemoi Semi-Centralized Multi-Agent System
Automatically installs Ollama and required models in Colab environment.
"""

import os
import subprocess
import time
import requests
import asyncio
from pathlib import Path

def install_ollama_colab():
    """Install Ollama in Google Colab environment."""
    print("🚀 Setting up Ollama in Google Colab...")
    
    # Install Ollama
    print("📦 Installing Ollama...")
    subprocess.run([
        "curl -fsSL https://ollama.com/install.sh | sh"
    ], shell=True, check=True)
    
    # Start Ollama service
    print("🔄 Starting Ollama service...")
    subprocess.Popen([
        "ollama", "serve"
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    # Wait for service to start
    time.sleep(5)
    
    # Verify installation
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=10)
        if response.status_code == 200:
            print("✅ Ollama installed and running successfully!")
            return True
    except:
        print("❌ Failed to start Ollama service")
        return False

def setup_colab_environment():
    """Setup the Colab environment."""
    print("🔧 Setting up Colab environment...")
    
    # Install Python dependencies
    subprocess.run([
        "pip", "install", "requests", "asyncio"
    ], check=True)
    
    # Create necessary directories
    os.makedirs("memory", exist_ok=True)
    os.makedirs("client", exist_ok=True)
    
    print("✅ Colab environment setup complete!")

async def main():
    """Main setup function for Colab."""
    print("🌟 Anemoi Semi-Centralized Multi-Agent System - Colab Setup")
    print("=" * 60)
    
    # Setup environment
    setup_colab_environment()
    
    # Install Ollama
    if not install_ollama_colab():
        print("❌ Failed to install Ollama")
        return
    
    # Pull models
    if not await pull_models():
        print("❌ Failed to pull required models")
        return
    
    print("\n🎉 Setup complete! You can now run the Anemoi system.")
    print("\n📝 Next steps:")
    print("1. Run: python start_anemoi.py --interactive")
    print("2. Or: python start_anemoi.py -q 'Your question here'")

if __name__ == "__main__":
    asyncio.run(main())






