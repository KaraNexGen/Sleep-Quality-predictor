
import os
import sys
import subprocess
from pathlib import Path

def check_dependencies():
    """Check if all required dependencies are installed"""
    try:
        import flask
        import sklearn
        import pandas
        import numpy
        import matplotlib
        import seaborn
        print("✅ All dependencies are installed")
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Please run: pip install -r requirements.txt")
        return False

def create_directories():
    """Create necessary directories if they don't exist"""
    directories = ['models', 'static/css', 'static/js', 'static/images', 'templates']
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"📁 Directory created/verified: {directory}")

def generate_sample_data():
    """Generate sample data for demonstration"""
    try:
        from data_generator import SleepDataGenerator
        
        print("🔄 Generating sample sleep data...")
        generator = SleepDataGenerator()
        df = generator.generate_realistic_dataset(500)
        df.to_csv('sample_sleep_data.csv', index=False)
        print("✅ Sample data generated: sample_sleep_data.csv")
        
    except Exception as e:
        print(f"⚠️  Could not generate sample data: {e}")

def main():
    """Main application runner"""
    print("🌙 Sleep Quality Predictor - Starting Application")
    print("=" * 50)
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Create directories
    create_directories()
    
    # Generate sample data
    generate_sample_data()
    
    # Start the Flask application
    print("\n🚀 Starting Flask application...")
    print("📍 Application will be available at: http://localhost:5000")
    print("🛑 Press Ctrl+C to stop the application")
    print("=" * 50)
    
    try:
        from app import app
        app.run(debug=True, host='0.0.0.0', port=5000)
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user")
    except Exception as e:
        print(f"❌ Error starting application: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
