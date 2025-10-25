#!/usr/bin/env python3
"""
Quick Fix Script for Resume Parser API
This script helps resolve common setup issues
"""

import os
import sys
import subprocess

def install_dependencies():
    """Install required dependencies"""
    print("🔧 Installing dependencies...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False

def create_env_file():
    """Create a basic .env file if it doesn't exist"""
    env_file = ".env"
    if os.path.exists(env_file):
        print(f"✅ {env_file} already exists")
        return True
    
    print(f"📝 Creating {env_file}...")
    
    env_content = """# Resume Parser API - Environment Configuration

# API Configuration
HOST=0.0.0.0
PORT=2010
API_BASE_URL=http://localhost:2010
FRONTEND_URL=http://localhost:3000

# Database Configuration
MONGODB_URL=mongodb://localhost:27017
MONGODB_DATABASE=resume_parser

# JWT Authentication Configuration
JWT_SECRET_KEY=your-super-secret-jwt-key-change-this-in-production
JWT_ALGORITHM=HS256
JWT_ACCESS_TOKEN_EXPIRE_MINUTES=1440

# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key_here

# IONOS SMTP Configuration
SMTP_SERVER=smtp.ionos.com
SMTP_PORT=587
SMTP_USERNAME=your-email@yourdomain.com
SMTP_PASSWORD=your-email-password
FROM_EMAIL=your-email@yourdomain.com
FROM_NAME=Resume Parser API
ADMIN_EMAIL=admin@yourdomain.com

# Admin Configuration
ADMIN_USERNAME=admin
ADMIN_EMAIL=admin@yourdomain.com
ADMIN_PASSWORD=admin123

# Development Settings
DEBUG=false
ENVIRONMENT=development
LOG_LEVEL=INFO

# Security Configuration
ALLOWED_ORIGINS=http://localhost:3000
MAX_FILE_SIZE=10485760
RATE_LIMIT_ENABLED=true
RATE_LIMIT_REQUESTS_PER_MINUTE=60

# Subscription Tiers Configuration
FREE_TIER_DAILY_LIMIT=10
FREE_TIER_MONTHLY_LIMIT=100
FREE_TIER_FILE_SIZE_LIMIT=5242880

BASIC_TIER_DAILY_LIMIT=50
BASIC_TIER_MONTHLY_LIMIT=1000
BASIC_TIER_FILE_SIZE_LIMIT=10485760

PREMIUM_TIER_DAILY_LIMIT=200
PREMIUM_TIER_MONTHLY_LIMIT=5000
PREMIUM_TIER_FILE_SIZE_LIMIT=26214400

ENTERPRISE_TIER_DAILY_LIMIT=-1
ENTERPRISE_TIER_MONTHLY_LIMIT=-1
ENTERPRISE_TIER_FILE_SIZE_LIMIT=52428800

# Logging Configuration
LOG_LEVEL=INFO
LOG_FILE=logs/resume_parser.log
LOG_MAX_SIZE=10485760
LOG_BACKUP_COUNT=5

# Monitoring & Analytics
USAGE_ALERT_THRESHOLD=80
USAGE_ALERT_EMAIL_ENABLED=true
ANALYTICS_RETENTION_DAYS=365
ANALYTICS_CLEANUP_ENABLED=true

# SSL/TLS Configuration
SSL_ENABLED=false
SSL_CERT_PATH=/path/to/cert.pem
SSL_KEY_PATH=/path/to/key.pem

# Backup Configuration
BACKUP_ENABLED=true
BACKUP_SCHEDULE=0 2 * * *
BACKUP_RETENTION_DAYS=30
BACKUP_STORAGE_PATH=/backups

# External Services
REDIS_URL=redis://localhost:6379
REDIS_ENABLED=false
SENTRY_DSN=your-sentry-dsn-here
SENTRY_ENABLED=false

# API Documentation
API_DOCS_ENABLED=true
API_DOCS_URL=/docs
API_REDOC_URL=/redoc

# Health Check Configuration
HEALTH_CHECK_ENABLED=true
HEALTH_CHECK_INTERVAL=300
HEALTH_CHECK_TIMEOUT=30
"""
    
    try:
        with open(env_file, 'w') as f:
            f.write(env_content)
        print(f"✅ {env_file} created successfully")
        print("📝 Please edit the .env file with your actual values:")
        print("   - MONGODB_URL (if using MongoDB Atlas)")
        print("   - OPENAI_API_KEY")
        print("   - SMTP credentials for IONOS")
        print("   - JWT_SECRET_KEY (change from default)")
        return True
    except Exception as e:
        print(f"❌ Failed to create {env_file}: {e}")
        return False

def check_mongodb():
    """Check if MongoDB is running"""
    print("🔍 Checking MongoDB connection...")
    try:
        import pymongo
        client = pymongo.MongoClient("mongodb://localhost:27017", serverSelectionTimeoutMS=5000)
        client.server_info()
        print("✅ MongoDB is running and accessible")
        return True
    except Exception as e:
        print(f"⚠️  MongoDB connection failed: {e}")
        print("💡 Please start MongoDB:")
        print("   - Windows: Run 'mongod' in command prompt")
        print("   - Linux/Mac: Run 'sudo systemctl start mongod' or 'brew services start mongodb'")
        return False

def main():
    """Main setup function"""
    print("🚀 Resume Parser API - Quick Fix Script")
    print("=" * 50)
    
    # Step 1: Install dependencies
    if not install_dependencies():
        print("❌ Setup failed at dependency installation")
        return False
    
    # Step 2: Create .env file
    if not create_env_file():
        print("❌ Setup failed at .env file creation")
        return False
    
    # Step 3: Check MongoDB
    check_mongodb()
    
    print("\n" + "=" * 50)
    print("✅ Quick fix completed!")
    print("\n📋 Next steps:")
    print("1. Edit .env file with your actual values")
    print("2. Start MongoDB: mongod")
    print("3. Run the API: python main.py")
    print("4. Visit: http://localhost:2010/docs")
    print("\n🔧 Common fixes:")
    print("- Change JWT_SECRET_KEY from default value")
    print("- Set your OPENAI_API_KEY")
    print("- Configure IONOS SMTP credentials")
    print("- Ensure MongoDB is running")
    
    return True

if __name__ == "__main__":
    main()

