import os
from typing import List, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Settings:
    """Application settings and configuration"""
    
    # ===========================================
    # API Configuration
    # ===========================================
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "2010"))
    API_BASE_URL: str = os.getenv("API_BASE_URL", "http://localhost:2010")
    FRONTEND_URL: str = os.getenv("FRONTEND_URL", "http://localhost:3000")
    
    # ===========================================
    # Database Configuration
    # ===========================================
    MONGODB_URL: str = os.getenv("MONGODB_URL", "mongodb://localhost:27017")
    MONGODB_DATABASE: str = os.getenv("MONGODB_DATABASE", "resume_parser")
    
    # ===========================================
    # JWT Authentication Configuration
    # ===========================================
    JWT_SECRET_KEY: str = os.getenv("JWT_SECRET_KEY", "your-super-secret-jwt-key-change-this-in-production")
    JWT_ALGORITHM: str = os.getenv("JWT_ALGORITHM", "HS256")
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES: int = int(os.getenv("JWT_ACCESS_TOKEN_EXPIRE_MINUTES", "1440"))
    
    # ===========================================
    # OpenAI Configuration
    # ===========================================
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "your_openai_api_key_here")
    
    # ===========================================
    # Email Service Configuration
    # ===========================================
    SMTP_SERVER: str = os.getenv("SMTP_SERVER", "smtp.ionos.com")
    SMTP_PORT: int = int(os.getenv("SMTP_PORT", "587"))
    SMTP_USERNAME: Optional[str] = os.getenv("SMTP_USERNAME")
    SMTP_PASSWORD: Optional[str] = os.getenv("SMTP_PASSWORD")
    FROM_EMAIL: str = os.getenv("FROM_EMAIL", SMTP_USERNAME or "noreply@resumeparser.com")
    FROM_NAME: str = os.getenv("FROM_NAME", "Resume Parser API")
    ADMIN_EMAIL: Optional[str] = os.getenv("ADMIN_EMAIL")
    
    # ===========================================
    # Security Configuration
    # ===========================================
    ALLOWED_ORIGINS: List[str] = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000").split(",")
    MAX_FILE_SIZE: int = int(os.getenv("MAX_FILE_SIZE", "10485760"))  # 10MB
    RATE_LIMIT_ENABLED: bool = os.getenv("RATE_LIMIT_ENABLED", "true").lower() == "true"
    RATE_LIMIT_REQUESTS_PER_MINUTE: int = int(os.getenv("RATE_LIMIT_REQUESTS_PER_MINUTE", "60"))
    
    # ===========================================
    # Subscription Tiers Configuration
    # ===========================================
    # Free Tier
    FREE_TIER_DAILY_LIMIT: int = int(os.getenv("FREE_TIER_DAILY_LIMIT", "10"))
    FREE_TIER_MONTHLY_LIMIT: int = int(os.getenv("FREE_TIER_MONTHLY_LIMIT", "100"))
    FREE_TIER_FILE_SIZE_LIMIT: int = int(os.getenv("FREE_TIER_FILE_SIZE_LIMIT", "5242880"))  # 5MB
    
    # Basic Tier
    BASIC_TIER_DAILY_LIMIT: int = int(os.getenv("BASIC_TIER_DAILY_LIMIT", "50"))
    BASIC_TIER_MONTHLY_LIMIT: int = int(os.getenv("BASIC_TIER_MONTHLY_LIMIT", "1000"))
    BASIC_TIER_FILE_SIZE_LIMIT: int = int(os.getenv("BASIC_TIER_FILE_SIZE_LIMIT", "10485760"))  # 10MB
    
    # Premium Tier
    PREMIUM_TIER_DAILY_LIMIT: int = int(os.getenv("PREMIUM_TIER_DAILY_LIMIT", "200"))
    PREMIUM_TIER_MONTHLY_LIMIT: int = int(os.getenv("PREMIUM_TIER_MONTHLY_LIMIT", "5000"))
    PREMIUM_TIER_FILE_SIZE_LIMIT: int = int(os.getenv("PREMIUM_TIER_FILE_SIZE_LIMIT", "26214400"))  # 25MB
    
    # Enterprise Tier
    ENTERPRISE_TIER_DAILY_LIMIT: int = int(os.getenv("ENTERPRISE_TIER_DAILY_LIMIT", "-1"))  # Unlimited
    ENTERPRISE_TIER_MONTHLY_LIMIT: int = int(os.getenv("ENTERPRISE_TIER_MONTHLY_LIMIT", "-1"))  # Unlimited
    ENTERPRISE_TIER_FILE_SIZE_LIMIT: int = int(os.getenv("ENTERPRISE_TIER_FILE_SIZE_LIMIT", "52428800"))  # 50MB
    
    # ===========================================
    # Logging Configuration
    # ===========================================
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE: Optional[str] = os.getenv("LOG_FILE")
    LOG_MAX_SIZE: int = int(os.getenv("LOG_MAX_SIZE", "10485760"))  # 10MB
    LOG_BACKUP_COUNT: int = int(os.getenv("LOG_BACKUP_COUNT", "5"))
    
    # ===========================================
    # Monitoring & Analytics
    # ===========================================
    USAGE_ALERT_THRESHOLD: int = int(os.getenv("USAGE_ALERT_THRESHOLD", "80"))
    USAGE_ALERT_EMAIL_ENABLED: bool = os.getenv("USAGE_ALERT_EMAIL_ENABLED", "true").lower() == "true"
    ANALYTICS_RETENTION_DAYS: int = int(os.getenv("ANALYTICS_RETENTION_DAYS", "365"))
    ANALYTICS_CLEANUP_ENABLED: bool = os.getenv("ANALYTICS_CLEANUP_ENABLED", "true").lower() == "true"
    
    # ===========================================
    # Development Settings
    # ===========================================
    DEBUG: bool = os.getenv("DEBUG", "false").lower() == "true"
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "production")
    
    # ===========================================
    # SSL/TLS Configuration
    # ===========================================
    SSL_ENABLED: bool = os.getenv("SSL_ENABLED", "false").lower() == "true"
    SSL_CERT_PATH: Optional[str] = os.getenv("SSL_CERT_PATH")
    SSL_KEY_PATH: Optional[str] = os.getenv("SSL_KEY_PATH")
    
    # ===========================================
    # Backup Configuration
    # ===========================================
    BACKUP_ENABLED: bool = os.getenv("BACKUP_ENABLED", "true").lower() == "true"
    BACKUP_SCHEDULE: str = os.getenv("BACKUP_SCHEDULE", "0 2 * * *")
    BACKUP_RETENTION_DAYS: int = int(os.getenv("BACKUP_RETENTION_DAYS", "30"))
    BACKUP_STORAGE_PATH: str = os.getenv("BACKUP_STORAGE_PATH", "/backups")
    
    # ===========================================
    # External Services
    # ===========================================
    REDIS_URL: Optional[str] = os.getenv("REDIS_URL")
    REDIS_ENABLED: bool = os.getenv("REDIS_ENABLED", "false").lower() == "true"
    SENTRY_DSN: Optional[str] = os.getenv("SENTRY_DSN")
    SENTRY_ENABLED: bool = os.getenv("SENTRY_ENABLED", "false").lower() == "true"
    
    # ===========================================
    # Admin Configuration
    # ===========================================
    ADMIN_USERNAME: str = os.getenv("ADMIN_USERNAME", "admin")
    ADMIN_EMAIL: str = os.getenv("ADMIN_EMAIL", "admin@yourcompany.com")
    ADMIN_PASSWORD: str = os.getenv("ADMIN_PASSWORD", "admin123")
    
    # ===========================================
    # API Documentation
    # ===========================================
    API_DOCS_ENABLED: bool = os.getenv("API_DOCS_ENABLED", "true").lower() == "true"
    API_DOCS_URL: str = os.getenv("API_DOCS_URL", "/docs")
    API_REDOC_URL: str = os.getenv("API_REDOC_URL", "/redoc")
    
    # ===========================================
    # Health Check Configuration
    # ===========================================
    HEALTH_CHECK_ENABLED: bool = os.getenv("HEALTH_CHECK_ENABLED", "true").lower() == "true"
    HEALTH_CHECK_INTERVAL: int = int(os.getenv("HEALTH_CHECK_INTERVAL", "300"))
    HEALTH_CHECK_TIMEOUT: int = int(os.getenv("HEALTH_CHECK_TIMEOUT", "30"))
    
    # ===========================================
    # Validation Methods
    # ===========================================
    @classmethod
    def validate_config(cls) -> List[str]:
        """Validate configuration and return list of issues"""
        issues = []
        
        # Check required fields
        if cls.JWT_SECRET_KEY == "your-super-secret-jwt-key-change-this-in-production":
            issues.append("JWT_SECRET_KEY should be changed from default value")
        
        if cls.OPENAI_API_KEY == "your_openai_api_key_here":
            issues.append("OPENAI_API_KEY should be set to a valid OpenAI API key")
        
        if not cls.SMTP_USERNAME or not cls.SMTP_PASSWORD:
            issues.append("SMTP credentials not configured - email features will be disabled")
        
        if not cls.ADMIN_EMAIL:
            issues.append("ADMIN_EMAIL not configured - admin notifications will be disabled")
        
        if cls.ENVIRONMENT == "production" and cls.DEBUG:
            issues.append("DEBUG should be False in production environment")
        
        if cls.ENVIRONMENT == "production" and not cls.SSL_ENABLED:
            issues.append("SSL should be enabled in production environment")
        
        return issues
    
    @classmethod
    def get_subscription_limits(cls, tier: str) -> dict:
        """Get limits for a subscription tier"""
        limits = {
            "free": {
                "daily_limit": cls.FREE_TIER_DAILY_LIMIT,
                "monthly_limit": cls.FREE_TIER_MONTHLY_LIMIT,
                "file_size_limit": cls.FREE_TIER_FILE_SIZE_LIMIT
            },
            "basic": {
                "daily_limit": cls.BASIC_TIER_DAILY_LIMIT,
                "monthly_limit": cls.BASIC_TIER_MONTHLY_LIMIT,
                "file_size_limit": cls.BASIC_TIER_FILE_SIZE_LIMIT
            },
            "premium": {
                "daily_limit": cls.PREMIUM_TIER_DAILY_LIMIT,
                "monthly_limit": cls.PREMIUM_TIER_MONTHLY_LIMIT,
                "file_size_limit": cls.PREMIUM_TIER_FILE_SIZE_LIMIT
            },
            "enterprise": {
                "daily_limit": cls.ENTERPRISE_TIER_DAILY_LIMIT,
                "monthly_limit": cls.ENTERPRISE_TIER_MONTHLY_LIMIT,
                "file_size_limit": cls.ENTERPRISE_TIER_FILE_SIZE_LIMIT
            }
        }
        return limits.get(tier.lower(), limits["free"])
    
    @classmethod
    def is_development(cls) -> bool:
        """Check if running in development mode"""
        return cls.ENVIRONMENT == "development" or cls.DEBUG
    
    @classmethod
    def is_production(cls) -> bool:
        """Check if running in production mode"""
        return cls.ENVIRONMENT == "production" and not cls.DEBUG

# Global settings instance
settings = Settings()

# Validate configuration on import
config_issues = settings.validate_config()
if config_issues:
    print("⚠️  Configuration Issues Found:")
    for issue in config_issues:
        print(f"   - {issue}")
    print("\nPlease review your environment configuration.")
