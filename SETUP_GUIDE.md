# Resume Parser API - Setup Guide

## Quick Setup for IONOS SMTP

### 1. Create `.env` File

Create a `.env` file in your project root with the following configuration:

```env
# ===========================================
# API Configuration
# ===========================================
HOST=0.0.0.0
PORT=2010
API_BASE_URL=https://your-api-domain.com
FRONTEND_URL=http://localhost:3000

# ===========================================
# Database Configuration
# ===========================================
MONGODB_URL=mongodb://localhost:27017
MONGODB_DATABASE=resume_parser

# ===========================================
# JWT Authentication Configuration
# ===========================================
JWT_SECRET_KEY=your-super-secret-jwt-key-change-this-in-production
JWT_ALGORITHM=HS256
JWT_ACCESS_TOKEN_EXPIRE_MINUTES=1440

# ===========================================
# OpenAI Configuration
# ===========================================
OPENAI_API_KEY=your_openai_api_key_here

# ===========================================
# IONOS SMTP Configuration
# ===========================================
SMTP_SERVER=smtp.ionos.com
SMTP_PORT=587
SMTP_USERNAME=your-email@yourdomain.com
SMTP_PASSWORD=your-email-password
FROM_EMAIL=your-email@yourdomain.com
FROM_NAME=Resume Parser API
ADMIN_EMAIL=admin@yourdomain.com

# ===========================================
# Admin Configuration
# ===========================================
ADMIN_USERNAME=admin
ADMIN_EMAIL=admin@yourdomain.com
ADMIN_PASSWORD=change-this-admin-password

# ===========================================
# Development Settings
# ===========================================
DEBUG=false
ENVIRONMENT=development
LOG_LEVEL=INFO
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Start MongoDB

```bash
# Local MongoDB
mongod

# Or use MongoDB Atlas (cloud)
# Update MONGODB_URL in .env file
```

### 4. Run the Application

```bash
python main.py
```

The API will be available at `http://localhost:2010`

## IONOS SMTP Configuration Details

### SMTP Settings for IONOS:

- **Server**: `smtp.ionos.com`
- **Port**: `587` (TLS/STARTTLS)
- **Security**: TLS/STARTTLS
- **Authentication**: Required

### Email Configuration:

```env
SMTP_SERVER=smtp.ionos.com
SMTP_PORT=587
SMTP_USERNAME=your-email@yourdomain.com
SMTP_PASSWORD=your-email-password
FROM_EMAIL=your-email@yourdomain.com
FROM_NAME=Resume Parser API
ADMIN_EMAIL=admin@yourdomain.com
```

## MongoDB Configuration

### Local MongoDB:

```env
MONGODB_URL=mongodb://localhost:27017
MONGODB_DATABASE=resume_parser
```

### MongoDB Atlas (Cloud):

```env
MONGODB_URL=mongodb+srv://username:password@cluster.mongodb.net/resume_parser?retryWrites=true&w=majority
MONGODB_DATABASE=resume_parser
```

## Testing the Setup

### 1. Check API Health

```bash
curl http://localhost:2010/health
```

### 2. Login as Admin

```bash
curl -X POST "http://localhost:2010/auth/login" \
  -H "Content-Type: application/json" \
  -d '{"username": "admin", "password": "change-this-admin-password"}'
```

### 3. Create a User Account

```bash
curl -X POST "http://localhost:2010/auth/admin/create-user" \
  -H "Authorization: Bearer YOUR_JWT_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "username": "test_user",
    "email": "test@example.com",
    "subscription_tier": "basic",
    "api_calls_limit": 1000,
    "company_name": "Test Company",
    "contact_person": "Test User"
  }'
```

## Troubleshooting

### Common Issues:

1. **MongoDB Connection Failed**

   - Ensure MongoDB is running
   - Check MONGODB_URL is correct
   - Verify network connectivity

2. **Email Not Sending**

   - Check IONOS SMTP credentials
   - Verify email service configuration
   - Check firewall settings

3. **Authentication Errors**

   - Verify JWT_SECRET_KEY is set
   - Check token expiration
   - Ensure user account is active

4. **Configuration Warnings**
   - Update JWT_SECRET_KEY from default
   - Configure SMTP credentials
   - Set proper environment variables

## Production Deployment

### Security Checklist:

1. ✅ Change default JWT_SECRET_KEY
2. ✅ Set strong admin password
3. ✅ Configure IONOS SMTP credentials
4. ✅ Set up MongoDB with authentication
5. ✅ Enable SSL/TLS in production
6. ✅ Configure proper CORS origins
7. ✅ Set up monitoring and logging

### Environment Variables for Production:

```env
ENVIRONMENT=production
DEBUG=false
SSL_ENABLED=true
JWT_SECRET_KEY=your-very-secure-random-key-here
```

## API Documentation

Once running, visit:

- **Swagger UI**: `http://localhost:2010/docs`
- **ReDoc**: `http://localhost:2010/redoc`

## Support

For technical support:

- Check the API documentation at `/docs`
- Review the logs for error details
- Contact the system administrator
- Check the GitHub repository for updates


