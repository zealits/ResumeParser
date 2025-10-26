# Resume Parser API - Authentication System

## Overview

This document describes the complete authentication system implemented for the Resume Parser API. The system provides JWT-based authentication with admin-created user accounts, usage tracking, and analytics.

## Features

- **JWT Token Authentication**: Secure token-based authentication
- **Admin-Created Users**: Only admins can create user accounts
- **Usage Tracking**: Monitor API calls, processing time, and file sizes
- **Rate Limiting**: Prevent abuse with daily and monthly limits
- **Analytics Dashboard**: Comprehensive usage analytics for admins
- **Email Notifications**: Automatic credential delivery and usage alerts
- **Subscription Tiers**: Different limits for different user types
- **MongoDB Storage**: Scalable document-based storage

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Client App    │    │   FastAPI App   │    │   MongoDB       │
│                 │    │                 │    │                 │
│ 1. Login        │───▶│ 2. Authenticate │───▶│ 3. Store Token  │
│ 4. Use API      │───▶│ 5. Verify Token │───▶│ 6. Track Usage  │
│ 7. Get Results  │◀───│ 8. Process      │◀───│ 9. Log Analytics│
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Setup Instructions

### 1. Environment Configuration

Create a `.env` file with the following variables:

```env
# API Configuration
HOST=0.0.0.0
PORT=2010
API_BASE_URL=https://your-api-domain.com

# Database
MONGODB_URL=mongodb://localhost:27017
MONGODB_DATABASE=resume_parser

# JWT Authentication
JWT_SECRET_KEY=your-super-secret-jwt-key-change-this-in-production
JWT_ALGORITHM=HS256
JWT_ACCESS_TOKEN_EXPIRE_MINUTES=1440

# OpenAI
OPENAI_API_KEY=your_openai_api_key_here

# Email Service
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
SMTP_USERNAME=your-email@gmail.com
SMTP_PASSWORD=your-app-password
FROM_EMAIL=your-email@gmail.com
FROM_NAME=Resume Parser API
ADMIN_EMAIL=admin@yourcompany.com

# Admin Account
ADMIN_USERNAME=admin
ADMIN_EMAIL=admin@yourcompany.com
ADMIN_PASSWORD=change-this-admin-password
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

## API Endpoints

### Authentication Endpoints

#### 1. Admin Login

```bash
POST /auth/login
Content-Type: application/json

{
  "username": "admin",
  "password": "change-this-admin-password"
}
```

**Response:**

```json
{
  "access_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
  "token_type": "bearer",
  "expires_in": 86400,
  "user": {
    "id": "user_id",
    "username": "admin",
    "email": "admin@yourcompany.com",
    "subscription_tier": "enterprise",
    "api_calls_limit": -1,
    "api_calls_used": 0
  }
}
```

#### 2. Create User Account (Admin Only)

```bash
POST /auth/admin/create-user
Authorization: Bearer your-jwt-token
Content-Type: application/json

{
  "username": "client_company",
  "email": "client@company.com",
  "password": "optional-password",
  "subscription_tier": "basic",
  "api_calls_limit": 1000,
  "company_name": "Client Company",
  "contact_person": "John Doe"
}
```

**Response:**

```json
{
  "success": true,
  "message": "User account created successfully. Credentials sent via email.",
  "user_id": "user_id",
  "username": "client_company",
  "email": "client@company.com",
  "subscription_tier": "basic",
  "api_calls_limit": 1000
}
```

#### 3. User Login

```bash
POST /auth/login
Content-Type: application/json

{
  "username": "client_company",
  "password": "generated-password"
}
```

### Protected API Endpoints

#### 1. Parse Resume PDF

```bash
POST /parse-resume
Authorization: Bearer your-jwt-token
Content-Type: multipart/form-data

file: resume.pdf
```

#### 2. Parse Resume Text

```bash
POST /parse-resume-text
Authorization: Bearer your-jwt-token
Content-Type: application/json

{
  "resume_text": "John Doe\nSoftware Engineer..."
}
```

### User Management Endpoints

#### 1. Get User Profile

```bash
GET /auth/profile
Authorization: Bearer your-jwt-token
```

#### 2. Get Usage Statistics

```bash
GET /auth/usage-stats?days=30
Authorization: Bearer your-jwt-token
```

#### 3. Check Rate Limits

```bash
GET /auth/rate-limit
Authorization: Bearer your-jwt-token
```

### Admin Endpoints

#### 1. Get All Users

```bash
GET /auth/admin/users?skip=0&limit=100
Authorization: Bearer admin-jwt-token
```

#### 2. Get Analytics

```bash
GET /auth/admin/analytics?days=30
Authorization: Bearer admin-jwt-token
```

#### 3. Suspend User

```bash
PUT /auth/admin/users/{user_id}/suspend
Authorization: Bearer admin-jwt-token
```

#### 4. Upgrade User Subscription

```bash
PUT /auth/admin/users/{user_id}/upgrade
Authorization: Bearer admin-jwt-token
Content-Type: application/json

{
  "subscription_tier": "premium",
  "api_calls_limit": 5000
}
```

## Subscription Tiers

| Tier       | Daily Limit | Monthly Limit | File Size Limit |
| ---------- | ----------- | ------------- | --------------- |
| Free       | 10 calls    | 100 calls     | 5MB             |
| Basic      | 50 calls    | 1,000 calls   | 10MB            |
| Premium    | 200 calls   | 5,000 calls   | 25MB            |
| Enterprise | Unlimited   | Unlimited     | 50MB            |

## Usage Tracking

The system automatically tracks:

- **API Calls**: Number of requests per user
- **Processing Time**: Time taken to process each request
- **File Sizes**: Size of uploaded files
- **Success/Failure Rates**: Success and error rates
- **Endpoint Usage**: Which endpoints are used most
- **Daily/Monthly Statistics**: Aggregated usage data

## Rate Limiting

- **Monthly Limits**: Based on subscription tier
- **Daily Limits**: Prevents abuse
- **File Size Limits**: Prevents large file uploads
- **Automatic Alerts**: Email notifications at 80% usage

## Email Notifications

### 1. User Credentials

When a new user is created, they receive an email with:

- Username and password
- API endpoints and usage examples
- Subscription details
- Security recommendations

### 2. Usage Alerts

Users receive email alerts when:

- They reach 80% of their monthly limit
- They exceed their daily limit
- Their account is suspended

### 3. Admin Notifications

Admins receive notifications for:

- New user registrations
- System errors
- Usage anomalies

## Security Features

- **JWT Tokens**: Secure, stateless authentication
- **Password Hashing**: bcrypt encryption
- **Rate Limiting**: Prevents abuse
- **File Size Limits**: Prevents DoS attacks
- **CORS Protection**: Configurable origins
- **Security Headers**: XSS, CSRF protection
- **Input Validation**: Pydantic models
- **Error Handling**: Secure error messages

## Database Schema

### Users Collection

```json
{
  "_id": "ObjectId",
  "username": "string",
  "email": "string",
  "hashed_password": "string",
  "subscription_tier": "string",
  "api_calls_limit": "number",
  "api_calls_used": "number",
  "company_name": "string",
  "contact_person": "string",
  "status": "string",
  "role": "string",
  "created_at": "datetime",
  "last_login": "datetime",
  "is_active": "boolean"
}
```

### Usage Stats Collection

```json
{
  "_id": "ObjectId",
  "user_id": "string",
  "date": "date",
  "api_calls": "number",
  "total_processing_time": "number",
  "successful_calls": "number",
  "failed_calls": "number",
  "file_sizes": ["number"],
  "endpoints_used": { "endpoint": "count" },
  "created_at": "datetime"
}
```

### API Calls Collection

```json
{
  "_id": "ObjectId",
  "user_id": "string",
  "endpoint": "string",
  "method": "string",
  "status_code": "number",
  "response_time": "number",
  "file_size": "number",
  "ip_address": "string",
  "user_agent": "string",
  "timestamp": "datetime",
  "error_message": "string"
}
```

## Monitoring & Analytics

### Admin Dashboard Metrics

- Total users and active users
- Total API calls and processing time
- Usage by subscription tier
- Daily usage trends
- Top users by API calls
- Success/failure rates

### User Dashboard Metrics

- Personal usage statistics
- Remaining API calls
- Daily breakdown
- Success rate
- Average response time

## Error Handling

### Common Error Responses

#### 401 Unauthorized

```json
{
  "detail": "Could not validate credentials"
}
```

#### 403 Forbidden

```json
{
  "detail": "Account is inactive"
}
```

#### 429 Too Many Requests

```json
{
  "error": "Rate limit exceeded",
  "message": "Monthly API limit exceeded",
  "limit": 1000,
  "used": 1000
}
```

#### 413 Request Entity Too Large

```json
{
  "error": "File too large",
  "message": "File size exceeds 10MB limit",
  "max_size": 10485760,
  "received_size": 15728640
}
```

## Deployment Considerations

### Production Environment

1. **Change Default Passwords**: Update admin credentials
2. **Secure JWT Secret**: Use a strong, random secret key
3. **Enable SSL**: Use HTTPS in production
4. **Database Security**: Secure MongoDB access
5. **Email Configuration**: Set up proper SMTP
6. **Monitoring**: Set up logging and monitoring
7. **Backup**: Regular database backups

### Environment Variables

- Set all required environment variables
- Use secure values for production
- Enable SSL/TLS
- Configure proper CORS origins
- Set up email service

## Troubleshooting

### Common Issues

1. **Database Connection Failed**

   - Check MongoDB is running
   - Verify MONGODB_URL is correct
   - Check network connectivity

2. **Email Not Sending**

   - Check SMTP credentials
   - Verify email service configuration
   - Check firewall settings

3. **Authentication Errors**

   - Verify JWT_SECRET_KEY is set
   - Check token expiration
   - Ensure user account is active

4. **Rate Limit Issues**
   - Check subscription tier limits
   - Verify usage tracking
   - Review rate limit configuration

### Logs

- Check application logs for errors
- Monitor database connection logs
- Review email service logs
- Check authentication logs

## Support

For technical support or questions:

- Check the API documentation at `/docs`
- Review the logs for error details
- Contact the system administrator
- Check the GitHub repository for updates

## Security Best Practices

1. **Regular Updates**: Keep dependencies updated
2. **Password Policy**: Enforce strong passwords
3. **Token Rotation**: Implement token refresh
4. **Audit Logs**: Monitor user activities
5. **Backup Strategy**: Regular data backups
6. **Access Control**: Limit admin access
7. **Monitoring**: Set up alerts and monitoring
8. **Documentation**: Keep security docs updated


