import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Dict, Any, Optional
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmailService:
    def __init__(self):
        self.smtp_server = os.getenv("SMTP_SERVER", "smtp.gmail.com")
        self.smtp_port = int(os.getenv("SMTP_PORT", "587"))
        self.smtp_username = os.getenv("SMTP_USERNAME")
        self.smtp_password = os.getenv("SMTP_PASSWORD")
        self.from_email = os.getenv("FROM_EMAIL", self.smtp_username)
        self.from_name = os.getenv("FROM_NAME", "Resume Parser API")
        self.admin_email = os.getenv("ADMIN_EMAIL")
        
        # Check if email configuration is available
        self.email_enabled = bool(self.smtp_username and self.smtp_password)

    def create_connection(self) -> Optional[smtplib.SMTP]:
        """Create SMTP connection"""
        if not self.email_enabled:
            logger.warning("Email service not configured - SMTP credentials missing")
            return None
        
        try:
            server = smtplib.SMTP(self.smtp_server, self.smtp_port)
            server.starttls()
            server.login(self.smtp_username, self.smtp_password)
            return server
        except Exception as e:
            logger.error(f"Failed to create SMTP connection: {e}")
            return None

    def send_email(self, to_email: str, subject: str, html_content: str, text_content: str = None) -> bool:
        """Send email to recipient"""
        if not self.email_enabled:
            logger.warning("Email service not enabled - skipping email send")
            return False
        
        try:
            server = self.create_connection()
            if not server:
                return False
            
            # Create message
            msg = MIMEMultipart('alternative')
            msg['From'] = f"{self.from_name} <{self.from_email}>"
            msg['To'] = to_email
            msg['Subject'] = subject
            
            # Add text and HTML parts
            if text_content:
                text_part = MIMEText(text_content, 'plain')
                msg.attach(text_part)
            
            html_part = MIMEText(html_content, 'html')
            msg.attach(html_part)
            
            # Send email
            server.send_message(msg)
            server.quit()
            
            logger.info(f"Email sent successfully to {to_email}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send email to {to_email}: {e}")
            return False

    def send_user_credentials(self, user_data: Dict[str, Any]) -> bool:
        """Send user credentials via email"""
        username = user_data["username"]
        email = user_data["email"]
        password = user_data["password"]
        subscription_tier = user_data.get("subscription_tier", "FREE")
        api_calls_limit = user_data.get("api_calls_limit", 100)
        company_name = user_data.get("company_name", "")
        contact_person = user_data.get("contact_person", "")
        
        # Email subject
        subject = f"Your Resume Parser API Account - {company_name or username}"
        
        # HTML content
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <title>Resume Parser API Account</title>
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
                .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
                .header {{ background-color: #2c3e50; color: white; padding: 20px; text-align: center; }}
                .content {{ padding: 20px; background-color: #f9f9f9; }}
                .credentials {{ background-color: #e8f4fd; padding: 15px; border-left: 4px solid #3498db; margin: 20px 0; }}
                .warning {{ background-color: #fff3cd; padding: 15px; border-left: 4px solid #ffc107; margin: 20px 0; }}
                .footer {{ text-align: center; padding: 20px; color: #666; font-size: 12px; }}
                .button {{ display: inline-block; background-color: #3498db; color: white; padding: 10px 20px; text-decoration: none; border-radius: 5px; }}
                .api-endpoint {{ background-color: #f8f9fa; padding: 10px; border-radius: 5px; font-family: monospace; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>Resume Parser API</h1>
                    <p>Your account has been created successfully!</p>
                </div>
                
                <div class="content">
                    <h2>Welcome to Resume Parser API!</h2>
                    
                    <p>Hello {contact_person or username},</p>
                    
                    <p>Your account has been created and is ready to use. Below are your login credentials:</p>
                    
                    <div class="credentials">
                        <h3>🔐 Login Credentials</h3>
                        <p><strong>Username:</strong> {username}</p>
                        <p><strong>Password:</strong> {password}</p>
                        <p><strong>Subscription Tier:</strong> {subscription_tier.upper()}</p>
                        <p><strong>Monthly API Limit:</strong> {api_calls_limit} calls</p>
                    </div>
                    
                    <div class="warning">
                        <h3>⚠️ Important Security Notice</h3>
                        <p>Please change your password after your first login for security reasons.</p>
                    </div>
                    
                    <h3>🚀 Getting Started</h3>
                    <p>To start using the API, you need to:</p>
                    <ol>
                        <li>Login using the credentials above</li>
                        <li>Get your JWT access token</li>
                        <li>Use the token in your API requests</li>
                    </ol>
                    
                    <h3>📡 API Endpoints</h3>
                    <div class="api-endpoint">
                        <p><strong>Base URL:</strong> {os.getenv("API_BASE_URL", "https://your-api-domain.com")}</p>
                        <p><strong>Login:</strong> POST /auth/login</p>
                        <p><strong>Parse Resume:</strong> POST /parse-resume</p>
                        <p><strong>Parse Text:</strong> POST /parse-resume-text</p>
                    </div>
                    
                    <h3>📖 Example Usage</h3>
                    <p>1. Login to get your token:</p>
                    <div class="api-endpoint">
                        curl -X POST "{os.getenv("API_BASE_URL", "https://your-api-domain.com")}/auth/login" \<br>
                        &nbsp;&nbsp;-H "Content-Type: application/json" \<br>
                        &nbsp;&nbsp;-d '{{"username": "{username}", "password": "{password}"}}'
                    </div>
                    
                    <p>2. Use the token to parse resumes:</p>
                    <div class="api-endpoint">
                        curl -X POST "{os.getenv("API_BASE_URL", "https://your-api-domain.com")}/parse-resume" \<br>
                        &nbsp;&nbsp;-H "Authorization: Bearer YOUR_JWT_TOKEN" \<br>
                        &nbsp;&nbsp;-F "file=@resume.pdf"
                    </div>
                    
                    <h3>📊 Usage Tracking</h3>
                    <p>Your API usage is tracked and you can monitor it through your account dashboard.</p>
                    
                    <h3>🆘 Support</h3>
                    <p>If you have any questions or need assistance, please contact our support team.</p>
                    
                    <p>Best regards,<br>Resume Parser API Team</p>
                </div>
                
                <div class="footer">
                    <p>This is an automated message. Please do not reply to this email.</p>
                    <p>© {datetime.now().year} Resume Parser API. All rights reserved.</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        # Plain text content
        text_content = f"""
        Resume Parser API - Account Created
        
        Hello {contact_person or username},
        
        Your Resume Parser API account has been created successfully!
        
        LOGIN CREDENTIALS:
        Username: {username}
        Password: {password}
        Subscription Tier: {subscription_tier.upper()}
        Monthly API Limit: {api_calls_limit} calls
        
        IMPORTANT: Please change your password after your first login for security reasons.
        
        GETTING STARTED:
        1. Login using the credentials above
        2. Get your JWT access token
        3. Use the token in your API requests
        
        API ENDPOINTS:
        Base URL: {os.getenv("API_BASE_URL", "https://your-api-domain.com")}
        Login: POST /auth/login
        Parse Resume: POST /parse-resume
        Parse Text: POST /parse-resume-text
        
        EXAMPLE USAGE:
        1. Login to get your token:
        curl -X POST "{os.getenv("API_BASE_URL", "https://your-api-domain.com")}/auth/login" \\
             -H "Content-Type: application/json" \\
             -d '{{"username": "{username}", "password": "{password}"}}'
        
        2. Use the token to parse resumes:
        curl -X POST "{os.getenv("API_BASE_URL", "https://your-api-domain.com")}/parse-resume" \\
             -H "Authorization: Bearer YOUR_JWT_TOKEN" \\
             -F "file=@resume.pdf"
        
        USAGE TRACKING:
        Your API usage is tracked and you can monitor it through your account dashboard.
        
        SUPPORT:
        If you have any questions or need assistance, please contact our support team.
        
        Best regards,
        Resume Parser API Team
        
        ---
        This is an automated message. Please do not reply to this email.
        © {datetime.now().year} Resume Parser API. All rights reserved.
        """
        
        return self.send_email(email, subject, html_content, text_content)

    def send_usage_alert(self, user_data: Dict[str, Any], usage_percentage: float) -> bool:
        """Send usage alert email"""
        username = user_data["username"]
        email = user_data["email"]
        api_calls_used = user_data.get("api_calls_used", 0)
        api_calls_limit = user_data.get("api_calls_limit", 100)
        
        subject = f"Resume Parser API - Usage Alert ({usage_percentage:.0f}% used)"
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <title>Usage Alert</title>
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
                .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
                .header {{ background-color: #e74c3c; color: white; padding: 20px; text-align: center; }}
                .content {{ padding: 20px; background-color: #f9f9f9; }}
                .alert {{ background-color: #fff3cd; padding: 15px; border-left: 4px solid #ffc107; margin: 20px 0; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>⚠️ Usage Alert</h1>
                </div>
                
                <div class="content">
                    <h2>API Usage Alert</h2>
                    
                    <p>Hello {username},</p>
                    
                    <div class="alert">
                        <h3>Your API usage is at {usage_percentage:.0f}%</h3>
                        <p><strong>Used:</strong> {api_calls_used} calls</p>
                        <p><strong>Limit:</strong> {api_calls_limit} calls</p>
                        <p><strong>Remaining:</strong> {api_calls_limit - api_calls_used} calls</p>
                    </div>
                    
                    <p>You're approaching your monthly API limit. Consider upgrading your subscription if you need more calls.</p>
                    
                    <p>Best regards,<br>Resume Parser API Team</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        return self.send_email(email, subject, html_content)

    def send_admin_notification(self, subject: str, message: str) -> bool:
        """Send notification to admin"""
        if not self.admin_email:
            logger.warning("Admin email not configured")
            return False
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <title>Admin Notification</title>
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
                .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
                .header {{ background-color: #2c3e50; color: white; padding: 20px; text-align: center; }}
                .content {{ padding: 20px; background-color: #f9f9f9; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>Admin Notification</h1>
                </div>
                
                <div class="content">
                    <h2>{subject}</h2>
                    <p>{message}</p>
                    <p>Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        return self.send_email(self.admin_email, f"Admin Alert: {subject}", html_content)

# Global email service instance
email_service = EmailService()

