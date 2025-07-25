from kiteconnect import KiteConnect
import os
from dotenv import load_dotenv
import json
import logging
import hashlib
import time
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from cryptography.fernet import Fernet
import base64

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SecureZerodhaAuth:
    """Production-grade Zerodha authentication with enhanced security"""
    
    def __init__(self):
        self.cache_dir = os.path.join(os.path.dirname(__file__), 'cache')
        self.env_path = '.env'
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Generate or load encryption key
        self.encryption_key = self._get_or_create_encryption_key()
        self.cipher_suite = Fernet(self.encryption_key)
    
    def _get_or_create_encryption_key(self) -> bytes:
        """Generate or retrieve encryption key for secure token storage"""
        key_file = os.path.join(self.cache_dir, '.key')
        
        if os.path.exists(key_file):
            with open(key_file, 'rb') as f:
                return f.read()
        else:
            key = Fernet.generate_key()
            with open(key_file, 'wb') as f:
                f.write(key)
            # Set restrictive permissions on Windows
            os.chmod(key_file, 0o600)
            return key
    
    def save_to_env(self, key: str, value: str) -> None:
        """Save a key-value pair to .env file with validation"""
        try:
            # Read existing content
            content = {}
            if os.path.exists(self.env_path):
                with open(self.env_path, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if '=' in line and not line.startswith('#'):
                            k, v = line.split('=', 1)
                            content[k.strip()] = v.strip()
            
            # Update or add the new value
            content[key] = value
            
            # Write back to file with proper formatting
            with open(self.env_path, 'w') as f:
                f.write("# Zerodha API Configuration\n")
                f.write("# Generated by SecureZerodhaAuth\n\n")
                for k, v in content.items():
                    f.write(f"{k}={v}\n")
            
            # Set restrictive permissions
            os.chmod(self.env_path, 0o600)
            logger.info(f"Updated {key} in .env file securely")
            
        except Exception as e:
            logger.error(f"Error saving to .env file: {e}")
            raise

    
    def save_encrypted_token(self, access_token: str, user_id: Optional[str] = None) -> None:
        """Save encrypted access token to cache with metadata"""
        try:
            token_data = {
                "access_token": access_token,
                "user_id": user_id,
                "timestamp": datetime.now().isoformat(),
                "expires_at": (datetime.now() + timedelta(hours=6)).isoformat(),  # Zerodha tokens expire daily
                "checksum": hashlib.sha256(access_token.encode()).hexdigest()[:16]
            }
            
            # Encrypt the sensitive data
            encrypted_data = self.cipher_suite.encrypt(json.dumps(token_data).encode())
            
            # Save to cache file
            cache_file = os.path.join(self.cache_dir, 'secure_token_cache.dat')
            with open(cache_file, 'wb') as f:
                f.write(encrypted_data)
            
            # Set restrictive permissions
            os.chmod(cache_file, 0o600)
            logger.info("Token saved securely to encrypted cache")
            
        except Exception as e:
            logger.error(f"Error saving encrypted token: {e}")
            raise
    
    def load_encrypted_token(self) -> Optional[Dict[str, Any]]:
        """Load and decrypt access token from cache"""
        try:
            cache_file = os.path.join(self.cache_dir, 'secure_token_cache.dat')
            
            if not os.path.exists(cache_file):
                return None
            
            # Read and decrypt data
            with open(cache_file, 'rb') as f:
                encrypted_data = f.read()
            
            decrypted_data = self.cipher_suite.decrypt(encrypted_data)
            token_data = json.loads(decrypted_data.decode())
            
            # Check if token is still valid
            expires_at = datetime.fromisoformat(token_data.get('expires_at', ''))
            if datetime.now() >= expires_at:
                logger.warning("Cached token has expired")
                return None
            
            # Verify checksum
            expected_checksum = hashlib.sha256(token_data['access_token'].encode()).hexdigest()[:16]
            if token_data.get('checksum') != expected_checksum:
                logger.error("Token checksum mismatch - possible corruption")
                return None
            
            logger.info("Successfully loaded valid token from cache")
            return token_data
            
        except Exception as e:
            logger.error(f"Error loading encrypted token: {e}")
            return None
    
    def validate_credentials(self, api_key: str, api_secret: str) -> bool:
        """Validate API credentials format"""
        if not api_key or not api_secret:
            return False
        
        # Basic format validation for Zerodha credentials
        if len(api_key) < 10 or len(api_secret) < 10:
            return False
        
        return True
    
    def authenticate_with_retry(self, max_retries: int = 3) -> Optional[str]:
        """Authenticate with Zerodha with retry mechanism"""
        load_dotenv()
        
        # Try to load cached token first
        cached_token = self.load_encrypted_token()
        if cached_token:
            access_token = cached_token['access_token']
            logger.info("Using cached access token")
            return access_token
        
        # Get credentials
        api_key = os.getenv("KITE_API_KEY")
        api_secret = os.getenv("KITE_API_SECRET")
        
        # Prompt for credentials if not available
        if not api_key:
            api_key = input("Enter your Zerodha API key: ").strip()
            if api_key:
                self.save_to_env("KITE_API_KEY", api_key)
        
        if not api_secret:
            api_secret = input("Enter your Zerodha API secret: ").strip()
            if api_secret:
                self.save_to_env("KITE_API_SECRET", api_secret)
        
        # Validate credentials
        if not self.validate_credentials(api_key, api_secret):
            logger.error("Invalid API credentials provided")
            return None
        
        # Initialize KiteConnect
        kite = KiteConnect(api_key=api_key)
        
        for attempt in range(max_retries):
            try:
                # Generate login URL
                print(f"\n==== Zerodha Authentication (Attempt {attempt + 1}/{max_retries}) ====")
                print(f"Login URL: {kite.login_url()}")
                print("Visit the above URL, authorize the app, and paste the request token below.")
                
                # Prompt for request token
                request_token = input("\nEnter the request token: ").strip()
                
                if not request_token:
                    logger.error("Request token cannot be empty")
                    continue
                
                # Exchange request token for access token
                logger.info("Exchanging request token for access token...")
                data = kite.generate_session(request_token, api_secret=api_secret)
                
                if not isinstance(data, dict) or 'access_token' not in data:
                    logger.error("Invalid response from Zerodha API")
                    continue
                
                access_token = data["access_token"]
                user_id = data.get("user_id")
                
                # Save tokens securely
                self.save_to_env("ZERODHA_ACCESS_TOKEN", access_token)
                self.save_encrypted_token(access_token, user_id)
                
                logger.info("✅ Authentication successful!")
                print(f"\n==== Success! ====")
                print(f"Access Token: {access_token[:20]}...")
                print(f"User ID: {user_id}")
                print("Tokens saved securely to .env file and encrypted cache")
                print("You can now run your algo-bot!")
                
                return access_token
                
            except Exception as e:
                logger.error(f"Authentication attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    print(f"Retrying... ({max_retries - attempt - 1} attempts remaining)")
                    time.sleep(2)
                else:
                    print("\n❌ All authentication attempts failed")
                    print("Please verify your API credentials and request token.")
        
        return None

def get_authenticated_kite() -> Optional[KiteConnect]:
    """Get authenticated KiteConnect instance for use in trading system"""
    auth = SecureZerodhaAuth()
    access_token = auth.authenticate_with_retry()
    
    if not access_token:
        return None
    
    load_dotenv()
    api_key = os.getenv("KITE_API_KEY")
    
    if not api_key:
        logger.error("API key not found in environment")
        return None
    
    kite = KiteConnect(api_key=api_key)
    kite.set_access_token(access_token)
    
    # Test the connection
    try:
        profile = kite.profile()
        # Handle both dict and bytes response types
        if isinstance(profile, dict):
            user_name = profile.get('user_name', 'Unknown')
        else:
            # If response is bytes, convert to string
            user_name = str(profile) if profile else 'Unknown'
        
        logger.info(f"Successfully authenticated user: {user_name}")
        return kite
    except Exception as e:
        logger.error(f"Failed to verify authentication: {e}")
        return None

def main():
    """Main authentication flow"""
    auth = SecureZerodhaAuth()
    access_token = auth.authenticate_with_retry()
    
    if access_token:
        print("\n🚀 Authentication completed successfully!")
        print("Your algo-bot is ready to trade!")
    else:
        print("\n❌ Authentication failed!")
        print("Please check your credentials and try again.")

if __name__ == "__main__":
    main()
