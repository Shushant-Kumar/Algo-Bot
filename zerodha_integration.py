from kiteconnect import KiteConnect
import os
from dotenv import load_dotenv
import json
from datetime import datetime

def save_to_env(key, value):
    """Save a key-value pair to .env file"""
    env_path = '.env'
    
    # Read existing content
    content = {}
    if os.path.exists(env_path):
        with open(env_path, 'r') as f:
            for line in f:
                if '=' in line:
                    k, v = line.strip().split('=', 1)
                    content[k] = v
    
    # Update or add the new value
    content[key] = value
    
    # Write back to file
    with open(env_path, 'w') as f:
        for k, v in content.items():
            f.write(f"{k}={v}\n")
    
    print(f"Updated {key} in .env file")

def save_token_to_cache(access_token):
    """Save token to cache directory"""
    cache_dir = os.path.join(os.path.dirname(__file__), 'cache')
    os.makedirs(cache_dir, exist_ok=True)
    
    token_data = {
        "access_token": access_token,
        "timestamp": datetime.now().isoformat()
    }
    
    with open(os.path.join(cache_dir, 'token_cache.json'), 'w') as f:
        json.dump(token_data, f)
    
    print(f"Token saved to cache file")

def main():
    # Load existing environment variables
    load_dotenv()
    
    # Use environment variables if available, otherwise prompt
    api_key = os.getenv("KITE_API_KEY") or input("Enter your Zerodha API key: ")
    api_secret = os.getenv("KITE_API_SECRET") or input("Enter your Zerodha API secret: ")
    
    # Save to .env file if entered manually
    if not os.getenv("KITE_API_KEY"):
        save_to_env("KITE_API_KEY", api_key)
    if not os.getenv("KITE_API_SECRET"):
        save_to_env("KITE_API_SECRET", api_secret)
    
    # Initialize KiteConnect
    kite = KiteConnect(api_key=api_key)
    
    # Generate login URL
    print("\n==== Zerodha Authentication ====")
    print(f"Login URL: {kite.login_url()}")
    print("Visit the above URL, authorize the app, and paste the request token below.")
    
    # Prompt user for request token
    request_token = input("\nEnter the request token: ")
    
    # Exchange request token for access token
    try:
        data = kite.generate_session(request_token, api_secret=api_secret)
        access_token = data["access_token"]
        
        # Save token to environment and cache
        save_to_env("ZERODHA_ACCESS_TOKEN", access_token)
        save_token_to_cache(access_token)
        
        print("\n==== Success! ====")
        print(f"Access Token: {access_token}")
        print("Token saved to .env file and cache directory")
        print("You can now run your algo-bot!")
        
    except Exception as e:
        print(f"\n❌ Error generating access token: {e}")
        print("Please verify your API credentials and request token.")

if __name__ == "__main__":
    main()
