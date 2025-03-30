from kiteconnect import KiteConnect

def main():
    # Prompt user for API key and secret
    api_key = input("Enter your Zerodha API key: ")
    api_secret = input("Enter your Zerodha API secret: ")
    
    # Initialize KiteConnect
    kite = KiteConnect(api_key=api_key)
    
    # Generate login URL
    print("Login URL:", kite.login_url())
    print("Visit the above URL, authorize the app, and paste the request token below.")
    
    # Prompt user for request token
    request_token = input("Enter the request token: ")
    
    # Exchange request token for access token
    try:
        data = kite.generate_session(request_token, api_secret=api_secret)
        access_token = data["access_token"]
        print("Access Token:", access_token)
        print("Save this access token securely for live trading.")
    except Exception as e:
        print("Error generating access token:", e)

if __name__ == "__main__":
    main()
