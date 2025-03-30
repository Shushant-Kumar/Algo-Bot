from kiteconnect import KiteConnect
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Retrieve the API key from the environment
api_key = os.getenv("KITE_API_KEY")

kite = KiteConnect(api_key=api_key)