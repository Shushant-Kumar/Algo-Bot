import os
from datetime import datetime

def log(message):
    os.makedirs('logs', exist_ok=True)  # Ensure logs directory exists
    log_file = f"logs/log_{datetime.now().strftime('%Y-%m-%d')}.txt"  # Log file with current date
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')  # Add timestamp
    with open(log_file, 'a') as f:
        f.write(f"[{timestamp}] {message}\n")
    print(f"[{timestamp}] {message}")
