import requests

def check_service(url, name):
    try:
        response = requests.get(url, timeout=5)
        print(f"{name}: {response.status_code}")
        try:
            print(f"  Response: {response.json()}")
        except:
            print(f"  Response: {response.text[:100]}...")
    except Exception as e:
        print(f"{name}: Error - {str(e)}")

print("Self Brain AGI System Status Check")
print("==================================")

# Check main web interface
check_service("http://localhost:5000", "Main Web Interface")

# Check A Management Model
check_service("http://localhost:5001", "A Management Model")

# Check basic API endpoints if available
check_service("http://localhost:5000/api", "Main API")