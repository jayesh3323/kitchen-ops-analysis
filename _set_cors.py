import os
import json
from dotenv import load_dotenv
import firebase_admin
from firebase_admin import credentials, storage

load_dotenv()

# The user's bucket name from app.js
BUCKET_NAME = "kitchen-analysis-dashboard.firebasestorage.app"

# Initialize Firebase using the JSON from .env
sa_json = os.environ.get("FIREBASE_SERVICE_ACCOUNT_JSON")
if not sa_json:
    print("No FIREBASE_SERVICE_ACCOUNT_JSON found!")
    exit(1)

sa_dict = json.loads(sa_json)
if "private_key" in sa_dict:
    sa_dict["private_key"] = sa_dict["private_key"].replace("\\n", "\n")

cred = credentials.Certificate(sa_dict)
firebase_admin.initialize_app(cred, {
    'storageBucket': BUCKET_NAME
})

# Get the bucket
bucket = storage.bucket()

# Define CORS rules
cors_configuration = [
    {
        "origin": ["*"],
        "method": ["GET", "PUT", "POST", "DELETE", "OPTIONS", "HEAD"],
        "responseHeader": ["Content-Type", "Authorization", "Content-Length", "User-Agent", "x-goog-resumable"],
        "maxAgeSeconds": 3600
    }
]

# Set CORS
bucket.cors = cors_configuration
bucket.patch()

print(f"Successfully updated CORS rules for bucket: {BUCKET_NAME}")
