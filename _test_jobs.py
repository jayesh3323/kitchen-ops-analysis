import os
os.environ["USE_FIREBASE"] = "true"
import json
import logging
from config import USE_FIREBASE
from database import SessionLocal, Job, init_db

print(f"USE_FIREBASE={USE_FIREBASE}")
init_db()
s = SessionLocal()

import firebase_db
print(f"is it using firebase session? {isinstance(s, firebase_db.FirestoreSession)}")

# Create a job
job = Job(video_name="dummy.mp4", video_path="/app/uploads/dummy.mp4")
s.add(job)
s.commit()
print("Saved job id:", job.id)

jobs = s.query(Job).order_by(Job.created_at.desc()).all()
output = []
for j in jobs:
    d = j.to_dict()
    output.append(d)

print(json.dumps(output, indent=2, default=str))
