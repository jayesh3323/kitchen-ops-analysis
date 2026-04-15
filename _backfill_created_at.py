import os
import datetime
os.environ["USE_FIREBASE"] = "true"
import firebase_db
db = firebase_db._get_db()
for doc in db.collection("jobs").stream():
    data = doc.to_dict()
    if "created_at" not in data:
        print(f"Backfilling {doc.id}")
        doc.reference.update({
            "created_at": firebase_db._get_idst()
        })
print("Done")
