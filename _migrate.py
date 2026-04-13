"""One-time DB migration: add timestamp_region_coords, remove langfuse columns."""
import sqlite3

conn = sqlite3.connect('jobs.db')
cur = conn.cursor()

cur.execute('PRAGMA table_info(jobs)')
existing = [row[1] for row in cur.fetchall()]
print('Existing columns:', existing)

if 'timestamp_region_coords' not in existing:
    cur.execute('ALTER TABLE jobs ADD COLUMN timestamp_region_coords TEXT')
    print('Added timestamp_region_coords')

if 'agent' not in existing:
    cur.execute('ALTER TABLE jobs ADD COLUMN agent VARCHAR(50)')
    print('Added agent')

# SQLite cannot DROP columns directly (< 3.35), so we just leave langfuse cols
# as unused — they won't cause errors.

conn.commit()
conn.close()
print('Migration complete.')
