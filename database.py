
import sqlite3
import json
from datetime import datetime

DB_NAME = "road_hazards.db"

# Available report statuses
REPORT_STATUSES = [
    "Pending",
    "Verified",
    "In Progress",
    "Resolved",
    "Rejected"
]


def get_connection():
    return sqlite3.connect(DB_NAME)


def initialize_database():
    conn = get_connection()
    cursor = conn.cursor()

    # Create reports table (for new installations)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            report_date TEXT NOT NULL,
            image_name TEXT,
            hazard_data TEXT,
            total_hazards INTEGER,
            severity TEXT,
            latitude REAL,
            longitude REAL,
            location_name TEXT,
            status TEXT NOT NULL DEFAULT 'Pending',
            media_type TEXT NOT NULL DEFAULT 'image',
            media_path TEXT,
            processed_media_path TEXT,
            report_metadata TEXT
        )
    """)

    # Migration: Add status column to existing database
    cursor.execute("PRAGMA table_info(reports)")
    columns = [column[1] for column in cursor.fetchall()]

    if "status" not in columns:
        cursor.execute("""
            ALTER TABLE reports
            ADD COLUMN status TEXT NOT NULL DEFAULT 'Pending'
        """)

    # Backward-compatible migration for image/video evidence and analysis metadata.
    migrations = {
        "media_type": "TEXT NOT NULL DEFAULT 'image'",
        "media_path": "TEXT",
        "processed_media_path": "TEXT",
        "report_metadata": "TEXT",
    }
    for column_name, column_type in migrations.items():
        if column_name not in columns:
            cursor.execute(f"ALTER TABLE reports ADD COLUMN {column_name} {column_type}")

    conn.commit()
    conn.close()


# =========================
# SAVE NEW REPORT
# =========================

def save_report(
    image_name,
    detections,
    total_hazards,
    severity,
    latitude,
    longitude,
    location_name,
    media_type="image",
    media_path=None,
    processed_media_path=None,
    report_metadata=None,
):
    conn = get_connection()
    cursor = conn.cursor()

    report_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    hazard_json = json.dumps(detections)

    cursor.execute("""
        INSERT INTO reports (
            report_date,
            image_name,
            hazard_data,
            total_hazards,
            severity,
            latitude,
            longitude,
            location_name,
            status,
            media_type,
            media_path,
            processed_media_path,
            report_metadata
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        report_date,
        image_name,
        hazard_json,
        total_hazards,
        severity,
        latitude,
        longitude,
        location_name,
        "Pending",
        media_type,
        media_path,
        processed_media_path,
        json.dumps(report_metadata) if report_metadata is not None else None,
    ))

    report_id = cursor.lastrowid

    conn.commit()
    conn.close()

    return report_id


# =========================
# GET ALL REPORTS
# =========================

def get_all_reports():
    conn = get_connection()
    conn.row_factory = sqlite3.Row

    cursor = conn.cursor()

    cursor.execute("""
        SELECT * FROM reports
        ORDER BY id DESC
    """)

    reports = [dict(row) for row in cursor.fetchall()]

    conn.close()

    return reports


# =========================
# UPDATE REPORT STATUS
# =========================

def update_report_status(report_id, new_status):
    if new_status not in REPORT_STATUSES:
        raise ValueError(
            f"Invalid status: {new_status}. "
            f"Allowed statuses: {REPORT_STATUSES}"
        )

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        UPDATE reports
        SET status = ?
        WHERE id = ?
    """, (new_status, report_id))

    updated = cursor.rowcount > 0

    conn.commit()
    conn.close()

    return updated


# =========================
# GET STATUS COUNTS
# =========================

def get_report_status_statistics():
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT status, COUNT(*)
        FROM reports
        GROUP BY status
    """)

    results = cursor.fetchall()
    conn.close()

    # Include all statuses, even if count is zero
    status_counts = {
        status: 0 for status in REPORT_STATUSES
    }

    for status, count in results:
        if status in status_counts:
            status_counts[status] = count

    return status_counts


# =========================
# DASHBOARD STATISTICS
# =========================

def get_report_statistics():
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM reports")
    total_reports = cursor.fetchone()[0]

    cursor.execute("""
        SELECT COALESCE(SUM(total_hazards), 0)
        FROM reports
    """)
    total_hazards = cursor.fetchone()[0]

    cursor.execute("""
        SELECT COUNT(*) FROM reports WHERE severity = 'High'
    """)
    high_severity = cursor.fetchone()[0]

    cursor.execute("""
        SELECT COUNT(*) FROM reports WHERE severity = 'Medium'
    """)
    medium_severity = cursor.fetchone()[0]

    cursor.execute("""
        SELECT COUNT(*) FROM reports WHERE severity = 'Low'
    """)
    low_severity = cursor.fetchone()[0]

    # Report status statistics
    cursor.execute("""
        SELECT status, COUNT(*)
        FROM reports
        GROUP BY status
    """)
    status_results = cursor.fetchall()

    status_counts = {
        status: 0 for status in REPORT_STATUSES
    }

    for status, count in status_results:
        if status in status_counts:
            status_counts[status] = count

    conn.close()

    return {
        "total_reports": total_reports,
        "total_hazards": total_hazards,
        "high": high_severity,
        "medium": medium_severity,
        "low": low_severity,

        # New status statistics
        "pending": status_counts["Pending"],
        "verified": status_counts["Verified"],
        "in_progress": status_counts["In Progress"],
        "resolved": status_counts["Resolved"],
        "rejected": status_counts["Rejected"]
    }


# =========================
# DELETE ALL REPORTS
# =========================

def delete_all_reports():
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("DELETE FROM reports")

    conn.commit()
    conn.close()
