
import sqlite3
import json
from datetime import datetime

DB_NAME = "road_hazards.db"


def get_connection():
    return sqlite3.connect(DB_NAME)


def initialize_database():
    conn = get_connection()
    cursor = conn.cursor()

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
            location_name TEXT
        )
    """)

    conn.commit()
    conn.close()


def save_report(
    image_name,
    detections,
    total_hazards,
    severity,
    latitude,
    longitude,
    location_name
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
            location_name
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        report_date,
        image_name,
        hazard_json,
        total_hazards,
        severity,
        latitude,
        longitude,
        location_name
    ))

    report_id = cursor.lastrowid

    conn.commit()
    conn.close()

    return report_id


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

    conn.close()

    return {
        "total_reports": total_reports,
        "total_hazards": total_hazards,
        "high": high_severity,
        "medium": medium_severity,
        "low": low_severity
    }


def delete_all_reports():
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("DELETE FROM reports")

    conn.commit()
    conn.close()
