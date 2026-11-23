# MajorProject_DrishtiPath
### AI-Based Road Hazard Detection and Geo-Tagged Reporting System

RoadGuard AI is an AI-powered application designed to detect road hazards and damaged road surfaces using computer vision. It identifies road damage from uploaded images, estimates the severity of detected hazards, and helps visualize reported locations through an interactive map.

The system aims to make road hazard identification and reporting easier, supporting safer roads and more efficient road maintenance.

---

## ✨ Features

- **AI-Based Road Hazard Detection** – Detects road damage using a trained YOLO model.
- **Multiple Hazard Classification** – Identifies various road damage categories.
- **Severity Assessment** – Provides a preliminary severity level based on detected hazards.
- **Interactive Map Visualization** – Displays reported road hazards geographically.
- **Geo-Tagged Image Support** – Extracts GPS coordinates from image metadata when available.
- **Manual Location Adjustment** – Allows users to adjust the detected location on the map.
- **Report Management** – Stores road hazard reports using SQLite.
- **Report Status Tracking** – Track reports through Pending, Verified, In Progress, Resolved, and Rejected.
- **Duplicate Report Detection** – Identifies potentially duplicate reports in nearby locations.
- **Analytics Dashboard** – Displays detection statistics and report summaries.
- **Image Upload & Annotation** – Upload road images and view detection results with bounding boxes.
- **CSV Export** – Export report data for further analysis.

---

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Streamlit | Interactive web application |
| YOLOv8 | Road hazard detection |
| OpenCV | Image processing and visualization |
| SQLite | Report and data storage |
| Folium | Interactive maps |
| Streamlit-Folium | Map integration with Streamlit |
| Pandas | Data analysis and statistics |
| Pillow | Image processing and EXIF metadata handling |

---

## 🚨 Road Damage Categories

The model is designed to detect the following road hazard categories:

- 🕳️ Pothole
- 🛣️ Longitudinal Crack
- 🛣️ Transverse Crack
- 🕸️ Alligator Crack
- ⚠️ Edge Damage
- 🧱 Block Crack

---

## 📊 Severity Assessment

Detected hazards are assigned a preliminary severity level based on factors such as:

- Hazard type
- Detection confidence
- Estimated damaged area

Severity is categorized into:

- 🟢 Low
- 🟠 Medium
- 🔴 High

*Severity estimation is a preliminary prototype-level assessment and is not a substitute for professional road inspection.*

---

## 🏗️ Project Architecture

    RoadGuard AI
    │
    ├── Image Upload
    │
    ├── YOLO Detection Engine
    │       └── Hazard Detection & Classification
    │
    ├── Severity Assessment
    │
    ├── Location Processing
    │       ├── GPS Metadata Extraction
    │       └── Manual Map Location Adjustment
    │
    ├── Report Management
    │       └── SQLite Database
    │
    ├── Interactive Map
    │       └── Folium
    │
    └── Analytics Dashboard

---

## 📁 Project Structure

    AI-Road-Hazard-System/
    │
    ├── app.py
    ├── detector.py
    ├── database.py
    ├── map_view.py
    ├── test_location.py
    ├── requirements.txt
    ├── models/
    │   └── road_damage_best.pt
    ├── road_hazards.db
    └── README.md

---

## ⚙️ Installation & Setup

### 1. Clone the repository

```bash
git clone https://github.com/Anjana-Dhakad/AI-Road-Hazard-System.git
```

### 2. Navigate to the project directory

```bash
cd AI-Road-Hazard-System
```

### 3. Create a virtual environment

```bash
python -m venv venv
```

### 4. Activate the virtual environment

**Windows:**

```powershell
venv\Scripts\activate
```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

### 6. Run the application

```bash
streamlit run app.py
```

The application will open in your browser at:

```text
http://localhost:8501
```

---

## 🖥️ How to Use

1. Launch RoadGuard AI.
2. Navigate to **Detect Road Hazards**.
3. Upload a road image (JPG, JPEG, or PNG).
4. Click **Analyze Road Image**.
5. View detected hazards and bounding boxes.
6. Review the preliminary severity assessment.
7. Adjust the location manually on the map if required.
8. Save the report.
9. View and manage reports from the **Map & Reports** section.

---

## 🗺️ Report Management

Each saved report can contain information such as:

- Report ID
- Detected hazard category
- Detection confidence
- Severity level
- Geographic coordinates
- Report location
- Report date and time
- Report status

### Report Status Workflow

`Pending → Verified → In Progress → Resolved`

Reports may also be marked as **Rejected** when they are invalid or not actionable.

---

## 📈 Dashboard

The dashboard provides an overview of road hazard reports and detection activity, including:

- Total reports
- Hazard distribution
- Severity distribution
- Report status statistics
- Recent reports
- Road hazard analytics

---

## 🔮 Future Scope

- Real-time road hazard detection using video.
- GPS-based automatic location extraction improvements.
- Mobile application integration.
- Improved severity estimation using advanced models.
- Automatic notification to relevant road maintenance authorities.
- Crowdsourced road hazard reporting.
- Route safety analysis using hazard locations.
- Integration with government road maintenance systems.

---

## 🎯 Objective

The primary objective of RoadGuard AI is to leverage artificial intelligence and computer vision to identify road hazards, visualize their locations, and organize reports in a centralized system to support road safety and infrastructure maintenance.

---

## 👩‍💻 Developed As

**Academic Major Project – B.Tech Computer Science & Engineering**

---

## ⚖️ Disclaimer

RoadGuard AI is an academic prototype developed for educational and demonstration purposes. AI predictions and severity estimates may not always be accurate and should be verified through actual inspection.

---

⭐ If you find this project interesting, consider starring the repository!
