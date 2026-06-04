\# Water Hyacinth Detection System 🌿

A real-time aquatic plant detection system built with YOLOv5 and Flask.

\## Features

\- Upload image detection with bounding boxes

\- Live webcam detection

\- Web dashboard with detection logs and stats

\- Detects 7 aquatic plant species including Water Hyacinth

\## Tech Stack

\- YOLOv5 (Object Detection)

\- Flask (Web Dashboard)

\- PyTorch

\- OpenCV

\- Python 3.13

\## Setup

1\. Clone the repo

2\. Activate virtual environment: `yoloenv\\Scripts\\activate`

3\. Install Flask: `pip install flask`

4\. Add `best.pt` to project root

5\. Run: `cd dashboard \&\& python app.py`

6\. Open: `http://127.0.0.1:5000`

\## Dataset

Trained on 415 water hyacinth images from Roboflow Universe.

Single class detection: `water\_hyacinth`
