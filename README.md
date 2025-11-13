# Video Feature Extraction Tool

This Python tool analyzes a local video file and extracts key visual and temporal features, outputting them as a JSON object.

It calculates the following features:

1. **Shot Cut Detection**: Counts the number of "hard cuts" in the video.
2. **Motion Analysis**: Calculates the average motion magnitude using Optical Flow.
3. **Text Analysis**: Provides a ratio of frames containing text and extracts the top keywords.
4. **Object/Person Dominance**: Calculates the ratio of people detected versus other objects using a YOLOv8 model.

## CRITICAL: Setup Instructions

This tool has dependencies outside of Python. Please follow these steps carefully.

### Step 1: Install Tesseract-OCR

`pytesseract` is a Python wrapper for the Tesseract-OCR engine. You must install the engine on your system first.

- **Windows**: Download and run the installer from the [official Tesseract repository](https://github.com/tesseract-ocr/tesseract). Make sure to note the installation path (e.g., `C:\Program Files\Tesseract-OCR\tesseract.exe`). You may need to add this path to your system's PATH environment variable.

- **macOS**: Use Homebrew:
  ```
  brew install tesseract
  ```

- **Linux (Ubuntu/Debian)**:
  ```
  sudo apt-get update
  sudo apt-get install tesseract-ocr
  ```

### Step 2: Create a Python Environment

It is highly recommended to use a virtual environment.

```
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

### Step 3: Install Python Dependencies

With your virtual environment active, install the required packages from `requirements.txt`.

```
pip install -r requirements.txt
```

## How to Run

Once setup is complete, you can run the tool from your terminal.

```
python video_feature_extractor.py --video_path /path/to/your/video.mp4
```

Replace `/path/to/your/video.mp4` with the actual path to your video file. You can use any common video file (e.g., `.mp4`, `.avi`, `.mov`).

## Sample Output

The tool will print a JSON object to the console, similar to this:

```json
{
  "file_name": "Recording 2025-11-13 094737.mp4",
  "duration_seconds": 7.566666666666666,
  "fps": 30.0,
  "resolution": "1000x650",
  "shot_cut_detection": {
    "hard_cuts_found": 0,
    "frames_analyzed": 226,
    "correlation_threshold": 0.8
  },
  "motion_analysis": {
    "average_motion_magnitude": 0.044
  },
  "text_analysis": {
    "text_present_ratio": 0.0,
    "total_frames_sampled": 4,
    "frames_with_text": 0,
    "top_keywords": []
  },
  "object_person_dominance": {
    "person_dominance_ratio": 0.5,
    "total_persons_detected": 4,
    "total_other_objects_detected": 4
  }
}
```
