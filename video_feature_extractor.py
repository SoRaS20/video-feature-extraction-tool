import cv2
import numpy as np
import pytesseract
from ultralytics import YOLO
from tqdm import tqdm
import argparse
import json
import os
import sys
from sklearn.feature_extraction.text import CountVectorizer


class VideoAnalyzer:

    def __init__(self, video_path):
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found at: {video_path}")

        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise IOError(f"Could not open video file: {video_path}")

        # Basic video properties
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.duration_seconds = self.frame_count / self.fps
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Main features dictionary
        self.features = {
            "file_name": os.path.basename(video_path),
            "duration_seconds": self.duration_seconds,
            "fps": self.fps,
            "resolution": f"{self.width}x{self.height}"
        }

        try:
            self.yolo_model = YOLO('yolov8n.pt')
        except Exception as e:
            print(f"Warning: Could not load YOLO model. Object/Person detection will be skipped. Error: {e}",
                  file=sys.stderr)
            self.yolo_model = None

    def _reset_video(self):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def analyze_shot_cuts(self, frame_skip=1, threshold=0.8):
        print("Analyzing shot cuts...")
        self._reset_video()

        hard_cuts = 0
        total_frames_analyzed = 0

        ret, prev_frame = self.cap.read()
        if not ret:
            self.features['shot_cut_detection'] = {"error": "Could not read first frame."}
            return

        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        prev_hist = cv2.calcHist([prev_gray], [0], None, [256], [0, 256])
        cv2.normalize(prev_hist, prev_hist, 0, 1, cv2.NORM_MINMAX)

        frame_number = 0
        with tqdm(total=self.frame_count, desc="Shot Cut Detection") as pbar:
            while True:
                ret, curr_frame = self.cap.read()
                if not ret:
                    break

                frame_number += 1
                pbar.update(1)

                if frame_number % frame_skip != 0:
                    continue

                total_frames_analyzed += 1
                curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
                curr_hist = cv2.calcHist([curr_gray], [0], None, [256], [0, 256])
                cv2.normalize(curr_hist, curr_hist, 0, 1, cv2.NORM_MINMAX)

                method = getattr(cv2, "HISTCMP_CORRELATION", 0)
                correlation = cv2.compareHist(prev_hist, curr_hist, method)

                if correlation < threshold:
                    hard_cuts += 1

                prev_hist = curr_hist

        self.features['shot_cut_detection'] = {
            "hard_cuts_found": hard_cuts,
            "frames_analyzed": total_frames_analyzed,
            "correlation_threshold": threshold
        }

    def analyze_motion(self, frame_skip=5):
        print("Analyzing motion...")
        self._reset_video()

        motion_magnitudes = []
        ret, prev_frame = self.cap.read()
        if not ret:
            self.features['motion_analysis'] = {"error": "Could not read first frame."}
            return

        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

        frame_number = 0
        with tqdm(total=self.frame_count, desc="Motion Analysis") as pbar:
            while True:
                ret, curr_frame = self.cap.read()
                if not ret:
                    break

                frame_number += 1
                pbar.update(1)

                if frame_number % frame_skip != 0:
                    continue

                curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

                flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

                mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                motion_magnitudes.append(np.mean(mag))

                prev_gray = curr_gray

        avg_motion = np.mean(motion_magnitudes) if motion_magnitudes else 0
        self.features['motion_analysis'] = {
            "average_motion_magnitude": round(float(avg_motion), 3)
        }

    def analyze_text(self, sample_rate_seconds=2):
        print("Analyzing text (OCR)")
        self._reset_video()

        frame_skip = int(self.fps * sample_rate_seconds)
        if frame_skip == 0:
            frame_skip = 1

        frames_with_text = 0
        total_frames_sampled = 0
        all_detected_text = []

        frame_number = 0
        with tqdm(total=self.frame_count // frame_skip, desc="Text Analysis") as pbar:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break

                if frame_number % frame_skip == 0:
                    total_frames_sampled += 1
                    try:
                        data = pytesseract.image_to_data(frame, output_type=pytesseract.Output.DICT)
                        n_boxes = len(data['level'])
                        frame_text = []
                        for i in range(n_boxes):
                            text = data['text'][i].strip()
                            if len(text) > 2:
                                frame_text.append(text)

                        if frame_text:
                            frames_with_text += 1
                            all_detected_text.extend(frame_text)
                    except Exception as e:
                        print(f"Warning: pytesseract error on frame {frame_number}: {e}", file=sys.stderr)

                    pbar.update(1)

                frame_number += 1

        top_keywords = []
        if all_detected_text:
            try:
                vectorizer = CountVectorizer(stop_words='english', max_features=10)
                word_counts = vectorizer.fit_transform(all_detected_text)
                top_keywords = vectorizer.get_feature_names_out().tolist()
            except ValueError:
                pass

        self.features['text_analysis'] = {
            "text_present_ratio": round(frames_with_text / total_frames_sampled, 3) if total_frames_sampled > 0 else 0,
            "total_frames_sampled": total_frames_sampled,
            "frames_with_text": frames_with_text,
            "top_keywords": top_keywords
        }

    def analyze_object_person(self, sample_rate_seconds=2):
        if not self.yolo_model:
            self.features['object_person_dominance'] = {"error": "YOLO model not loaded. Skipped."}
            return

        print("Analyzing object/person dominance...")
        self._reset_video()

        frame_skip = int(self.fps * sample_rate_seconds)
        if frame_skip == 0:
            frame_skip = 1

        total_persons = 0
        total_objects = 0

        frame_number = 0
        with tqdm(total=self.frame_count // frame_skip, desc="Object/Person Detection") as pbar:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break

                if frame_number % frame_skip == 0:
                    results = self.yolo_model(frame, verbose=False)

                    for r in results:
                        for box in r.boxes:
                            cls_id = int(box.cls[0])
                            class_name = self.yolo_model.names[cls_id]

                            if class_name == 'person':
                                total_persons += 1
                            else:
                                total_objects += 1
                    pbar.update(1)

                frame_number += 1

        total_detections = total_persons + total_objects
        person_ratio = total_persons / (total_detections + 1e-6)

        self.features['object_person_dominance'] = {
            "person_dominance_ratio": round(person_ratio, 3),
            "total_persons_detected": total_persons,
            "total_other_objects_detected": total_objects
        }

    def analyze_all(self):
        try:
            self.analyze_shot_cuts()
            self.analyze_motion()
            self.analyze_text()
            self.analyze_object_person()
        finally:
            self.close()

    def get_features_json(self):
        return json.dumps(self.features, indent=2)

    def close(self):
        if self.cap.isOpened():
            self.cap.release()


# Main execution
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Video Feature Extraction Tool")
    parser.add_argument(
        "--video_path",
        type=str,
        required=True,
        help="Path to the local video file."
    )

    args = parser.parse_args()

    try:
        analyzer = VideoAnalyzer(args.video_path)
        print(f"Starting analysis for: {args.video_path}\n")

        analyzer.analyze_all()

        # Print the final JSON output
        print("\nAnalysis Report")
        print(analyzer.get_features_json())

    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except IOError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}", file=sys.stderr)
        sys.exit(1)