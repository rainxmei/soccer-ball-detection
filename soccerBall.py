import cv2
import argparse
import numpy as np
from ultralytics import YOLO
import time
from collections import deque

def parse_args():
    parser = argparse.ArgumentParser(description='YOLO Object Detection dengan OpenCV')
    parser.add_argument('--source', type=str, default='0', 
                        help='Sumber input: path gambar/video atau 0 untuk webcam')
    parser.add_argument('--model', type=str, default='yolov8n.pt',
                        help='Path ke model YOLO (.pt file)')
    parser.add_argument('--conf', type=float, default=0.25,
                        help='Confidence threshold (0-1)')
    parser.add_argument('--show-fps', action='store_true', default=True,
                        help='Tampilkan FPS counter')
    parser.add_argument('--blur-bg', action='store_true',
                        help='Blur background (objek tetap tajam)')
    parser.add_argument('--track', action='store_true',
                        help='Enable object tracking')
    parser.add_argument('--count', action='store_true',
                        help='Hitung jumlah objek per kelas')
    return parser.parse_args()

class ObjectDetector:
    def __init__(self, model_path, conf_threshold=0.25):
        print(f"Loading model: {model_path}")
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.fps_queue = deque(maxlen=30)
        self.colors = np.random.randint(0, 255, size=(100, 3), dtype=np.uint8)
        
    def detect(self, frame, track=False):
        if track:
            results = self.model.track(frame, conf=self.conf_threshold, persist=True)
        else:
            results = self.model(frame, conf=self.conf_threshold)
        return results[0]
    
    def draw_detections(self, frame, result, show_fps=True, blur_bg=False, count_objects=False):
        annotated = frame.copy()
        
        # Bagian untuk blur background
        if blur_bg and len(result.boxes) > 0:
            mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
            
            blurred = cv2.GaussianBlur(frame, (21, 21), 0)
            annotated = np.where(mask[:, :, None] == 255, frame, blurred)
        
        # Untuk menghitung objek
        object_count = {}
        
        # Draw bounding boxes (dari AI)
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            label = result.names[cls]
            
            # Bagian hitung objek
            if count_objects:
                object_count[label] = object_count.get(label, 0) + 1
            
            # Warna berdasarkan kelas(AI)
            color = tuple(map(int, self.colors[cls % len(self.colors)]))
            
            # Draw box dan label(AI)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            
            # ID tracking jika ada(AI)
            track_id = ""
            if hasattr(box, 'id') and box.id is not None:
                track_id = f" ID:{int(box.id[0])}"
            
            text = f"{label}{track_id} {conf:.2f}"
            
            # Background untuk text(AI)
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(annotated, (x1, y1 - th - 10), (x1 + tw, y1), color, -1)
            cv2.putText(annotated, text, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Untuk menampilkan FPS
        if show_fps and len(self.fps_queue) > 0:
            fps = 1.0 / (sum(self.fps_queue) / len(self.fps_queue))
            cv2.putText(annotated, f"FPS: {fps:.1f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # untuk menampilkan jumlah objek
        if count_objects and object_count:
            y_pos = 70
            for obj_name, count in object_count.items():
                text = f"{obj_name}: {count}"
                cv2.putText(annotated, text, (10, y_pos),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                y_pos += 35
        
        return annotated

def main():
    args = parse_args()
    
    # Initialize detector
    detector = ObjectDetector(args.model, args.conf)
    
    # Setup input source
    if args.source == '0':
        cap = cv2.VideoCapture(0)
        is_video = True
    elif args.source.isdigit():
        cap = cv2.VideoCapture(int(args.source))
        is_video = True
    else:
        # Untuk cek gambar atau video
        if args.source.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            is_video = False
            frame = cv2.imread(args.source)
            if frame is None:
                print(f"Error: Tidak dapat membaca gambar {args.source}")
                return
        else:
            cap = cv2.VideoCapture(args.source)
            is_video = True
    
    # Setup video writer jika perlu save(AI)
    if is_video:
        fps = int(cap.get(cv2.CAP_PROP_FPS)) if cap.get(cv2.CAP_PROP_FPS) > 0 else 30
    
    print("\n=== YOLO Object Detection ===")
    print(f"Model: {args.model}")
    print(f"Source: {args.source}")
    print(f"Confidence: {args.conf}")
    print("\nTekan 'q' untuk keluar, 's' untuk screenshot")
    print("==============================\n")
    
    # Process gambar statis(AI)
    if not is_video:
        start = time.time()
        result = detector.detect(frame, track=False)
        detector.fps_queue.append(time.time() - start)
        
        annotated = detector.draw_detections(frame, result, args.show_fps, 
                                             args.blur_bg, args.count)
        
        cv2.imshow('YOLO Detection', annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return
    
    # Process start video/webcam
    frame_count = 0
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            start = time.time()
            
            # Deteksi objek
            result = detector.detect(frame, track=args.track)
            
            # Draw hasil
            annotated = detector.draw_detections(frame, result, args.show_fps,
                                                args.blur_bg, args.count)
            
            # Hitung FPS
            detector.fps_queue.append(time.time() - start)
            
            # Tampilkan hasil
            cv2.imshow('YOLO Detection', annotated)
            
            # Q untuk keluar dan S untuk screenshot
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                filename = f'screenshot_{frame_count}.jpg'
                cv2.imwrite(filename, annotated)
                print(f"Screenshot disimpan: {filename}")
            
            frame_count += 1
            
    except KeyboardInterrupt:
        print("\nDeteksi dihentikan oleh user")
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()