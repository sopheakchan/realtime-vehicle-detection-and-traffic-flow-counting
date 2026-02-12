from flask import Flask, render_template, request, jsonify, send_file
import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict
from datetime import datetime
import openpyxl
from openpyxl.styles import Font, Alignment, PatternFill
import os
import base64
from werkzeug.utils import secure_filename
import threading

app = Flask(__name__)

# Config
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app.config['UPLOAD_FOLDER'] = os.path.join(BASE_DIR, 'uploads')
app.config['OUTPUT_FOLDER'] = os.path.join(BASE_DIR, 'outputs')
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

#Hawktuahuuuuuu

# Global State
video_path = None
detection_area = []
counting_line = []
processing_fps = 30 # Default
processing_progress = 0
processing_status = "idle"
is_cancelled = False 
counter_instance = None
current_frame = 0
total_frames = 0
output_video_path = None
show_detection_area = True # Global flag

class VehicleLineCounter:
    def __init__(self, detection_area, counting_line, show_area=True, model_path="../best.pt"):
        self.show_area = show_area
        self.detection_area = np.array(detection_area, np.int32)
        self.line_start = (int(counting_line[0][0]), int(counting_line[0][1]))
        self.line_end = (int(counting_line[1][0]), int(counting_line[1][1]))
        self.model = YOLO(model_path) 
        self.tracked_vehicles = {}
        self.total_count = 0
        self.class_counts = defaultdict(int)
        self.class_names = ['Bus', 'Car', 'Motocycle', 'Rickshaw', 'Truck']
        self.colors = {0: (0, 0, 255), 1: (255, 0, 0), 2: (0, 255, 0), 3: (255, 255, 0), 4: (255, 0, 255)}
    
    def point_in_polygon(self, point):
        return cv2.pointPolygonTest(self.detection_area, point, False) >= 0
    
    def line_intersection(self, p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = self.line_start
        x4, y4 = self.line_end
        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(denom) < 1e-10: return False
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
        u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denom
        return 0 <= t <= 1 and 0 <= u <= 1
    
    def update_tracking(self, track_id, center, class_id, class_name):
        if track_id not in self.tracked_vehicles:
            self.tracked_vehicles[track_id] = {'last_position': center, 'crossed': False, 'class_id': class_id, 'class_name': class_name}
            return False
        vehicle = self.tracked_vehicles[track_id]
        if not vehicle['crossed'] and self.line_intersection(vehicle['last_position'], center):
            vehicle['crossed'] = True
            self.total_count += 1
            self.class_counts[class_name] += 1
            vehicle['last_position'] = center
            return True
        vehicle['last_position'] = center
        return False

    def draw_ui(self, frame):
        # Only draw the Detection Polygon if the toggle is enabled
        if self.show_area:
            overlay = frame.copy()
            cv2.fillPoly(overlay, [self.detection_area], (0, 255, 0))
            cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
            cv2.polylines(frame, [self.detection_area], True, (0, 255, 0), 2)
        
        # Always draw the yellow counting line
        cv2.line(frame, self.line_start, self.line_end, (0, 255, 255), 4)
        
        # Draw stats dashboard
        y_offset = 40
        cv2.rectangle(frame, (10, 10), (300, 40 + len(self.class_names) * 35), (0, 0, 0), -1)
        cv2.putText(frame, f"TOTAL: {self.total_count}", (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y_offset += 35
        for i, name in enumerate(self.class_names):
            cv2.putText(frame, f"{name}: {self.class_counts[name]}", (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors[i], 2)
            y_offset += 30
        return frame

    def process_video(self, video_path, output_path, target_fps):
        global processing_progress, processing_status, current_frame, total_frames, is_cancelled
        try:
            cap = cv2.VideoCapture(video_path)
            orig_fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            frame_skip = max(1, int(orig_fps / target_fps)) if target_fps > 0 else 1
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, target_fps, (width, height))
            
            processing_status = "processing"
            
            while True:
                if is_cancelled:
                    processing_status = "idle"
                    break

                ret, frame = cap.read()
                if not ret: break
                
                current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                
                results = self.model.track(source=frame, imgsz=640, conf=0.25, verbose=False, persist=True)
                
                if results[0].boxes is not None and results[0].boxes.id is not None:
                    boxes = results[0].boxes.xyxy.cpu().numpy()
                    track_ids = results[0].boxes.id.cpu().numpy().astype(int)
                    classes = results[0].boxes.cls.cpu().numpy().astype(int)
                    
                    for box, tid, cls in zip(boxes, track_ids, classes):
                        x1, y1, x2, y2 = box
                        center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
                        
                        if self.point_in_polygon(center):
                            c_name = self.class_names[cls]
                            just_crossed = self.update_tracking(tid, center, cls, c_name)
                            color = (0, 255, 0) if just_crossed else self.colors[cls]
                            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

                frame = self.draw_ui(frame)
                out.write(frame)
                
                processing_progress = min(int((current_frame / total_frames) * 100), 100)
                
                if frame_skip > 1:
                    for _ in range(frame_skip - 1):
                        cap.grab() 
            
            cap.release()
            out.release()
            
            if not is_cancelled:
                processing_status = "complete"
                processing_progress = 100
            
        except Exception as e:
            processing_status = "error"
            print(f"Error: {e}")

    def save_to_excel(self, excel_path):
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.title = "Traffic Report"
        ws.append(["Vehicle Class", "Count", "Percentage"])
        for name in self.class_names:
            count = self.class_counts[name]
            perc = (count / self.total_count * 100) if self.total_count > 0 else 0
            ws.append([name, count, f"{perc:.1f}%"])
        ws.append(["TOTAL", self.total_count, "100%"])
        wb.save(excel_path)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_video():
    global video_path, is_cancelled
    is_cancelled = False 
    file = request.files['video']
    filename = secure_filename(file.filename)
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(video_path)
    return jsonify({'success': True})

@app.route('/first_frame')
def get_first_frame():
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    _, buffer = cv2.imencode('.jpg', frame)
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    return jsonify({'image': f'data:image/jpeg;base64,{img_base64}'})

@app.route('/set_regions', methods=['POST'])
def set_regions():
    global detection_area, counting_line, processing_fps, show_detection_area
    data = request.json
    detection_area = data.get('detection_area', [])
    counting_line = data.get('counting_line', [])
    processing_fps = int(data.get('processing_fps', 30))
    # Capture the toggle value from frontend
    show_detection_area = data.get('show_detection_area', True) 
    return jsonify({'success': True})

@app.route('/process', methods=['POST'])
def process_video():
    global video_path, detection_area, counting_line, processing_fps, show_detection_area
    global counter_instance, processing_status, processing_progress, output_video_path, is_cancelled
    
    is_cancelled = False
    output_video_path = os.path.join(app.config['OUTPUT_FOLDER'], f'out_{datetime.now().strftime("%H%M%S")}.mp4')
    
    # Initialize instance with the show_area flag
    counter_instance = VehicleLineCounter(
        detection_area, 
        counting_line, 
        show_area=show_detection_area, 
        model_path="../best.pt"
    )
    
    thread = threading.Thread(target=counter_instance.process_video, args=(video_path, output_video_path, processing_fps))
    thread.daemon = True
    thread.start()
    return jsonify({'success': True})

@app.route('/cancel', methods=['POST'])
def cancel_processing():
    global is_cancelled
    is_cancelled = True
    return jsonify({'success': True})

@app.route('/progress')
def get_progress():
    global processing_progress, processing_status, current_frame, total_frames
    return jsonify({
        'progress': processing_progress, 
        'status': processing_status,
        'current_frame': current_frame,
        'total_frames': total_frames
    })

@app.route('/download_video')
def download_video():
    return send_file(output_video_path, as_attachment=True)

@app.route('/download_excel')
def download_excel():
    excel_path = os.path.join(app.config['OUTPUT_FOLDER'], 'report.xlsx')
    counter_instance.save_to_excel(excel_path)
    return send_file(excel_path, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True, port=5000)