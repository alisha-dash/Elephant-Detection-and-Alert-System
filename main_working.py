import tkinter as tk
from tkinter import messagebox, filedialog
import threading
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image, ImageTk
import time
import mysql.connector
from datetime import datetime


# Load your elephant classification model
model = tf.keras.models.load_model("final_elephant_model.h5")

# Strict parameters for ELEPHANT-ONLY detection
CONFIDENCE_THRESHOLD = 0.85  # High threshold for initial elephant detection
CONFIRMATION_THRESHOLD = 0.75  # High threshold for confirming existing detection
PATCH_SIZE = 220
STRIDE = 110
DETECTION_INTERVAL = 20  # Run detection every 20 frames when tracking
CONFIRMATION_INTERVAL = 5   # Check confirmation every 5 frames

PERSISTENCE_FRAMES = 60     # Keep box for 60 frames after last confirmation

def preprocess_frame(frame):
    """Preprocess frame for model prediction"""
    resized = cv2.resize(frame, (PATCH_SIZE, PATCH_SIZE))
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    eq = cv2.equalizeHist(gray)
    enhanced = cv2.cvtColor(eq, cv2.COLOR_GRAY2RGB)
    img_array = np.array(enhanced, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

class ElephantDetectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Persistent Elephant Detection")
        self.root.geometry("900x700")

        # Video capture variables
        self.cap = None
        self.thread = None
        self.stop_event = threading.Event()
        
        # Detection state variables
        self.frame_count = 0
        self.elephant_detected = False
        self.current_bbox = None
        self.last_confirmed_frame = 0
        self.detection_confidence = 0.0
        self.confirmation_count = 0
        
        # Tracking variables
        self.tracker = None
        self.tracking_active = False
        self.tracking_bbox = None
        
        self.setup_widgets()
        
                # Setup MySQL connection
        try:
            self.db = mysql.connector.connect(
                host="localhost",           # ✅ Your MySQL host (use IP if remote)
                user="root",       # ✅ Your MySQL username
                password="alishadash",   # ✅ Your MySQL password
                database="elephant_detection_db"    # ✅ Your MySQL database name
            )
            self.cursor = self.db.cursor()
        except Exception as e:
            messagebox.showerror("Database Error", f"MySQL connection failed: {e}")

    def log_detection_to_mysql(self, detected):
        """Log detection status (YES/NO) with timestamp to MySQL"""
        try:
            status = "YES" if detected else "NO"
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            query = "INSERT INTO elephant_detection_log (timestamp, status) VALUES (%s, %s)"
            self.cursor.execute(query, (timestamp, status))
            self.db.commit()
        except Exception as e:
            print(f"MySQL Logging Error: {e}")


    def setup_widgets(self):
        """Setup the GUI widgets"""
        # Title
        title_label = tk.Label(self.root, text="Elephant Detection System", 
                              font=('Arial', 16, 'bold'))
        title_label.pack(pady=10)
        
        # Main frame for video
        main_frame = tk.Frame(self.root, bg='black', relief='sunken', bd=2)
        main_frame.pack(expand=True, fill='both', padx=15, pady=10)
        
        # Video display
        self.video_label = tk.Label(main_frame, bg='black', text="Click 'Start Webcam' to begin detection")
        self.video_label.pack(expand=True, fill='both')

        # Control buttons frame
        btn_frame = tk.Frame(self.root)
        btn_frame.pack(pady=15)

        self.start_btn = tk.Button(btn_frame, text="Start Webcam", 
                                  command=self.start_webcam, width=15, height=2,
                                  bg='green', fg='white', font=('Arial', 10, 'bold'))
        self.start_btn.grid(row=0, column=0, padx=10)

        self.capture_btn = tk.Button(btn_frame, text="Capture Photo", 
                                    command=self.capture_photo, width=15, height=2,
                                    bg='blue', fg='white', font=('Arial', 10, 'bold'))
        self.capture_btn.grid(row=0, column=1, padx=10)

        self.stop_btn = tk.Button(btn_frame, text="Stop Detection", 
                                 command=self.stop_video, width=15, height=2,
                                 bg='red', fg='white', font=('Arial', 10, 'bold'))
        self.stop_btn.grid(row=0, column=2, padx=10)

        self.reset_btn = tk.Button(btn_frame, text="Reset Detection", 
                                  command=self.reset_detection, width=15, height=2,
                                  bg='orange', fg='white', font=('Arial', 10, 'bold'))
        self.reset_btn.grid(row=0, column=3, padx=10)

        # Status frame
        status_frame = tk.Frame(self.root, relief='ridge', bd=2)
        status_frame.pack(fill='x', padx=15, pady=5)
        
        self.status_label = tk.Label(status_frame, text="Status: Ready to start detection", 
                                   font=('Arial', 12), fg='blue')
        self.status_label.pack(pady=5)
        
        self.detection_info = tk.Label(status_frame, text="Detection Info: No elephant detected", 
                                     font=('Arial', 10))
        self.detection_info.pack()

    def start_webcam(self):
        """Start webcam capture and detection"""
        if self.cap:
            self.stop_video()
        
        try:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                messagebox.showerror("Error", "Cannot open webcam")
                return
            
            # Set optimal camera properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            self.reset_detection_state()
            self.start_detection_thread()
            self.status_label.config(text="Status: Webcam active - Scanning for elephants...", fg='green')
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start webcam: {str(e)}")

    def stop_video(self):
        """Stop video capture and detection"""
        self.stop_event.set()
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=3)
        
        self.stop_event.clear()
        if self.cap:
            self.cap.release()
            self.cap = None
        
        self.video_label.config(image='', text="Detection stopped")
        self.status_label.config(text="Status: Stopped", fg='red')
        self.detection_info.config(text="Detection Info: Stopped")

    def reset_detection(self):
        """Reset detection state while keeping video running"""
        self.reset_detection_state()
        self.status_label.config(text="Status: Detection reset - Scanning for elephants...", fg='blue')
        self.detection_info.config(text="Detection Info: Detection state reset")

    def reset_detection_state(self):
        """Reset all detection variables"""
        self.frame_count = 0
        self.elephant_detected = False
        self.current_bbox = None
        self.last_confirmed_frame = 0
        self.detection_confidence = 0.0
        self.confirmation_count = 0
        self.tracker = None
        self.tracking_active = False
        self.tracking_bbox = None

    def capture_photo(self):
        """Capture and save current frame"""
        if self.cap is None:
            messagebox.showwarning("Warning", "Start webcam first!")
            return
        
        ret, frame = self.cap.read()
        if not ret:
            messagebox.showerror("Error", "Failed to capture frame")
            return
        
        filename = filedialog.asksaveasfilename(
            defaultextension=".jpg",
            filetypes=[("JPEG files", "*.jpg"), ("PNG files", "*.png"), ("All files", "*.*")]
        )
        if filename:
            cv2.imwrite(filename, frame)
            messagebox.showinfo("Success", f"Photo saved as {filename}")

    def start_detection_thread(self):
        """Start the video processing thread"""
        self.thread = threading.Thread(target=self.video_loop, daemon=True)
        self.thread.start()

    def scan_for_elephant(self, frame):
        """Strict elephant-only detection in frame"""
        h, w, _ = frame.shape
        elephant_detections = []

        # Single scale detection with strict criteria
        for y in range(0, max(1, h - PATCH_SIZE + 1), STRIDE):
            for x in range(0, max(1, w - PATCH_SIZE + 1), STRIDE):
                y_end = min(y + PATCH_SIZE, h)
                x_end = min(x + PATCH_SIZE, w)
                patch = frame[y:y_end, x:x_end]
                
                # Ensure patch is exactly the right size
                if patch.shape[0] != PATCH_SIZE or patch.shape[1] != PATCH_SIZE:
                    continue
                
                try:
                    img_array = preprocess_frame(patch)
                    pred = model.predict(img_array, verbose=0)[0]
                    class_idx = np.argmax(pred)
                    confidence = pred[class_idx]

                    # STRICT: Only class 1 (elephant) with HIGH confidence
                    if class_idx == 1 and confidence > CONFIDENCE_THRESHOLD:
                        # Additional validation: check that non-elephant confidence is low
                        non_elephant_confidence = pred[0] if len(pred) > 1 else 0
                        confidence_gap = confidence - non_elephant_confidence
                        
                        # Only accept if elephant confidence is significantly higher
                        if confidence_gap > 0.3:  # At least 30% gap
                            elephant_detections.append({
                                'bbox': (x, y, x_end - x, y_end - y),
                                'confidence': confidence,
                                'confidence_gap': confidence_gap,
                                'center': (x + (x_end - x)//2, y + (y_end - y)//2)
                            })
                except Exception:
                    continue

        # Return only the most confident elephant detection
        if elephant_detections:
            # Sort by confidence gap first, then by confidence
            best_detection = max(elephant_detections, 
                               key=lambda d: (d['confidence_gap'], d['confidence']))
            return best_detection
        
        return None

    def confirm_elephant_in_region(self, frame, bbox):
        """Strict confirmation of elephant presence in region"""
        x, y, w, h = bbox
        
        # Use exact region without expansion to avoid false positives
        region = frame[y:y+h, x:x+w]
        
        if region.shape[0] < PATCH_SIZE//2 or region.shape[1] < PATCH_SIZE//2:
            return False, 0.0
        
        try:
            img_array = preprocess_frame(region)
            pred = model.predict(img_array, verbose=0)[0]
            class_idx = np.argmax(pred)
            confidence = pred[class_idx]
            
            # STRICT: Must be elephant class with high confidence
            if class_idx == 1 and confidence > CONFIRMATION_THRESHOLD:
                # Additional check: ensure it's significantly more elephant than non-elephant
                non_elephant_confidence = pred[0] if len(pred) > 1 else 0
                confidence_gap = confidence - non_elephant_confidence
                
                if confidence_gap > 0.25:  # At least 25% gap for confirmation
                    return True, confidence
            
            return False, confidence
        except Exception:
            return False, 0.0

    def update_tracker_if_active(self, frame):
        """Update tracker and return updated bbox if successful"""
        if not self.tracking_active or self.tracker is None:
            return None
        
        try:
            success, bbox = self.tracker.update(frame)
            if success:
                x, y, w, h = [int(v) for v in bbox]
                # Validate bbox bounds
                if (x >= 0 and y >= 0 and x + w <= frame.shape[1] and 
                    y + h <= frame.shape[0] and w > 30 and h > 30):
                    return (x, y, w, h)
        except Exception:
            pass
        
        return None

    def initialize_tracker(self, frame, bbox):
        """Initialize tracker with detected bbox"""
        try:
            self.tracker = cv2.TrackerCSRT_create()
            success = self.tracker.init(frame, bbox)
            if success:
                self.tracking_active = True
                self.tracking_bbox = bbox
                return True
        except Exception:
            pass
        
        self.tracker = None
        self.tracking_active = False
        return False

    def draw_persistent_detection(self, frame):
        """Draw the persistent detection box and info"""
        if not self.elephant_detected or self.current_bbox is None:
            return
        
        x, y, w, h = self.current_bbox
        
        # Determine box color based on how recent the confirmation is
        frames_since_confirmation = self.frame_count - self.last_confirmed_frame
        
        if frames_since_confirmation <= 10:
            color = (0, 255, 0)  # Bright green - recently confirmed
            status = "DETECTED"
        elif frames_since_confirmation <= 30:
            color = (0, 200, 100)  # Medium green - somewhat recent
            status = "TRACKING"
        else:
            color = (0, 150, 150)  # Yellow-green - older detection
            status = "PERSISTENT"
        
        # Draw main bounding box
        thickness = 3
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)
        
        # Draw corner markers for better visibility
        corner_size = 15
        for corner_x, corner_y in [(x, y), (x + w, y), (x, y + h), (x + w, y + h)]:
            cv2.rectangle(frame, (corner_x - corner_size//2, corner_y - corner_size//2),
                         (corner_x + corner_size//2, corner_y + corner_size//2), color, -1)
        
        # Draw info text with background
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        thickness = 2
        
        text1 = f"ELEPHANT {status}"
        text2 = f"Confidence: {self.detection_confidence:.3f}"
        text3 = f"Verified: {self.confirmation_count} times"
        
        # Calculate text positions
        y_offset = y - 15 if y > 80 else y + h + 25
        
        for i, text in enumerate([text1, text2, text3]):
            (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, thickness)
            text_y = y_offset + i * (text_h + 5)
            
            # Draw text background
            cv2.rectangle(frame, (x, text_y - text_h - 5), (x + text_w + 10, text_y + 5), color, -1)
            # Draw text
            cv2.putText(frame, text, (x + 5, text_y), font, font_scale, (0, 0, 0), thickness)

    def video_loop(self):
        """Main video processing loop with persistent detection"""
        while not self.stop_event.is_set():
            try:
                ret, frame = self.cap.read()
                if not ret:
                    break

                self.frame_count += 1
                
                # Resize frame for consistent processing
                height, width = frame.shape[:2]
                if width > 640:
                    scale = 640 / width
                    frame = cv2.resize(frame, (640, int(height * scale)))

                # MAIN DETECTION LOGIC
                if not self.elephant_detected:
                    # Initial detection phase
                    detection = self.scan_for_elephant(frame)
                    if detection and detection['confidence'] > CONFIDENCE_THRESHOLD:
                        # Elephant found!
                        self.elephant_detected = True
                        self.current_bbox = detection['bbox']
                        self.detection_confidence = detection['confidence']
                        self.last_confirmed_frame = self.frame_count
                        self.confirmation_count = 1
                        
                        # Initialize tracker
                        self.initialize_tracker(frame, self.current_bbox)
                        
                        self.status_label.config(text="Status: ELEPHANT DETECTED! Maintaining lock...", fg='red')
                        self.detection_info.config(text=f"Detection Info: Elephant found with {detection['confidence']:.2f} confidence")
                        self.log_detection_to_mysql(True)

                else:
                    # Elephant already detected - maintain detection
                    bbox_updated = False
                    
                    # Try to update with tracker first
                    if self.tracking_active:
                        tracked_bbox = self.update_tracker_if_active(frame)
                        if tracked_bbox:
                            self.current_bbox = tracked_bbox
                            bbox_updated = True
                    
                    # Periodic confirmation check
                    if self.frame_count % CONFIRMATION_INTERVAL == 0:
                        is_confirmed, conf_score = self.confirm_elephant_in_region(frame, self.current_bbox)
                        if is_confirmed and conf_score > CONFIRMATION_THRESHOLD:
                            self.last_confirmed_frame = self.frame_count
                            self.confirmation_count += 1
                            self.detection_confidence = max(self.detection_confidence, conf_score)
                    
                    # Full re-detection check (less frequent)
                    if self.frame_count % DETECTION_INTERVAL == 0:
                        detection = self.scan_for_elephant(frame)
                        if detection and detection['confidence'] > CONFIRMATION_THRESHOLD:
                            # Double-check: must be significantly better than current detection
                            if detection['confidence'] > self.detection_confidence * 0.9:
                                # Update bbox with new detection
                                self.current_bbox = detection['bbox']
                                self.detection_confidence = detection['confidence']
                                self.last_confirmed_frame = self.frame_count
                                self.confirmation_count += 1
                                
                                # Reinitialize tracker with new position
                                self.initialize_tracker(frame, self.current_bbox)
                    
                    # Check if we should keep the detection
                    frames_since_confirmation = self.frame_count - self.last_confirmed_frame
                    if frames_since_confirmation > PERSISTENCE_FRAMES:
                        # Lost elephant for too long
                        self.elephant_detected = False
                        self.current_bbox = None
                        self.tracking_active = False
                        self.tracker = None
                        self.status_label.config(text="Status: Elephant lost - Scanning again...", fg='orange')
                        self.detection_info.config(text="Detection Info: Detection timeout - scanning for new elephant")
                        self.log_detection_to_mysql(False)


                # Draw detection box if elephant is detected
                if self.elephant_detected:
                    self.draw_persistent_detection(frame)
                else:
                    # Show scanning message
                    cv2.putText(frame, "Scanning for ELEPHANTS ONLY...", (30, 50),
                              cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)
                    cv2.putText(frame, "High precision mode - No false positives", (30, 90),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                # Convert and display frame
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(rgb_frame)
                imgtk = ImageTk.PhotoImage(image=img)

                self.video_label.configure(image=imgtk)
                self.video_label.image = imgtk  # Keep reference
                
                # Update GUI
                self.root.update_idletasks()
                
                # Control frame rate
                time.sleep(0.033)  # ~30 FPS

            except Exception as e:
                print(f"Video loop error: {e}")
                continue

def main():
    root = tk.Tk()
    app = ElephantDetectorApp(root)
    
    def on_closing():
        app.stop_video()
        try:
            if app.db.is_connected():
                app.cursor.close()
                app.db.close()
        except:
            pass
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()

if __name__ == "__main__":
    main()