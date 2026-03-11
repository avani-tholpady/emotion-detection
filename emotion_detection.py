import cv2
from deepface import DeepFace
import time
import csv
import os
import threading
import numpy as np


EMOTION_STYLES = {
    'happy':     {'color': (0, 255, 100),   'icon': '😊 Happy'},
    'sad':       {'color': (255, 100, 50),   'icon': '😢 Sad'},
    'angry':     {'color': (0, 0, 255),      'icon': '😠 Angry'},
    'surprise':  {'color': (0, 200, 255),    'icon': '😲 Surprise'},
    'fear':      {'color': (180, 0, 255),    'icon': '😨 Fear'},
    'disgust':   {'color': (0, 180, 80),     'icon': '🤢 Disgust'},
    'neutral':   {'color': (200, 200, 200),  'icon': '😐 Neutral'},
}

SIDEBAR_WIDTH = 260
CONFIDENCE_THRESHOLD = 40.0   


os.makedirs('logs', exist_ok=True)
log_file = 'logs/detection_log.csv'
log_exists = os.path.isfile(log_file)
csv_file = open(log_file, mode='a', newline='')
csv_writer = csv.writer(csv_file)
if not log_exists:
    csv_writer.writerow(['Timestamp', 'Emotion', 'Emotion Confidence'])


cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
print("Press 'q' to quit")


latest_results = []
lock = threading.Lock()
analyzing = False


emotion_timer = {'emotion': None, 'start': time.time()}
session_start = time.time()
total_detections = 0
emotion_counts = {e: 0 for e in EMOTION_STYLES}


def get_emotion_color(emotion):
    return EMOTION_STYLES.get(emotion, {}).get('color', (200, 200, 200))

def get_emotion_icon(emotion):
    return EMOTION_STYLES.get(emotion, {}).get('icon', emotion)

def draw_label_with_bg(img, text, pos, font, font_scale, text_color, bg_color, thickness=1, padding=5):
    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = pos
    cv2.rectangle(img,
                  (x - padding, y - text_h - padding),
                  (x + text_w + padding, y + baseline + padding),
                  bg_color, -1)
    cv2.putText(img, text, (x, y), font, font_scale, text_color, thickness)


def draw_sidebar(canvas, results, fps):
    h, w = canvas.shape[:2]

    
    overlay = canvas.copy()
    cv2.rectangle(overlay, (w - SIDEBAR_WIDTH, 0), (w, h), (18, 18, 28), -1)
    cv2.addWeighted(overlay, 0.85, canvas, 0.15, 0, canvas)

    
    cv2.line(canvas, (w - SIDEBAR_WIDTH, 0), (w - SIDEBAR_WIDTH, h), (60, 60, 80), 1)

    font = cv2.FONT_HERSHEY_SIMPLEX
    sx = w - SIDEBAR_WIDTH + 12  # sidebar x start

    
    cv2.putText(canvas, 'EMOTION DETECTOR', (sx, 28), font, 0.48, (180, 180, 255), 1)
    cv2.line(canvas, (sx, 36), (w - 12, 36), (60, 60, 100), 1)

    
    cv2.putText(canvas, f'FPS: {int(fps)}', (sx, 58), font, 0.5, (120, 255, 120), 1)

    
    elapsed = int(time.time() - session_start)
    mins, secs = divmod(elapsed, 60)
    cv2.putText(canvas, f'Session: {mins:02d}:{secs:02d}', (sx, 80), font, 0.45, (160, 160, 160), 1)

    
    face_count = len(results)
    cv2.putText(canvas, f'Faces: {face_count}', (sx, 102), font, 0.45, (160, 160, 160), 1)

    cv2.line(canvas, (sx, 112), (w - 12, 112), (60, 60, 100), 1)

    
    y_cursor = 128
    for i, (x, yf, fw, fh, emotion, confidence, all_emotions) in enumerate(results):
        color = get_emotion_color(emotion)
        label = get_emotion_icon(emotion)

        
        cv2.putText(canvas, f'Face {i+1}', (sx, y_cursor), font, 0.45, (220, 220, 220), 1)
        y_cursor += 18

        
        cv2.putText(canvas, label, (sx, y_cursor), font, 0.5, color, 1)
        y_cursor += 18

        
        bar_w = SIDEBAR_WIDTH - 30
        bar_h = 8
        filled = int((confidence / 100.0) * bar_w)
        cv2.rectangle(canvas, (sx, y_cursor), (sx + bar_w, y_cursor + bar_h), (50, 50, 60), -1)
        cv2.rectangle(canvas, (sx, y_cursor), (sx + filled, y_cursor + bar_h), color, -1)
        cv2.putText(canvas, f'{confidence:.1f}%', (sx + bar_w + 4, y_cursor + 8),
                    font, 0.38, color, 1)
        y_cursor += 18

        
        for emo, val in sorted(all_emotions.items(), key=lambda x: -x[1])[:4]:
            emo_color = get_emotion_color(emo)
            mini_filled = int((val / 100.0) * (bar_w - 40))
            cv2.putText(canvas, emo[:3], (sx, y_cursor + 7), font, 0.32, (140, 140, 140), 1)
            cv2.rectangle(canvas, (sx + 28, y_cursor), (sx + 28 + (bar_w - 40), y_cursor + 6), (40, 40, 50), -1)
            cv2.rectangle(canvas, (sx + 28, y_cursor), (sx + 28 + mini_filled, y_cursor + 6), emo_color, -1)
            y_cursor += 10

        y_cursor += 8
        cv2.line(canvas, (sx, y_cursor), (w - 12, y_cursor), (40, 40, 55), 1)
        y_cursor += 8

        if y_cursor > h - 60:
            break

    
    timer_y = h - 40
    cv2.line(canvas, (sx, timer_y - 10), (w - 12, timer_y - 10), (60, 60, 100), 1)

    if emotion_timer['emotion']:
        held = int(time.time() - emotion_timer['start'])
        timer_color = get_emotion_color(emotion_timer['emotion'])
        cv2.putText(canvas, f"Holding: {emotion_timer['emotion']}", (sx, timer_y),
                    font, 0.42, timer_color, 1)
        cv2.putText(canvas, f'{held}s', (sx, timer_y + 18), font, 0.45, timer_color, 1)


def analyze_frame(small_frame):
    global analyzing, latest_results, total_detections, emotion_timer
    try:
        results = DeepFace.analyze(
            small_frame,
            actions=['emotion'],
            detector_backend='opencv',
            enforce_detection=False
        )
        if isinstance(results, dict):
            results = [results]

        detections = []
        for face_data in results:
            region = face_data['region']
            x, y, w, h = [int(c * 2) for c in (region['x'], region['y'], region['w'], region['h'])]
            emotion = face_data['dominant_emotion']
            confidence = face_data['emotion'][emotion]
            all_emotions = face_data['emotion']

            if confidence < CONFIDENCE_THRESHOLD:
                continue

            
            if emotion_timer['emotion'] != emotion:
                emotion_timer = {'emotion': emotion, 'start': time.time()}

            
            timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
            csv_writer.writerow([timestamp, emotion, f'{confidence:.1f}'])
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
            total_detections += 1

            detections.append((x, y, w, h, emotion, confidence, all_emotions))

        with lock:
            latest_results = detections

    except Exception as e:
        print("Analysis error:", e)
    finally:
        analyzing = False


prev_time = time.time()
frame_count = 0
ANALYZE_EVERY_N_FRAMES = 5

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    
    h, w = frame.shape[:2]
    canvas = np.zeros((h, w + SIDEBAR_WIDTH, 3), dtype=np.uint8)
    canvas[:h, :w] = frame

    
    if frame_count % ANALYZE_EVERY_N_FRAMES == 0 and not analyzing:
        analyzing = True
        small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        thread = threading.Thread(target=analyze_frame, args=(small_frame,), daemon=True)
        thread.start()

    
    with lock:
        current_results = list(latest_results)

    for (x, yf, fw, fh, emotion, confidence, all_emotions) in current_results:
        color = get_emotion_color(emotion)

        
        cv2.rectangle(canvas, (x, yf), (x + fw, yf + fh), color, 2)
        cv2.rectangle(canvas, (x - 1, yf - 1), (x + fw + 1, yf + fh + 1), (30, 30, 30), 1)

        
        label = f'{get_emotion_icon(emotion)}  {confidence:.1f}%'
        draw_label_with_bg(canvas, label, (x, max(yf - 10, 20)),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, (20, 20, 30), 1)

    
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time

    
    draw_sidebar(canvas, current_results, fps)

    cv2.imshow('Emotion Detection', canvas)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


elapsed = int(time.time() - session_start)
mins, secs = divmod(elapsed, 60)
print("\n" + "="*40)
print("       SESSION SUMMARY")
print("="*40)
print(f"  Duration       : {mins:02d}:{secs:02d}")
print(f"  Total Detections: {total_detections}")
if total_detections > 0:
    top = max(emotion_counts, key=emotion_counts.get)
    print(f"  Dominant Emotion: {top} ({emotion_counts[top]} times)")
    print("\n  Emotion Breakdown:")
    for emo, count in sorted(emotion_counts.items(), key=lambda x: -x[1]):
        if count > 0:
            pct = (count / total_detections) * 100
            print(f"    {emo:<12} {count:>4}x  ({pct:.1f}%)")
print("="*40)
print(f"  Log saved to: {log_file}")
print("="*40 + "\n")


cap.release()
cv2.destroyAllWindows()
csv_file.close()
