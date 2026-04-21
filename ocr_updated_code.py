import cv2
import easyocr
import paho.mqtt.client as mqtt
import time
import re
import numpy as np
from collections import Counter

# ================================================================
#  CONFIGURATION
# ================================================================
BROKER               = "mqtt.sar-analytic.in"
PORT                 = 1883
USERNAME             = "mqtt"
PASSWORD             = "mqtt"
TOPIC                = "ocr/detections"

CONFIDENCE_THRESHOLD = 0.65
MIN_TEXT_LENGTH      = 3
FRAME_SKIP           = 5
CONFIRM_FRAMES       = 3
MQTT_INTERVAL        = 10

WINDOW_NAME          = "OCR Camera"

# ================================================================
#  MQTT SETUP
# ================================================================
mqtt_client = mqtt.Client()
mqtt_client.username_pw_set(USERNAME, PASSWORD)
mqtt_client.connect(BROKER, PORT, 60)
mqtt_client.loop_start()

# ================================================================
#  OCR READER
# ================================================================
reader = easyocr.Reader(['en'], gpu=False)

# ================================================================
#  CUSTOM ROI SELECTOR (mouse callback based)
# ================================================================
roi             = None
drawing         = False   # True while left button held
roi_start       = (-1, -1)
roi_temp        = None    # Live preview rectangle while dragging
roi_mode        = False   # True = ROI selection mode active

def mouse_callback(event, x, y, flags, param):
    global roi, drawing, roi_start, roi_temp, roi_mode

    if not roi_mode:
        return

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing   = True
        roi_start = (x, y)
        roi_temp  = None

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            roi_temp = (roi_start[0], roi_start[1], x, y)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing  = False
        x1 = min(roi_start[0], x)
        y1 = min(roi_start[1], y)
        x2 = max(roi_start[0], x)
        y2 = max(roi_start[1], y)

        if (x2 - x1) > 10 and (y2 - y1) > 10:   # ignore accidental tiny clicks
            roi      = (x1, y1, x2, y2)
            roi_temp = None
            roi_mode = False
            print(f"[ROI] Selected: {roi}")
        else:
            roi_temp = None
            roi_mode = False
            print("[ROI] Too small — try again (press R)")

# ================================================================
#  HELPER FUNCTIONS
# ================================================================

def preprocess_roi(img):
    h, w = img.shape[:2]
    if w < 300 or h < 100:
        scale = max(300 / w, 100 / h)
        img = cv2.resize(img, (int(w * scale), int(h * scale)),
                         interpolation=cv2.INTER_CUBIC)
    gray   = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray   = cv2.fastNlMeansDenoising(gray, h=10)
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    gray   = cv2.filter2D(gray, -1, kernel)
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 31, 10)
    return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)


def is_valid_text(text: str) -> bool:
    text = text.strip()
    if len(text) < MIN_TEXT_LENGTH:
        return False
    if not re.search(r'[A-Za-z0-9]', text):
        return False
    alnum_ratio = sum(c.isalnum() for c in text) / len(text)
    if alnum_ratio < 0.4:
        return False
    if re.fullmatch(r'(.)\1{3,}', text):
        return False
    if len(text) > 4 and not re.search(r'[AEIOUaeiou]', text) \
            and re.fullmatch(r'[^0-9\s]+', text):
        return False
    return True


def normalize_text(text: str) -> str:
    return re.sub(r'\s+', ' ', text.strip())


# ================================================================
#  TEMPORAL CONFIRMATION
# ================================================================
text_frame_counter: Counter = Counter()
confirmed_texts: set        = set()

def update_confirmation(current_results):
    global text_frame_counter, confirmed_texts
    for key in list(text_frame_counter.keys()):
        text_frame_counter[key] -= 1
        if text_frame_counter[key] <= 0:
            del text_frame_counter[key]
    for text in current_results:
        text_frame_counter[text] += 1
        if text_frame_counter[text] >= CONFIRM_FRAMES:
            confirmed_texts.add(text)

# ================================================================
#  CAMERA SETUP
# ================================================================
cap = cv2.VideoCapture(0)

cv2.namedWindow(WINDOW_NAME)
cv2.setMouseCallback(WINDOW_NAME, mouse_callback)   # ← attach once, stays active

frame_count    = 0
last_send_time = time.time()
last_ocr_boxes = []

print("Controls:")
print("  R → ROI select mode ON  |  C → clear ROI  |  Q → quit")
print("  (Jab ROI mode ON ho: click + drag karke region select karo)")

# ================================================================
#  MAIN LOOP
# ================================================================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    display     = frame.copy()
    frame_count += 1

    # ── ROI selection mode overlay ────────────────────────────
    if roi_mode:
        overlay = display.copy()
        cv2.rectangle(overlay, (0, 0), (display.shape[1], display.shape[0]),
                      (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.35, display, 0.65, 0, display)
        cv2.putText(display, "DRAG TO SELECT ROI  |  Press C to cancel",
                    (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)

        # Live preview while dragging
        if roi_temp:
            x1t, y1t, x2t, y2t = roi_temp
            cv2.rectangle(display, (x1t, y1t), (x2t, y2t), (0, 200, 255), 2)

    # ── Draw confirmed ROI box ────────────────────────────────
    if roi and not roi_mode:
        x1, y1, x2, y2 = roi
        cv2.rectangle(display, (x1, y1), (x2, y2), (255, 165, 0), 2)
        cv2.putText(display, "ROI", (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 165, 0), 2)
        roi_frame = frame[y1:y2, x1:x2]
    else:
        roi_frame = frame

    # ── OCR every FRAME_SKIP frames ───────────────────────────
    if frame_count % FRAME_SKIP == 0 and not roi_mode:
        processed   = preprocess_roi(roi_frame)
        raw_results = reader.readtext(processed)

        current_valid_texts = []
        last_ocr_boxes      = []

        for bbox, text, score in raw_results:
            if score < CONFIDENCE_THRESHOLD:
                continue
            text = normalize_text(text)
            if not is_valid_text(text):
                continue
            current_valid_texts.append(text)

            if roi:
                bbox = [[pt[0] + x1, pt[1] + y1] for pt in bbox]

            last_ocr_boxes.append((bbox, text, score))

        update_confirmation(current_valid_texts)

    # ── Draw OCR boxes ────────────────────────────────────────
    if not roi_mode:
        for bbox, text, score in last_ocr_boxes:
            tl    = tuple(map(int, bbox[0]))
            br    = tuple(map(int, bbox[2]))
            color = (0, 255, 0) if text in confirmed_texts else (0, 200, 255)
            cv2.rectangle(display, tl, br, color, 2)
            cv2.putText(display, f"{text} ({score:.2f})",
                        (tl[0], tl[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

    # ── Status bar ────────────────────────────────────────────
    roi_label  = f"ROI:{roi}" if roi else "ROI:Full Frame"
    status_msg = f"{roi_label}  Confirmed:{len(confirmed_texts)}  R=Select C=Clear Q=Quit"
    cv2.putText(display, status_msg, (10, display.shape[0] - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    # ── MQTT publish ──────────────────────────────────────────
    if time.time() - last_send_time > MQTT_INTERVAL:
        if confirmed_texts:
            msg = ", ".join(sorted(confirmed_texts))
            mqtt_client.publish(TOPIC, msg)
            print(f"[MQTT] Sent → {msg}")
            confirmed_texts.clear()
        else:
            print("[MQTT] Nothing confirmed yet.")
        last_send_time = time.time()

    cv2.imshow(WINDOW_NAME, display)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('r'):
        roi_mode = True
        drawing  = False
        roi_temp = None
        print("[ROI] Mode ON — click and drag to select region")

    elif key == ord('c'):
        roi      = None
        roi_mode = False
        roi_temp = None
        last_ocr_boxes = []
        confirmed_texts.clear()
        text_frame_counter.clear()
        print("[ROI] Cleared")

    elif key == ord('q'):
        break

# ================================================================
#  CLEANUP
# ================================================================
cap.release()
cv2.destroyAllWindows()
mqtt_client.loop_stop()
mqtt_client.disconnect()
print("Done.")
