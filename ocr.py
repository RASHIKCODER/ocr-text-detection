 (cd "$(git rev-parse --show-toplevel)" && git apply --3way <<'EOF' 
diff --git a/app.py b/app.py
new file mode 100644
index 0000000000000000000000000000000000000000..5a97f74bbf84bd0e54d6fdb53cc22c8d6ce62841
--- /dev/null
+++ b/app.py
@@ -0,0 +1,178 @@
+import cv2
+import easyocr
+import paho.mqtt.client as mqtt
+import time
+from flask import Flask, Response, render_template_string
+
+# ---------------- MQTT CONFIG ----------------
+
+BROKER = "mqtt.sar-analytic.in"
+PORT = 1883
+USERNAME = "mqtt"
+PASSWORD = "mqtt"
+TOPIC = "ocr/detections"
+
+client = mqtt.Client()
+client.username_pw_set(USERNAME, PASSWORD)
+client.connect(BROKER, PORT, 60)
+client.loop_start()
+
+# ---------------- OCR SETUP ----------------
+
+reader = easyocr.Reader(['en'], gpu=False)
+cap = cv2.VideoCapture(0)
+
+last_send_time = time.time()
+detected_texts_buffer = set()
+
+latest_text = "Waiting for detection..."
+
+# ---------------- FLASK APP ----------------
+
+app = Flask(__name__)
+
+HTML = """
+<!DOCTYPE html>
+<html>
+<head>
+<title>OCR Dashboard</title>
+
+<style>
+
+body{
+background:#0f172a;
+color:white;
+font-family:Arial;
+text-align:center;
+}
+
+h1{
+color:#38bdf8;
+}
+
+.card{
+background:#1e293b;
+padding:20px;
+margin:20px auto;
+width:500px;
+border-radius:10px;
+}
+
+img{
+width:500px;
+border-radius:10px;
+}
+
+</style>
+
+</head>
+
+<body>
+
+<h1>AI OCR Monitoring Dashboard</h1>
+
+<div class="card">
+<h2>Detected Text</h2>
+<h3 id="text">{{text}}</h3>
+</div>
+
+<div class="card">
+<h2>Live Camera</h2>
+<img src="/video_feed">
+</div>
+
+<script>
+
+setInterval(()=>{
+fetch('/text')
+.then(res=>res.text())
+.then(data=>{
+document.getElementById("text").innerHTML=data
+})
+},1000)
+
+</script>
+
+</body>
+</html>
+"""
+
+
+@app.route('/')
+def index():
+    return render_template_string(HTML, text=latest_text)
+
+
+@app.route('/text')
+def text():
+    return latest_text
+
+
+# ---------------- VIDEO STREAM ----------------
+
+def generate():
+    global latest_text
+    global last_send_time
+
+    count = 0
+    results = []
+
+    while True:
+
+        ret, frame = cap.read()
+        if not ret:
+            break
+
+        if count % 10 == 0:
+            results = reader.readtext(frame)
+
+        for res in results:
+
+            bbox, text, score = res
+
+            if score > 0.25:
+
+                detected_texts_buffer.add(text)
+                latest_text = text
+
+                top_left = tuple(map(int, bbox[0]))
+                bottom_right = tuple(map(int, bbox[2]))
+
+                cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)
+                cv2.putText(frame, text, (top_left[0], top_left[1] - 10),
+                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
+
+        current_time = time.time()
+
+        if current_time - last_send_time >= 10:
+
+            if detected_texts_buffer:
+
+                message = ", ".join(detected_texts_buffer)
+                client.publish(TOPIC, message)
+                print("MQTT Sent:", message)
+
+                detected_texts_buffer.clear()
+
+            last_send_time = current_time
+
+        _, buffer = cv2.imencode('.jpg', frame)
+
+        frame = buffer.tobytes()
+
+        yield (b'--frame\r\n'
+               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
+
+        count += 1
+
+
+@app.route('/video_feed')
+def video_feed():
+    return Response(generate(),
+                    mimetype='multipart/x-mixed-replace; boundary=frame')
+
+
+# ---------------- MAIN ----------------
+
+if __name__ == '__main__':
+    app.run(host='0.0.0.0', port=5000)
 
EOF
)
