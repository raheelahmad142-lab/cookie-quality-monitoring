
import gradio as gr
import cv2
import numpy as np
from ultralytics import YOLO

model = YOLO("best.pt")

def compute_tray_score(detections, expected_count=6):
    score = 100
    issues = []
    total = len(detections)
    burned = detections.count("Burned")
    broken = detections.count("Broken")
    if total != expected_count:
        score -= abs(total - expected_count) * 10
        issues.append(f"Wrong count: found {total}, expected {expected_count}")
    if burned > 0:
        score -= burned * 15
        issues.append(f"{burned} burned cookie(s)")
    if broken > 0:
        score -= broken * 10
        issues.append(f"{broken} broken cookie(s)")
    return max(0, score), issues

def analyze_tray(image, expected_count):
    img_array = np.array(image)
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    temp_path = "temp_tray.jpg"
    cv2.imwrite(temp_path, img_bgr)
    image_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    results = model.predict(source=temp_path, conf=0.25, verbose=False)
    detections = []
    color_issues = []
    for r in results:
        for i, box in enumerate(r.boxes):
            cls_id = int(box.cls[0])
            label = model.names[cls_id]
            detections.append(label)
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            crop = image_hsv[y1:y2, x1:x2]
            brightness = crop[:,:,2].mean()
            if brightness < 80:
                color_issues.append(f"Cookie {i+1} overbaked (brightness {brightness:.0f})")
            elif brightness > 180:
                color_issues.append(f"Cookie {i+1} underbaked (brightness {brightness:.0f})")
    score, issues = compute_tray_score(detections, int(expected_count))
    all_issues = issues + color_issues
    status = "✅ PASS" if score >= 80 else "⚠️ WARNING" if score >= 50 else "❌ FAIL"
    result_img = results[0].plot()
    result_img_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
    report = f"Tray Score: {score}/100 {status}\n"
    report += f"Detected: {detections}\n\n"
    if all_issues:
        report += "Issues Found:\n"
        for issue in all_issues:
            report += f"  • {issue}\n"
    else:
        report += "No issues detected ✅"
    return result_img_rgb, report

demo = gr.Interface(
    fn=analyze_tray,
    inputs=[
        gr.Image(type="pil", label="Upload Tray Image"),
        gr.Slider(1, 12, value=6, step=1, label="Expected Cookie Count")
    ],
    outputs=[
        gr.Image(label="Detection Result"),
        gr.Textbox(label="Quality Report", lines=10)
    ],
    title="🍪 Cookie Quality Monitoring System",
    description="Upload a cookie tray image to get quality score and defect analysis."
)

demo.launch()
