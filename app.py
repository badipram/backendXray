from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import uuid
import cv2
import base64
import logging

logging.basicConfig(level=logging.INFO)

app = Flask(__name__)
CORS(app)

MODEL_PATH = "yolov8n-fracture.pt"
model = None  # Lazy load

os.makedirs("uploads", exist_ok=True)

def ensure_model():
    if not os.path.exists(MODEL_PATH):
        print("📥 Downloading model...")
        import gdown
        gdown.download(
            "https://drive.google.com/uc?id=1cPTgDraDSiKCugxAPP5XEPUCFUkvAUyd",
            MODEL_PATH,
            quiet=False
        )

def img_to_base64(img_path):
    with open(img_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def enhance_image(img):
    alpha = 1.0  # contrast
    beta = 20    # brightness
    return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        global model
        ensure_model()
        if model is None:
            print("⚙️ Loading model...")
            from ultralytics import YOLO
            model = YOLO(MODEL_PATH)

        file = request.files['image']
        filename = f"{uuid.uuid4().hex}_{file.filename}"
        file_path = os.path.join("uploads", filename)
        file.save(file_path)

        original_b64 = img_to_base64(file_path)

        img = cv2.imread(file_path)
        enhanced_img = enhance_image(img)
        enhanced_path = os.path.join("uploads", f"enhanced_{filename}")
        cv2.imwrite(enhanced_path, enhanced_img)
        enhanced_b64 = img_to_base64(enhanced_path)

        results = model(file_path)
        result = results[0]
        plot_img = result.plot()
        pred_path = os.path.join("uploads", f"pred_{filename}")
        cv2.imwrite(pred_path, plot_img)
        pred_b64 = img_to_base64(pred_path)

        detection_info = []
        for box in result.boxes:
            detection_info.append({
                "confidence": float(box.conf[0]),
                "bbox": [float(x) for x in box.xyxy[0].tolist()]
            })

        os.remove(file_path)
        os.remove(enhanced_path)
        os.remove(pred_path)

        return jsonify({
            "original": original_b64,
            "enhanced": enhanced_b64,
            "predicted": pred_b64,
            "detection_info": detection_info
        })

    except Exception as e:
        # print("❌ Error during prediction:", e)
        logging.exception("❌ Error during prediction:")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5050)