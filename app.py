# from flask import Flask, request, send_file, jsonify
# from flask_cors import CORS
# from ultralytics import YOLO
# import os
# import uuid
# import cv2

# app = Flask(__name__)
# CORS(app)

# model = YOLO("yolov8n-fracture.pt")
# os.makedirs("uploads", exist_ok=True)

# @app.route('/predict', methods=['POST'])
# def predict():
#     try:
#         file = request.files['image']
#         filename = f"{uuid.uuid4().hex}_{file.filename}"
#         file_path = os.path.join("uploads", filename)
#         file.save(file_path)

#         # Predict
#         results = model(file_path)
#         result = results[0]

#         # Plot and save the result image with bounding boxes
#         plot_img = result.plot()  # numpy array (BGR)
#         out_path = os.path.join("uploads", f"pred_{filename}")
#         cv2.imwrite(out_path, plot_img)

#         os.remove(file_path)  # optional

#         # Return the image with bounding boxes
#         return send_file(out_path, mimetype='image/jpeg')
#     except Exception as e:
#         print("❌ Error during prediction:", e)
#         return jsonify({"error": str(e)}), 500

# if __name__ == "__main__":
#     app.run(debug=True, port=5050)

from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO
import os
import uuid
import cv2
import base64

app = Flask(__name__)
CORS(app)

model = YOLO("yolov8n-fracture.pt")
os.makedirs("uploads", exist_ok=True)

def img_to_base64(img_path):
    with open(img_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def enhance_image(img):
    # Simple brightness/contrast enhancement
    alpha = 1.0  # contrast
    beta = 20    # brightness
    enhanced = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    return enhanced

@app.route('/predict', methods=['POST'])
def predict():
    try:
        file = request.files['image']
        filename = f"{uuid.uuid4().hex}_{file.filename}"
        file_path = os.path.join("uploads", filename)
        file.save(file_path)

        # 1. Original image (base64)
        original_b64 = img_to_base64(file_path)

        # 2. Enhanced image (brightness/contrast)
        img = cv2.imread(file_path)
        enhanced_img = enhance_image(img)
        enhanced_path = os.path.join("uploads", f"enhanced_{filename}")
        cv2.imwrite(enhanced_path, enhanced_img)
        enhanced_b64 = img_to_base64(enhanced_path)

        # 3. Bounding box image
        results = model(file_path)
        result = results[0]
        plot_img = result.plot()  # numpy array (BGR)
        pred_path = os.path.join("uploads", f"pred_{filename}")
        cv2.imwrite(pred_path, plot_img)
        pred_b64 = img_to_base64(pred_path)

        # Optional: clean up files
        os.remove(file_path)
        os.remove(enhanced_path)
        os.remove(pred_path)

        # Return all images as base64
        return jsonify({
            "original": original_b64,
            "enhanced": enhanced_b64,
            "predicted": pred_b64
        })
    except Exception as e:
        print("❌ Error during prediction:", e)
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5050)