from flask import Flask, request, jsonify
import cv2
import numpy as np
import os

app = Flask(__name__)

@app.route('/segment', methods=['POST'])
def segment():
    try:
        # 1. Receive the file from MIT App Inventor
        file = request.files['file']
        img_bytes = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(img_bytes, cv2.IMREAD_GRAYSCALE)

        # 2. Your Segmentation Logic (Simplified Example)
        # Using a threshold to find the 'bright' tumor area
        _, mask = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY)

        # 3. Calculate Area and Location
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        area = 0
        location = "Not Found"

        if contours:
            # Get the largest object (the tumor)
            largest = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest)
            
            # Calculate the Center (Centroid)
            M = cv2.moments(largest)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                location = f"X:{cx}, Y:{cy}"

        # 4. Return ONLY text data as JSON
        return jsonify({
            "status": "success",
            "area": f"{area} px",
            "location": location
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

if __name__ == '__main__':
    app.run()
