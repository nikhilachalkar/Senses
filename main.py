from fastapi import FastAPI, HTTPException, Request, File, UploadFile
import cv2
import numpy as np
import uvicorn

app = FastAPI()

def certificates(image_bytes: bytes) -> dict:
    try:
        # Decode the image from bytes
        image_array = np.frombuffer(image_bytes, dtype=np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        # Apply smoothing to reduce noise
        smoothed_image = cv2.GaussianBlur(image, (5, 5), 0)

        # Convert to grayscale
        gray = cv2.cvtColor(smoothed_image, cv2.COLOR_BGR2GRAY)

        # Apply sharpening to enhance edges
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
        sharpened_image = cv2.filter2D(gray, -1, kernel)

        # Create a sketch-like effect and make it bolder
        sketch = cv2.GaussianBlur(sharpened_image, (0, 0), 3)
        sketch = cv2.addWeighted(sharpened_image, 2.0, sketch, -1.0, 0)

        # Preprocess the image (e.g., apply blurring or equalization)
        sketch = cv2.GaussianBlur(sketch, (5, 5), 0)

        # Apply threshold
        _, sketch = cv2.threshold(sketch, 128, 255, cv2.THRESH_BINARY)

        # Invert the sketch
        sketch = cv2.bitwise_not(sketch)

        # Find contours
        contours, _ = cv2.findContours(sketch, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

         # Create a dictionary to store contour coordinates
        contour_coordinates = {
            idx: [(float(point[0][0]), float(point[0][1])) for point in contour]
            for idx, contour in enumerate(contours)
        }
        detected_shapes = {}

        # Process each contour
        for idx, contour in enumerate(contours):
           
            
            # Detect shape
            shape = detect_shape(contour)
            detected_shapes[idx] = shape

        return {"contour_coordinates": contour_coordinates, "detected_shapes": detected_shapes}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")


def detect_shape(contour):
    perimeter = cv2.arcLength(contour, True)
    approxCurve = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
    num_corners = len(approxCurve)

    if num_corners == 3:
        return "Triangle"
    elif num_corners == 4:
        x, y, w, h = cv2.boundingRect(contour)
        aspectRatio = float(w) / h
        if abs(aspectRatio - 1) < 0.1:
            return "Square"
        else:
            return "Rectangle"
    elif num_corners == 5:
        return "Pentagon"
    else:
        area = cv2.contourArea(contour)
        if area > 1000:
            perimeter = cv2.arcLength(contour, True)
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            if 0.5 <= circularity <= 1.5:
                if area < 5000:
                    return "Small Circle"
                else:
                    return "Big Circle"

def process_image_and_draw_contours(image_bytes):
    try:
        # Decode the image from bytes
        image_array = np.frombuffer(image_bytes, dtype=np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        # B, G, R channel splitting
        blue, _, _ = cv2.split(image)

        # Detect contours using blue channel and without thresholding
        contours, _ = cv2.findContours(image=blue, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)

        # Draw contours on the original image
        image_contour_blue = image.copy()
        cv2.drawContours(image=image_contour_blue, contours=contours, contourIdx=-1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)

        # Create a dictionary to store contour coordinates
        contour_coordinates = {
            idx: [(float(point[0][0]), float(point[0][1])) for point in contour]
            for idx, contour in enumerate(contours)
        }

        
        detected_shapes = {}

        # Process each contour
        for idx, contour in enumerate(contours):
           
            
            # Detect shape
            shape = detect_shape(contour)
            detected_shapes[idx] = shape

        return {"contour_coordinates": contour_coordinates, "detected_shapes": detected_shapes}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

@app.post("/predict/")
async def predict(image: UploadFile = File(...)):
    try:
        # Read the uploaded image bytes
        image_bytes = await image.read()

        # Process the image and detect contours and shapes
        contour_data = process_image_and_draw_contours(image_bytes)

        # Return JSON response with contour data
        return contour_data

    except Exception as e:
        return {"error": str(e)}

@app.post("/certificates/")
async def predict(image: UploadFile = File(...)):
    try:
        # Read the uploaded image bytes
        image_bytes1 = await image.read()

        # Process the image and detect contours and shapes
         
        result =certificates(image_bytes1)
        # Return JSON response with contour data
        return result

    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
