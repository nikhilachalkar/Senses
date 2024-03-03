from fastapi import FastAPI, HTTPException, Request, File, UploadFile
import cv2

import numpy as np


import uvicorn


app = FastAPI()

def process_image_and_draw_contours(image_bytes):
    
    try:# Add your existing image processing code here, handling bytes:
        image_array = np.frombuffer(image_bytes, dtype=np.uint8)
        image_mat = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    
    # Apply smoothing to reduce noise
        image_mat = cv2.GaussianBlur(image_mat, (5, 5), 0)

    # Apply sharpening to enhance edges
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
        image_mat = cv2.filter2D(image_mat, -1, kernel)
        
    # Convert to grayscale
        gray = cv2.cvtColor(image_mat, cv2.COLOR_BGR2GRAY)

    # Create a sketch-like effect and make it bolder
        sketch = cv2.GaussianBlur(gray, (0, 0), 5)
        sketch = cv2.addWeighted(gray, 2.0, sketch, -1.0, 0)
        


    # Preprocess the image (e.g., apply blurring or equalization)
        sketch = cv2.GaussianBlur(sketch, (5, 5), 0)

    # Apply threshold
        _, sketch = cv2.threshold(sketch, 100, 255, cv2.THRESH_BINARY)

        sketch = cv2.dilate(sketch, None, iterations=2)  # Dilate to make contours thicker


    # Find contours
       #contours, _ = cv2.findContours(sketch, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        #contours, _ = cv2.findContours(sketch, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contours, _ = cv2.findContours(sketch, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


    # Draw contours in red
    # contour_image = np.zeros_like(image_mat)
    # cv2.drawContours(contour_image, contours, -1, (0, 0, 255), 1)

    # Create a dictionary to store contour coordinates
        contour_coordinates = {
                idx: [(int(point[0][0]), int(point[0][1])) for point in contour]
                for idx, contour in enumerate(contours)
            }

        return {"contour_coordinates": contour_coordinates}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")
    
@app.post("/predict/")
async def predict(image: UploadFile = File(...)):
    try:
        # Read the uploaded image bytes
        image_bytes = await image.read()

        # Preprocess the image
        contour_coordinates = process_image_and_draw_contours(image_bytes)

        # Return JSON response with contour coordinates
        return contour_coordinates

    except Exception as e:
        return {"error": str(e)}
    

if __name__ == "__main__":
    uvicorn.run(app, host="https://senses.onrender.com", port=80)