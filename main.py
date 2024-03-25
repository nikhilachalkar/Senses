from fastapi import FastAPI, HTTPException, Request, File, UploadFile
import cv2
import numpy as np
import uvicorn

app = FastAPI()

def detect_shape(contour):
    # Approximate the contour to simplify its shape
    perimeter = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)
    
    # Determine the shape based on the number of vertices
    num_vertices = len(approx)
    if num_vertices == 4:
        return "Rectangle"
    elif num_vertices >= 6:
        return "Circle (Big)"
    else:
        return "Circle (Small)"

def process_image_and_draw_contours(image_bytes):
    try:
        image_array = np.frombuffer(image_bytes, dtype=np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        # Convert the image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Threshold the grayscale image
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Store contour coordinates and shapes
        contour_data = {}
        for idx, contour in enumerate(contours):
            shape = detect_shape(contour)
            coordinates = [(float(point[0][0]), float(point[0][1])) for point in contour]
            contour_data[idx] = {"contour_coordinates": coordinates, "shape": shape}

        return {"contour_data": contour_data}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

@app.post("/predict/")
async def predict(image: UploadFile = File(...)):
    try:
        # Read the uploaded image bytes
        image_bytes = await image.read()

        # Preprocess the image and detect contours
        result = process_image_and_draw_contours(image_bytes)

        # Return JSON response with contour data
        return result

    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="https://senses.onrender.com", port=80)
