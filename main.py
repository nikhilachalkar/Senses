from fastapi import FastAPI, HTTPException, File, UploadFile
import cv2
import numpy as np
import uvicorn

app = FastAPI()


def detect(image):
        # Convert the image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Threshold the grayscale image
        _, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)

        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        shape_and_rectangles = {}
        for contour in contours:
            shape, bounding_rect = detect_shape(contour)
            if shape is not None:
                shape_and_rectangles[shape] = bounding_rect

        return shape_and_rectangles

def detect_shape(contour):
    # Approximate the contour to simplify its shape
    perimeter = cv2.arcLength(contour, True)
    approxCurve = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
    
    # Get the number of corners
    numCorners = len(approxCurve)
    
    # Determine the type of object based on the number of corners
    if numCorners == 3:
        return "Triangle" , cv2.boundingRect(contour)
    elif numCorners == 4:
        # Check if it's a square or rectangle
        x, y, w, h = cv2.boundingRect(contour)
        aspectRatio = float(w) / h
        if abs(aspectRatio - 1) < 0.1:
            return "Square" , cv2.boundingRect(contour)
        else:
            return "Rectangle" , cv2.boundingRect(contour)
    elif numCorners == 5:
        return "Pentagon" , cv2.boundingRect(contour)
    else:
        # Circle detection
        area = cv2.contourArea(contour)
        if area > 1000:  # Adjust the area threshold as needed
            perimeter = cv2.arcLength(contour, True)
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            if 0.5 <= circularity <= 1.5:
                if area < 5000:
                    return "Small Circle" , cv2.boundingRect(contour)
                else:
                    return "Big Circle" , cv2.boundingRect(contour)
    return None, None


def process_image_and_draw_contours(image_bytes):
    try:
        image_array = np.frombuffer(image_bytes, dtype=np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)


        # B, G, R channel splitting
        blue, _, _ = cv2.split(image)

        # Detect contours using blue channel and without thresholding
        contours, _ = cv2.findContours(image=blue, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)

        # Draw contours on the original image
        image_contour_blue = image.copy()
        cv2.drawContours(image=image_contour_blue, contours=contours, contourIdx=-1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)



 # Create a dictionary to store contour coordinates
        contour_coordinates = {
            idx: [(float(point[0][0]), float(point[0][1])) for point in contour]
            for idx, contour in enumerate(contours)
        }



        # Store contour coordinates and shapes
        contour_data = []
        shape1=[]

        shape1=detect(image)
        
            
        contour_data.append({"coordinates": contour_coordinates, "shape": shape1})

        return contour_data

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

@app.post("/predict/")
async def predict(image: UploadFile = File(...)):
    try:
        # Read the uploaded image bytes
        image_bytes = await image.read()

        # Preprocess the image and detect contours
        contour_data = process_image_and_draw_contours(image_bytes)

        # Return JSON response with contour data
        return {"contours": contour_data}

    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    # Ensure the correct deployment configuration
    uvicorn.run(app, host="0.0.0.0", port=80)
