from fastapi import FastAPI, HTTPException, Request, File, UploadFile
import cv2

import numpy as np


import uvicorn


app = FastAPI()

def process_image_and_draw_contours(image_bytes):
    
    try:# Add your existing image processing code here, handling bytes:
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

        # Display the results
        # cv2_imshow(image_contour_blue)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

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