from io import BytesIO
import random
import os
import asyncio
import time
from datetime import datetime
import csv

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

from fastapi import FastAPI, Request, BackgroundTasks, Form
from fastapi.responses import StreamingResponse, JSONResponse
from starlette.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from pydantic import BaseModel

from Model.image_final import ImageModel

img_model = ImageModel('Model/better_classification.h5')
ratio = [1,1,1]
progress = {"value": 0} 

class ObjectGenerator:
    def __init__(self, image_folder):
        self.image_folder = image_folder
        self.image_files = self._get_image_files()
        self.current_index = -1  # Initialize to -1 to start from the first image

    ############################################
    #   FUNCTIONS FOR MEDIA HANDLING ENDPOINTS
    ############################################

    def _get_image_files(self):
        image_files = os.listdir(self.image_folder)
        image_files.sort()  # Sort files to ensure they are in order
        return image_files

    def _get_next_image_path(self):
        self.current_index += 1
        if self.current_index < len(self.image_files):
            return os.path.join(self.image_folder, self.image_files[self.current_index])
        else:
            return None  # No more images available

    def generate_image_input(self, image_path):
        if image_path:
            img = Image.open(image_path).convert("RGB")
        else:
            raise ValueError("image_path must be provided")

        buf = BytesIO()
        img.save(buf, format='PNG')
        buf.seek(0)
        return buf

    def generate_image_output(self, image_path):
        if not image_path:
            raise ValueError("image_path must be provided")

        # Call show_lime to process the image and get the LIME output
        boundary_img = img_model.show_lime(image_path)

        if boundary_img is None:
            raise ValueError("Failed to generate LIME output for the image.")
        
        return boundary_img

    def get_img_caption(self):
        return "caption etc."
    
    def get_text_explanability(self):
        return "text explanability"
    
    def get_model_answer(self, image_path):
        # Return a dummy answer provided by the model
        return img_model.predict_model(image_path)

    def get_actual_answer(self):
        # Return a random number between 1 and 6
        return "Dummy_actual_ans"

#######################################################
#                         APP
#######################################################
# Initialize FastAPI app
app = FastAPI()

# Templates directory for Jinja2Templates
templates = Jinja2Templates(directory="templates")

# CORS Middleware to allow frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5500"],  # Update with your frontend URL
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Directory for saving survey data
DATA_DIR = 'data_collected'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Initialize ImageModel and ObjectGenerator
img_model = ImageModel('Model/better_classification.h5')
generator = ObjectGenerator(image_folder='./images')

# Global variables for managing responses and image paths
responses_cache = []
current_image_path = None
last_image_path= None

# Class to handle survey data using Pydantic BaseModel
class SurveyData(BaseModel):
    actual_answer: str
    model_answer: str
    user_answer: int
    image_file: str

# Function to fetch the next image path
def get_next_image_path():
    global current_image_path
    current_image_path = generator._get_next_image_path()
    return current_image_path

# Endpoint to start survey
@app.get("/survey")
async def survey(request: Request):
    return templates.TemplateResponse("survey.html", {"request": request})

# Endpoint to fetch input image
@app.get("/input_image")
def input_image():
    next_image_path = get_next_image_path()  # Update the image path here
    try:
        buf = generator.generate_image_input(next_image_path)
        return StreamingResponse(buf, media_type="image/png")
    except Exception as e:
        return {"error": str(e)}

# Endpoint to fetch model answer
@app.get("/model_answer")
def model_answer():
    answer = generator.get_model_answer(current_image_path)
    return JSONResponse({"prediction": answer})

# Endpoint to fetch image caption
@app.get("/caption")
def get_caption():
    caption = generator.get_img_caption()
    return {"image_path": current_image_path, "caption": caption}

@app.get("/caption_explanability")
def get_caption():
    caption = generator.get_text_explanability()
    return {"image_path": current_image_path, "caption": caption}

# Endpoint to fetch output image (LIME boundary)
@app.get("/output_image")
def output_image():
    try:
        buf = generator.generate_image_output(current_image_path)
        return StreamingResponse(buf, media_type="image/png")
    except Exception as e:
        return {"error": str(e)}


# Background task to save survey data to CSV
def save_survey_data_to_csv(data):
    try:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"{timestamp}_survey_data.csv"
        file_path = os.path.join(DATA_DIR, filename)

        with open(file_path, mode='a', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=data[0].dict().keys())
            if os.stat(file_path).st_size == 0:
                writer.writeheader()
            for entry in data:
                writer.writerow(entry.dict())
        
        print(f"Survey data saved to {file_path}")
    except Exception as e:
        print(f"Error saving survey data: {e}")

# Endpoint to save survey data
@app.post("/save_responses")
async def save_survey_data(data: SurveyData, background_tasks: BackgroundTasks):
    global responses_cache

    responses_cache.append(data)

    # Check if we have collected 20 responses
    if len(responses_cache) >= 1:
        # Save responses to CSV in a background task
        background_tasks.add_task(save_survey_data_to_csv, responses_cache.copy())
        responses_cache = []  # Clear cache for next set of responses

    # Fetch the next image path based on current question count
    next_image_path = get_next_image_path()

    # Check if there are more images to display
    if next_image_path:
        return JSONResponse({"message": "Survey data received.", "next_image_path": next_image_path})
    else:
        return JSONResponse({"message": "Survey data received. No more images.", "next_image_path": None})

# Initial fetch of image path
get_next_image_path()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)