from io import BytesIO
import random
import os
import asyncio

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

from flask import Flask, send_file, jsonify
from fastapi import FastAPI, Request
from starlette.responses import StreamingResponse

from Model.image_final import ImageModel


img_model = ImageModel('Model/better_classification.h5')
ratio = [1,1,1]
class ObjectGenerator:
    def __init__(self, total_respondents, ratio):
        self.current_object_index = 0
        self.explainability_types = ["image", "text", "both"]
        self.total_respondents = total_respondents
        self.ratio = ratio
        self.distribution = self._create_distribution()
        self.image_folder = './images'
        self.image_files = self._get_image_files()

    def _create_distribution(self):
        total_ratio = sum(self.ratio)
        distribution = []
        for i, count in enumerate(self.ratio):
            distribution.extend([self.explainability_types[i]] * (self.total_respondents * count // total_ratio))
        random.shuffle(distribution)
        return distribution
    
    def _get_image_files(self):
        image_files = os.listdir(self.image_folder)
        image_files.sort()  # Sort files to ensure they are in order
        return image_files
   
    def _get_next_image_path(self):
        if self.current_object_index < len(self.image_files):
            image_filename = self.image_files[self.current_object_index]
            image_path = os.path.join(self.image_folder, image_filename)
            self.current_object_index += 1
            return image_path
        else:
            raise ValueError("Images Finished")
        
    def generate_image_input(self, image_path):
    # Load the image from the given path
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
    
    def get_explainability_type(self):
        return self.distribution[self.current_object_index % self.total_respondents]

    def get_model_answer(self, image_path):
        # Return a dummy answer provided by the model
        return img_model.predict_model(image_path)

    def get_actual_answer(self):
        # Return a random number between 1 and 6
        return "Dummy_actual_ans"

app= FastAPI()

generator=ObjectGenerator(total_respondents=20, ratio=ratio)
image_path=generator._get_next_image_path()
print(generator.get_model_answer(image_path))

@app.get("/input_image")
def index():
    try:
        buf = generator.generate_image_input(image_path)  # Generate the input image

        # Use StreamingResponse to return the image
        return StreamingResponse(buf, media_type="image/png")
    except Exception as e:
        return {"error": str(e)}
    
@app.get("/model_answer")
def index():
    return generator.get_model_answer(image_path) 

@app.get("/caption")
def index():
    return generator.get_img_caption(image_path)  
    
@app.get("/output_image")
def index():
    try:
        buf = generator.generate_image_output(image_path)  # Generate the output image

        # Use StreamingResponse to return the image
        return StreamingResponse(buf, media_type="image/png")
    except Exception as e:
        return {"error": str(e)}