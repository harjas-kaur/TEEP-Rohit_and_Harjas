from PIL import Image, ImageDraw
from io import BytesIO
import random
import Model.image_final as image_final
import os
import asyncio

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
    
    async def _get_image_files(self):
        image_files = os.listdir(self.image_folder)
        image_files.sort()  # Sort files to ensure they are in order
        return image_files
    
    async def _get_next_image_path(self):
        if self.current_object_index < len(self.image_files):
            image_filename = self.image_files[self.current_object_index]
            image_path = os.path.join(self.image_folder, image_filename)
            return image_path
        else:
            raise ValueError("No more images to fetch.")
    
    def generate_text_output(self, input_data):
        original_text = f"Input text: {input_data}, Object Number: {self.current_object_index + 1}"
        demo_text = f"Dummy output text from explainability. Number: {self.current_object_index + 1}"
        return original_text, demo_text

    @staticmethod
    async def generate_image_input(image_path):
        # Load the image from the given path
        if image_path:
            img = Image.open(image_path).convert("RGB")
        else:
            raise ValueError("image_path must be provided")
        
        buf = BytesIO()
        img.save(buf, format='PNG')
        buf.seek(0)
        return buf

    async def generate_image_output(self, is_input=False):
        # Get the path of the next image
        image_path = self._get_next_image_path()
        
        # Load the image using LIME or any other processing
        boundary_img = image_final.lime(image_path)
        
        # Convert the LIME output to a PIL image
        #time.sleep(5)  # Simulating a delay
        boundary_img = (boundary_img * 255).astype(np.uint8)
        img = Image.fromarray(boundary_img)
        
        # Save the image to a buffer
        buf = BytesIO()
        img.save(buf, format='PNG')
        buf.seek(0)
        
        return buf

    def update_object_index(self):
        self.current_object_index += 1

    def get_explainability_type(self):
        return self.distribution[self.current_object_index % self.total_respondents]

    def get_dummy_model_answer(self):
        # Return a dummy answer provided by the model
        return "Dummy_model_ans"

    def get_actual_answer(self):
        # Return a random number between 1 and 6
        return "Dummy_actual_ans"
