from PIL import Image, ImageDraw
from io import BytesIO
import random

class ObjectGenerator:
    def __init__(self, total_respondents, ratio):
        self.COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]  # Example colors
        self.current_object_index = 0
        self.explainability_types = ["image", "text", "both"]
        self.total_respondents = total_respondents
        self.ratio = ratio
        self.distribution = self._create_distribution()
    
    def _create_distribution(self):
        total_ratio = sum(self.ratio)
        distribution = []
        for i, count in enumerate(self.ratio):
            distribution.extend([self.explainability_types[i]] * (self.total_respondents * count // total_ratio))
        random.shuffle(distribution)
        return distribution
    
    def generate_text_output(self, input_data):
        original_text = f"Input text: {input_data}, Object Number: {self.current_object_index + 1}"
        demo_text = f"Dummy output text from explainability. Number: {self.current_object_index + 1}"
        return original_text, demo_text

    def generate_image_output(self, is_input=False):
        color = self.COLORS[self.current_object_index % len(self.COLORS)]
        img = Image.new('RGB', (200, 200), color=color)
        draw = ImageDraw.Draw(img)
        draw.text((50, 90), f"Object {self.current_object_index + 1}", fill=(255, 255, 255))
        
        if is_input:
            draw.text((50, 110), "Input Image", fill=(255, 255, 255))
        else:
            draw.text((50, 110), "Output Image", fill=(255, 255, 255))

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
