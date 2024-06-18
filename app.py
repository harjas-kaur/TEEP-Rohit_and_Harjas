from flask import Flask, jsonify, send_file, request, render_template
from flask_cors import CORS
from main import ObjectGenerator
import pandas as pd
import io
import secrets
import os
import time
import asyncio

app = Flask(__name__)
CORS(app)

total_respondents = 30
ratio = (40, 30, 30)
object_generator = ObjectGenerator(total_respondents, ratio)
survey_responses = []

@app.route('/')
def index():
    confirmation_code = secrets.token_hex(16)
    return render_template('survey.html', confirmation_code=confirmation_code)

@app.route('/input_text', methods=['GET'])
async def input_text():
    input_data = "Dummy input text"
    original_text, _ = object_generator.generate_text_output(input_data)
    return jsonify({'original_text': original_text})

@app.route('/output_data', methods=['GET'])
async def output_data():
    input_data = "Dummy output data"
    _, demo_text = object_generator.generate_text_output(input_data)
    exp_type = object_generator.get_explainability_type()
    output = {'exp_type': exp_type}

    if exp_type in ['image', 'both']:
        output['exp_image_url'] = '/output_image'
    if exp_type in ['text', 'both']:
        output['exp_text'] = demo_text

    return jsonify(output)

@app.route('/input_image', methods=['GET'])
async def get_image():
    image_path = await object_generator.get_next_image_path_async()
    buf_input = await object_generator.generate_image_input_async(image_path)
    return send_file(buf_input, mimetype='image/png')

@app.route('/output_image', methods=['GET'])
async def get_demo_image():
    buf_demo = await object_generator.generate_image_output_async()
    return send_file(buf_demo, mimetype='image/png')

@app.route('/submit_survey', methods=['POST'])
async def submit_survey():
    data = await request.get_json()
    question_id = data.get('question_id')
    human_answer = data.get('human_answer')
    actual_answer = object_generator.get_actual_answer()
    original_answer = object_generator.get_dummy_model_answer()
    explainability_combination = object_generator.get_explainability_type()

    survey_response = {
        'question_id': question_id,
        'actual_answer': actual_answer,
        'human_answer': human_answer,
        'model_prediction': original_answer,
        'explainability_combination': explainability_combination
    }

    survey_responses.append(survey_response)
    object_generator.update_object_index()

    if len(survey_responses) >= 20:
        return jsonify({'status': 'finished', 'confirmation_code': secrets.token_hex(16)})

    return jsonify({'status': 'success'})

@app.route('/save_to_csv', methods=['POST'])
async def save_to_csv():
    df = pd.DataFrame(survey_responses)
    csv_filename = f'survey_responses_{generate_timestamp()}.csv'
    csv_path = os.path.join(DATA_COLLECTED_DIR, csv_filename)
    df.to_csv(csv_path, index=False)
    return jsonify({'status': 'success', 'csv_path': csv_path})

def generate_timestamp():
    return datetime.datetime.now().strftime('%Y%m%d%H%M%S')

class ObjectGenerator:
    def __init__(self, total_respondents, ratio):
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

    async def get_next_image_path_async(self):
        # Async version to fetch the next image path
        image_path = f'./images/image_{self.current_object_index % 20}.jpeg'
        return image_path

    async def generate_image_input_async(self, image_path):
        # Async version to generate image input
        if image_path:
            img = Image.open(image_path).convert("RGB")
        else:
            raise ValueError("image_path must be provided")

        buf = io.BytesIO()
        img.save(buf, format='PNG')
        buf.seek(0)
        return buf

    async def generate_image_output_async(self):
        # Async version to generate image output
        boundary_img = await self.lime_async(image_path)  # Assuming lime has an async version
        await asyncio.sleep(5)  # Simulate some async operation
        boundary_img = (boundary_img * 255).astype(np.uint8)
        img = Image.fromarray(boundary_img)
        buf = io.BytesIO()
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

if __name__ == '__main__':
    app.run(debug=True)
