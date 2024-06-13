from flask import Flask, jsonify, send_file, request, render_template
from flask_cors import CORS
from example import ObjectGenerator
import pandas as pd
import io

app = Flask(__name__)
CORS(app)

total_respondents = 30  # Example total number of respondents
ratio = (40, 30, 30)  # Example ratio
object_generator = ObjectGenerator(total_respondents, ratio)
survey_responses = []

@app.route('/')
def index():
    return render_template('survey.html')

@app.route('/finish')
def finish():
    # Save the survey responses to an Excel file
    df = pd.DataFrame(survey_responses)
    output = io.BytesIO()
    df.to_excel(output, index=False)
    output.seek(0)

    return send_file(output, download_name='survey_responses.xlsx', as_attachment=True)

@app.route('/input_text', methods=['GET'])
def input_text():
    input_data = "Dummy input text"
    original_text, _ = object_generator.generate_text_output(input_data)
    return jsonify({'original_text': original_text})

@app.route('/output_data', methods=['GET'])
def output_data():
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
def get_image():
    buf_input = object_generator.generate_image_output(is_input=True)
    return send_file(buf_input, mimetype='image/png')

@app.route('/output_image', methods=['GET'])
def get_demo_image():
    buf_demo = object_generator.generate_image_output()
    return send_file(buf_demo, mimetype='image/png')

@app.route('/submit_survey', methods=['POST'])
def submit_survey():
    data = request.get_json()
    question_id = data.get('question_id')
    human_answer = data.get('human_answer')
    actual_answer = object_generator.get_actual_answer()
    original_answer=object_generator.get_dummy_model_answer()
    explainability_combination = object_generator.get_explainability_type()

    survey_response = {
        'question_id': question_id,
        'actual_answer': actual_answer,
        'human_answer': human_answer,
        'model_prediction': original_answer,  # Assuming the model prediction is the same as the original answer for this example
        'explainability_combination': explainability_combination
    }
    
    survey_responses.append(survey_response)
    object_generator.update_object_index()
    
    if len(survey_responses) >= 20:
        return jsonify({'status': 'finished'})
    
    return jsonify({'status': 'success'})

if __name__ == '__main__':
    app.run(debug=True)
