import json
from flask import Flask, request, jsonify
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)
from pprint import pprint

from dellma.utils.data_utils import convert_data_grade_agent_supported
from dellma.runner.dellma_predict import process_grades
from runner.test import get_new_data_from_video

def create_app():
    app = Flask(__name__)

    # Configure the upload folder
    app.config['UPLOAD_FOLDER'] = 'uploads'

    # Create the upload folder if it doesn't exist
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])

    @app.route('/')
    def welcome():
        return "Welcome to the API!"

    @app.route('/predict', methods=['POST'])
    def predict():
        if request.method == 'POST':
            # Check if the video file is present in the request
            if 'video' not in request.files:
                return jsonify(error="No video file found in the request"), 400

            video_file = request.files['video']

            # Check if the video file has a valid filename
            if video_file.filename == '':
                return jsonify(error="Invalid video filename"), 400

            # Save the uploaded video file to the upload folder
            video_path = os.path.join(app.config['UPLOAD_FOLDER'], video_file.filename)
            video_file.save(video_path)

            # Process the video file path

            result = get_new_data_from_video(video_path)
            result = "[" + json.dumps(result) + "]"
            current_student_action = result

            print(f" Actions generated: {current_student_action}")
            

        #     current_student_action = '''
        #     [{
        #         "actions": ["scrobar examination"],
        #         "discussions": [
        #             " Looking at the front of the patients, just looking at the posture, generally, any swelling",
        #             " and the knee, the muscle bulk, any potential injuries.",
        #             " The side, just looking at the pelvis, the rear, just looking at the back and the spine,",
        #             "",
        #             " and that's it, I think I'll go on.",
        #             ""
        #         ],
        #         "actions_and_discussions": [
        #             {
        #                 "actions": ["scrobar examination","hand examination"],
        #                 "discussions": " Looking at the front of the patients, just looking at the posture, generally, any swelling",
        #                 "start_time": 0,
        #                 "end_time": 15.0
        #             },
        #             {
        #                 "actions": [],
        #                 "discussions": " and the knee, the muscle bulk, any potential injuries.",
        #                 "start_time": 15.0,
        #                 "end_time": 26.0
        #             },
        #             {
        #                 "actions": [],
        #                 "discussions": " The side, just looking at the pelvis, the rear, just looking at the back and the spine,",
        #                 "start_time": 25.0,
        #                 "end_time": 42.0
        #             },
        #             {
        #                 "actions": [],
        #                 "discussions": "",
        #                 "start_time": 42.0,
        #                 "end_time": 44.0
        #             },
        #             {
        #                 "actions": [],
        #                 "discussions": " and that's it, I think I'll go on.",
        #                 "start_time": 44.0,
        #                 "end_time": 48.0
        #             },
        #             { "actions": [], "discussions": "", "start_time": 48.0, "end_time": 73 }
        #         ]
        #     }]
        # '''


            query, result = process_grades(
                sc_samples=5,
                dellma_mode="cot",
                current_physiotherapy_analysis_to_grade=convert_data_grade_agent_supported(current_student_action, query="")
            )

            # Remove the uploaded video file after processing
            os.remove(video_path)
            query = [i["prompt"].replace("\n", "<br/>").replace("\\n", "<br/>") for i in query]

            result = {
                "query": query,
                "result": result,
                "actions_and_discussions": current_student_action
            }
            

            return jsonify(result= result)

    return app