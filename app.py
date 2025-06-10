

# from flask import Flask, request, jsonify
# import numpy as np
# import tensorflow as tf
# from flask_cors import CORS

# app = Flask(__name__)
# CORS(app)

# # Load the trained Keras model
# model = tf.keras.models.load_model("suryanamaskar2_by_deeps.h5")

# # Pose class labels
# class_labels = [
#     "Pranamasana",
#     "HastaUttasana",
#     "ashwa _anchalanasana",
#     "bhujangasana",
#     "kumbhakasana",
#     "ashtanga_namaskara",
#     "Padahastasana",
#     "adho_mukh_svanasana"
# ]

# # Threshold settings
# kp_threshold = 0.3      # minimum confidence score for a keypoint to be considered valid
# min_valid_kp = 15       # minimum number of valid keypoints to proceed with classification
# confidence_threshold = 0.7  # minimum confidence to consider prediction reliable

# @app.route('/predict_pose', methods=['POST'])
# def predict_pose():
#     try:
#         data = request.json
#         keypoints = data.get('keypoints')
#         frame_width = data.get('frameWidth')
#         frame_height = data.get('frameHeight')

#         if keypoints is None or len(keypoints) != 17:
#             return jsonify({'error': 'Expected 17 keypoints with [x, y, score].'}), 400
#         if frame_width is None or frame_height is None:
#             return jsonify({'error': 'Missing frameWidth or frameHeight.'}), 400

#         keypoints = np.array(keypoints, dtype=np.float32)

#         # Count valid keypoints by confidence score
#         valid_kp = np.sum(keypoints[:, 2] > kp_threshold)

#         if valid_kp < min_valid_kp:
#             return jsonify({
#                 'pose': None,
#                 'confidence': 0.0,
#                 'warning': f'Not enough valid keypoints detected ({valid_kp} < {min_valid_kp})',
#                 'keypoints': keypoints.tolist()
#             })

#         # Optionally normalize x and y by frame size if needed
#         # keypoints[:, 0] /= frame_width
#         # keypoints[:, 1] /= frame_height

#         # Reshape for model input: (1, 17, 3, 1)
#         input_data = keypoints.reshape(1, 17, 3, 1)

#         # Predict pose
#         prediction = model.predict(input_data)
#         pred_class = int(np.argmax(prediction))
#         confidence = float(np.max(prediction))
#         label = class_labels[pred_class]

#         warning = ""
#         if confidence < confidence_threshold:
#             warning = "Low confidence in pose detection"

#         return jsonify({
#             'pose': label,
#             'confidence': round(confidence, 3),
#             'warning': warning,
#             'keypoints': keypoints.tolist()
#         })

#     except Exception as e:
#         return jsonify({'error': str(e)}), 500





# @app.route('/')
# def health_check():
#     return "✅ Pose classification API is live!"

# if __name__ == '__main__':
#     import os
#     port = int(os.environ.get("PORT", 8080))
#     app.run(host='0.0.0.0', port=port)





# try 1

from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
import pickle
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load the trained Keras model
model = tf.keras.models.load_model("suryanamaskar2_by_deeps.h5")

# Load the label encoder
with open("label_encoder2.pkl", "rb") as f:
    le = pickle.load(f)

# Thresholds
kp_threshold = 0.3       # Minimum score to consider a keypoint valid
min_valid_kp = 15        # Required number of valid keypoints to proceed
confidence_threshold = 0.7  # Minimum confidence to consider prediction reliable

@app.route('/predict_pose', methods=['POST'])
def predict_pose():
    try:
        data = request.json
        keypoints = data.get('keypoints')
        frame_width = data.get('frameWidth')
        frame_height = data.get('frameHeight')

        if keypoints is None or len(keypoints) != 17:
            return jsonify({'error': 'Expected 17 keypoints with [x, y, score].'}), 400
        if frame_width is None or frame_height is None:
            return jsonify({'error': 'Missing frameWidth or frameHeight.'}), 400

        keypoints = np.array(keypoints, dtype=np.float32)

        # Count valid keypoints
        valid_kp = np.sum(keypoints[:, 2] > kp_threshold)
        if valid_kp < min_valid_kp:
            return jsonify({
                'pose': None,
                'confidence': 0.0,
                'warning': f'Not enough valid keypoints detected ({valid_kp} < {min_valid_kp})',
                'keypoints': keypoints.tolist()
            })

        # Do NOT normalize — model is trained on raw keypoints
        input_data = keypoints.reshape(1, 17, 3, 1)

        # Predict pose
        prediction = model.predict(input_data)
        pred_class = int(np.argmax(prediction))
        confidence = float(np.max(prediction))
        label = le.inverse_transform([pred_class])[0]

        warning = ""
        if confidence < confidence_threshold:
            warning = "Low confidence in pose detection"

        return jsonify({
            'pose': label,
            'confidence': round(confidence, 3),
            'warning': warning,
            'keypoints': keypoints.tolist()
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/')
def health_check():
    return "✅ Pose classification API is live!"

if __name__ == '__main__':
    import os
    port = int(os.environ.get("PORT", 8080))
    app.run(host='0.0.0.0', port=port)
