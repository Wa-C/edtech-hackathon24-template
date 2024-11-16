from flask import Flask, request, jsonify
import joblib
import numpy as np


from flask_cors import CORS  # Import CORS




# Load the trained model and label encoders
model = joblib.load('student_learning_model.pkl')
subject_encoder = joblib.load('Subject_label_encoder.pkl')
learning_method_encoder = joblib.load('Preferred_Learning_Method_label_encoder.pkl')
test_type_encoder = joblib.load('Preferred_Test_Type_label_encoder.pkl')
effective_method_encoder = joblib.load('Effective_Learning_Method_label_encoder.pkl')

app = Flask(__name__)
CORS(app)  # Enable CORS

@app.route('/predict', methods=['POST'])
def predict_learning_method():
    try:
        # Parse request JSON
        data = request.get_json()
        print(data)
        student_name = data.get('name', 'Unknown Student')
        print(student_name)
        subject = data.get('subject')
        past_score = data.get('past_score')
        preferred_method = data.get('preferred_learning_method')
        preferred_test_type = data.get('preferred_test_type')
        
        # Validate input
        if not all([subject, past_score, preferred_method, preferred_test_type]):
            return jsonify({'error': 'Missing required fields'}), 400
        
        # Encode categorical inputs
        encoded_subject = subject_encoder.transform([subject])[0]
        encoded_method = learning_method_encoder.transform([preferred_method])[0]
        encoded_test_type = test_type_encoder.transform([preferred_test_type])[0]

        # Create input array for prediction
        input_data = np.array([[encoded_subject, past_score, encoded_method, encoded_test_type]])
        
        # Predict using the model
        prediction = model.predict(input_data)
        predicted_method = effective_method_encoder.inverse_transform(prediction)[0]

        # Return the result
        return jsonify({
            'student_name': student_name,
            'subject': subject,
            'predicted_learning_method': predicted_method
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
