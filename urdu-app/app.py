import os
import numpy as np
from flask import Flask, request, jsonify, render_template
from preprocess import extract_mfcc, extract_spectrogram, lstm_model, cnn_model

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Save the uploaded file temporarily
    file_path = os.path.join('temp', file.filename)
    file.save(file_path)

    # Get model type from the form (default to LSTM if not specified)
    model_type = request.form.get('model', 'lstm').lower()

    if model_type == 'cnn':
        # Process using CNN
        spectrogram = extract_spectrogram(file_path)
        spectrogram = np.expand_dims(spectrogram, axis=0)  # Add batch dimension
        prediction = cnn_model.predict(spectrogram)[0][0]
        class_label = 'Spoofed' if prediction > 0.5 else 'Bonafide'
        return jsonify({'model': 'cnn', 'prediction': class_label, 'confidence': float(prediction)})

    elif model_type == 'ensemble':
        # Process for Ensemble (soft and hard voting)
        # LSTM Prediction
        mfcc = extract_mfcc(file_path)
        mfcc = np.expand_dims(mfcc, axis=0)  # Add batch dimension
        lstm_prediction = lstm_model.predict(mfcc)[0][0]

        # CNN Prediction
        spectrogram = extract_spectrogram(file_path)
        spectrogram = np.expand_dims(spectrogram, axis=0)  # Add batch dimension
        cnn_prediction = cnn_model.predict(spectrogram)[0][0]

        # Soft Voting
        combined_prediction_soft = (lstm_prediction + cnn_prediction) / 2

        # Hard Voting
        lstm_class = 1 if lstm_prediction > 0.4 else 0
        cnn_class = 1 if cnn_prediction > 0.5 else 0
        combined_class_hard = 1 if (lstm_class + cnn_class) > 1 else 0

        soft_class_label = 'Spoofed' if combined_prediction_soft > 0.5 else 'Bonafide'
        hard_class_label = 'Spoofed' if combined_class_hard == 1 else 'Bonafide'

        return jsonify({
            'model': 'ensemble',
            'lstm_prediction': float(lstm_prediction),
            'cnn_prediction': float(cnn_prediction),
            'combined_prediction_soft': float(combined_prediction_soft),
            'soft_class_label': soft_class_label,
            'hard_class_label': hard_class_label
        })

    else:
        # Default to LSTM
        mfcc = extract_mfcc(file_path)
        mfcc = np.expand_dims(mfcc, axis=0)  # Add batch dimension
        prediction = lstm_model.predict(mfcc)[0][0]
        class_label = 'Spoofed' if prediction > 0.5 else 'Bonafide'
        return jsonify({'model': 'lstm', 'prediction': class_label, 'confidence': float(prediction)})

# Run the app
if __name__ == '__main__':
    os.makedirs('temp', exist_ok=True)  
    app.run(debug=True)
