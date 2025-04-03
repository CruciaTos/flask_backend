from flask import Flask, request, jsonify
from flask_cors import CORS
import whisper
import tensorflow as tf
import numpy as np
import os
import tempfile
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import logging

app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load models only once
if not hasattr(app, 'whisper_model'):
    print("Loading Whisper model...")
    app.whisper_model = whisper.load_model("base")
    print("Whisper model loaded successfully!")

if not hasattr(app, 'text_model'):
    print("Loading AI Text Classifier model...")
    app.text_model = tf.keras.models.load_model('ai_text_classifier.keras')
    print("AI Text Classifier model loaded successfully!")

# Initialize tokenizer and fit on dummy data
tokenizer = Tokenizer(num_words=50000)
tokenizer.fit_on_texts(["Sample text for tokenizer initialization."])  # Dummy training

@app.route('/transcribe', methods=['POST'])
def transcribe():
    try:
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400
        
        audio_file = request.files['audio']
        if audio_file.filename == '':
            return jsonify({'error': 'Empty file uploaded'}), 400

        # Save audio file temporarily
        with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as tmp:
            tmp_path = tmp.name
            audio_file.save(tmp_path)

        try:
            result = app.whisper_model.transcribe(tmp_path)
            os.remove(tmp_path)  # Delete temp file after transcription
            return jsonify({
                'transcription': result['text'].strip(),
                'language': result['language']
            })
        except Exception as e:
            logger.error(f"Transcription error: {str(e)}")
            os.remove(tmp_path)  # Delete file even if an error occurs
            return jsonify({'error': f'Transcription failed: {str(e)}'}), 500

    except Exception as e:
        logger.error(f"Server error: {str(e)}")
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({'error': 'Text parameter is required'}), 400
            
        text = str(data['text']).strip()
        if not text:
            return jsonify({'error': 'Text cannot be empty'}), 400

        # Convert text to sequence
        sequences = tokenizer.texts_to_sequences([text])
        padded = pad_sequences(sequences, maxlen=512, padding='post', truncating='post')
        
        # Predict
        raw_prediction = app.text_model.predict(padded)
        predicted_label = int(np.argmax(raw_prediction, axis=1)[0])
        
        label_map = {
            0: 'ai-generated',
            1: 'human-written', 
            2: 'human-spoken'
        }
        
        return jsonify({
            'prediction': predicted_label,
            'label': label_map[predicted_label],
            'confidence': float(np.max(raw_prediction))
        })

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    import os
    os.environ["FLASK_ENV"] = "development"  # Ensure development mode
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)  # Debug set to False
