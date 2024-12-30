# Sarcasm Detection Using Natural Language Processing (NLP)

This project implements a sarcasm detection system using Flask and NLP techniques. It provides an intuitive web interface where users can input sentences to check whether they contain sarcasm. Additionally, it features a text-to-speech (TTS) system using Google TTS for auditory feedback.

---

## Features

- **Web Interface**: Simple Flask-based frontend for user interaction.
- **Sarcasm Detection**: Predicts sarcasm in text input using a trained NLP model.
- **Text-to-Speech**: Google TTS reads out the sentence for auditory feedback.
- **Real-Time Results**: Detection results appear instantly on the interface.

---

## Requirements

To set up the project, ensure that the following dependencies are installed:

- **Python**: Version 3.7 or above
- **Libraries**:
  - Flask
  - Numpy
  - TensorFlow
  - Keras
  - gTTS
  - Pickle
  - Scikit-learn

Install the required libraries with:
```bash
pip install -r requirements.txt
```

---

## Project Workflow

1. **Input Sentence**: 
   - Users can type a sentence in the textbox or use the "Listen" button to input a sentence via Google TTS.
   
2. **Sarcasm Detection**: 
   - The system processes the sentence and predicts whether it is sarcastic.
   
3. **Text-to-Speech**: 
   - The system will read the sentence aloud using Google TTS for better accessibility.

4. **Results Display**: 
   - The application will display whether the sentence is sarcastic or not in real-time.

---

## Setup and Usage

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/rohanvenkatesha/Sarcasm-Detection-using-Natural-Language-Processing
   cd Sarcasm-Detection-using-Natural-Language-Processing
   ```

2. **Start the Flask Application**:
   Run the Flask application:
   ```bash
   python app.py
   ```

3. **Access the Web Interface**:
   Open your web browser and navigate to `http://localhost:5000`.

4. **Using the Application**:
   - Type a sentence in the provided textbox to check if it contains sarcasm.
   - Click the "Listen" button to hear the sentence via Google TTS.

---

## File Structure

```plaintext
Sarcasm-Detection-using-Natural-Language-Processing/
├── static/
│   ├── background-image.jpg          # Background image for the web interface
│   ├── script.js                     # JavaScript for dynamic web features
│   └── styles.css                    # CSS for styling the web interface
├── templates/
│   └── loading.html                  # HTML template for loading screen
├── app.py                            # Main Flask application code
├── predict_sarcasm.py                # Sarcasm prediction helper script
├── speech.py                         # Google TTS functionality
├── Sarcasm_Headlines_Dataset.json    # Dataset for sarcasm detection
├── sarcasm_detection.ipynb           # Jupyter Notebook for model development
├── sarcasm_detection_model.json      # Trained model architecture
├── sarcasm_detection_model_weights.h5# Trained model weights
├── tokenizer.pkl                     # Tokenizer used for text preprocessing
└── README.md                         # Project documentation
```

---

## Model Information

- **Model Architecture**: Stored in `sarcasm_detection_model.json`.
- **Model Weights**: Stored in `sarcasm_detection_model_weights.h5`.
- **Dataset**: The model is trained on the `Sarcasm_Headlines_Dataset.json` dataset.

---

## Future Improvements

- **Tone-Based Sarcasm Detection**: Implement a feature to predict sarcasm based on voice tone, enabling the system to analyze the intonation and emotional cues in spoken sentences. This would allow for more accurate sarcasm detection in audio inputs.
- Enhance the user interface for a more engaging experience.
- Expand support to additional languages for sarcasm detection and TTS.
- Improve model performance with more training data and advanced techniques.

---
