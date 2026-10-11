🎙️ Speech Emotion Recognition

A deep learning-based Speech Emotion Recognition (SER) system that analyzes human speech and predicts the emotion expressed in the audio.

The project uses MFCC-based speech feature extraction combined with a CNN + BiLSTM neural network and provides a user-friendly Streamlit web application supporting both microphone recording and WAV-file upload.

🎯 Objective

The objective of this project is to recognize human emotions such as:

😠 Angry
😌 Calm
🤢 Disgust
😨 Fearful
😊 Happy
😐 Neutral
😢 Sad
😲 Surprised

from speech audio using speech signal processing and deep learning techniques.

✨ Features
🎤 Real-time microphone recording
📁 WAV file upload
🎵 Audio feature extraction using MFCC
📈 Delta and Delta-Delta MFCC features
🧠 CNN-based feature extraction
🔄 BiLSTM for temporal speech patterns
🎯 Emotion prediction
📊 Confidence/probability scores
🏆 Top-3 emotion predictions
📉 Emotion probability visualization
🌐 Interactive Streamlit interface
🧠 Model Architecture

The system follows this pipeline:

                 🎙️ Speech Audio
                       │
              ┌────────┴────────┐
              │                 │
          Microphone       WAV Upload
              │                 │
              └────────┬────────┘
                       ↓
                Audio Processing
                       ↓
              MFCC Feature Extraction
                       ↓
             Delta + Delta-Delta MFCC
                       ↓
                3-Channel Input
                       ↓
                  CNN Layers
                       ↓
                Feature Extraction
                       ↓
                 BiLSTM Layers
                       ↓
              Fully Connected Layer
                       ↓
               Softmax Classification
                       ↓
              🎯 Emotion Prediction
CNN

The Convolutional Neural Network (CNN) extracts important patterns from the MFCC representation of the speech signal.

BiLSTM

The Bidirectional Long Short-Term Memory (BiLSTM) network learns temporal patterns in speech from both forward and backward directions.

Final Classifier

The extracted features are passed through fully connected layers to classify the speech into one of the eight emotion classes.

🎵 Feature Extraction

The project extracts three types of acoustic features:

1. MFCC

Mel-Frequency Cepstral Coefficients (MFCCs) represent important characteristics of human speech based on how humans perceive sound frequencies.

The project extracts:

40 MFCC coefficients
2. Delta MFCC

Delta features represent changes in MFCC values over time.

3. Delta-Delta MFCC

Delta-Delta features represent changes in the delta features and provide additional temporal information.

These three feature sets are combined as three input channels for the CNN.

📚 Dataset

The project uses the:

RAVDESS — Ryerson Audio-Visual Database of Emotional Speech and Song

The dataset contains speech recordings representing multiple emotional states.

The eight emotion classes used in this project are:

Label	Emotion
0	Angry
1	Calm
2	Disgust
3	Fearful
4	Happy
5	Neutral
6	Sad
7	Surprised
📊 Model Performance

The trained model achieved:

Best Validation Accuracy: 92.71%

Test Accuracy: 90.62%

The model was evaluated using a held-out test set from the RAVDESS dataset.

Note: Performance on microphone recordings can differ from RAVDESS test accuracy because microphone recordings contain different speakers, environments, microphones, background noise, and speaking styles.

🛠️ Technologies Used
Technology	Purpose
Python	Programming language
PyTorch	Deep learning framework
Librosa	Audio processing & feature extraction
NumPy	Numerical operations
Scikit-learn	Evaluation metrics
Matplotlib	Visualization
Pandas	Data handling
Streamlit	Web application
📁 Project Structure
Speech_Emotion_Recognition/
│
├── app.py
├── requirements.txt
├── README.md
│
├── model/
│   └── emotion_cnn_mfcc_delta.pth
│
├── src/
│   ├── train.py
│   ├── predict.py
│   └── evaluate.py
│
├── results/
│   └── confusion_matrix.png
│
├── dataset/
│   └── RAVDESS/
│
└── test_audio/
    └── test.wav
⚙️ Installation
1. Clone the repository
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd Speech_Emotion_Recognition
2. Create a virtual environment
python -m venv venv
Windows PowerShell
.\venv\Scripts\Activate.ps1

If your environment is already activated, you can skip this step.

3. Install dependencies
pip install -r requirements.txt
▶️ Running the Application

From the project root:

streamlit run app.py

The application will open in your browser.

Usually it will be available at:

http://localhost:8501
🎤 Using the Microphone
Start the Streamlit application.
Select 🎤 Record from Microphone.
Allow microphone access.
Record your speech.
Play the recording if required.
Click 🔍 Analyze Speech.
The model will display the predicted emotion and confidence.

Example:

🎯 Emotion Detection Result

😊 HAPPY

Confidence: 84.32%
📁 Using a WAV File
Select 📁 Upload WAV File.
Select a .wav speech recording.
Click 🔍 Analyze Speech.
The model processes the audio.
The predicted emotion is displayed.
📊 Output

The application provides:

Predicted Emotion

The most likely emotion detected by the model.

Confidence

The model's predicted probability for the selected emotion.

Top-3 Predictions

The three emotions with the highest predicted probabilities.

Probability Chart

A visualization of the predicted probabilities for all eight emotion classes.

🔬 Evaluation

The project includes an evaluation script:

cd src
python evaluate.py

This evaluates the trained model and generates a confusion matrix inside:

results/confusion_matrix.png
🧪 Prediction Using a WAV File

The standalone prediction script can also be used:

cd src
python predict.py

Then provide the path to a WAV file when prompted.

🔄 Complete Workflow
RAVDESS Dataset
       ↓
Audio Files
       ↓
Librosa
       ↓
MFCC Extraction
       ↓
Delta MFCC
       ↓
Delta-Delta MFCC
       ↓
CNN
       ↓
BiLSTM
       ↓
Fully Connected Layers
       ↓
Emotion Classification
       ↓
Streamlit Application
       ↓
🎯 Predicted Emotion
🚀 Future Improvements

Possible improvements include:

🎤 Better real-world microphone performance
🔇 Background-noise reduction
👥 Training with more diverse speakers
🎵 Additional audio features such as chroma and spectral features
🤖 Experimenting with Transformer-based speech models
📱 Mobile-friendly deployment
☁️ Cloud deployment
🌍 Support for additional datasets and languages
📈 More detailed model evaluation
🎓 Project Highlights

This project demonstrates the use of:

Speech Signal Processing
        +
Feature Engineering
        +
Deep Learning
        +
CNN
        +
BiLSTM
        +
Model Evaluation
        +
Streamlit Deployment
👨‍💻 Author

Avirup Saha

B.Tech — Computer Science & Engineering (AI & ML)

⭐ Project Summary

Speech Emotion Recognition is a deep learning application that extracts MFCC, Delta, and Delta-Delta features from speech and uses a CNN + BiLSTM architecture to classify the speaker's emotional state into eight categories. The system provides both microphone and WAV-file input through an interactive Streamlit interface and achieved 90.62% test accuracy on the project's RAVDESS test set.

