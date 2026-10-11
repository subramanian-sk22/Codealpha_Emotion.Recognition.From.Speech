import os
import tempfile

import librosa
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Speech Emotion Recognition",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        text-align: center;
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        font-size: 18px;
        margin-bottom: 30px;
    }

    .result-box {
        padding: 25px;
        border-radius: 15px;
        text-align: center;
        border: 1px solid rgba(128,128,128,0.3);
        margin: 10px 0;
    }

    .emotion {
        font-size: 36px;
        font-weight: 700;
    }

    .confidence {
        font-size: 20px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SETTINGS
# ============================================================

MODEL_PATH = "model/emotion_cnn_mfcc_delta.pth"

SAMPLE_RATE = 22050

NUM_MFCC = 40

MAX_LENGTH = 130

MODEL_ACCURACY = 90.62


# ============================================================
# EMOTIONS
# ============================================================

classes = [
    "angry",
    "calm",
    "disgust",
    "fearful",
    "happy",
    "neutral",
    "sad",
    "surprised"
]


emotion_icons = {
    "angry": "😠",
    "calm": "😌",
    "disgust": "🤢",
    "fearful": "😨",
    "happy": "😊",
    "neutral": "😐",
    "sad": "😢",
    "surprised": "😲"
}


# ============================================================
# CNN + BiLSTM MODEL
# ============================================================

class EmotionCNNLSTM(nn.Module):

    def __init__(self, num_classes):

        super().__init__()

        self.cnn = nn.Sequential(

            nn.Conv2d(
                3,
                32,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(32),

            nn.ReLU(),

            nn.MaxPool2d(2),

            nn.Dropout2d(0.15),


            nn.Conv2d(
                32,
                64,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(64),

            nn.ReLU(),

            nn.MaxPool2d(2),

            nn.Dropout2d(0.20),


            nn.Conv2d(
                64,
                128,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(128),

            nn.ReLU(),

            nn.MaxPool2d(2),

            nn.Dropout2d(0.25)
        )


        self.lstm = nn.LSTM(
            input_size=640,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            dropout=0.3,
            bidirectional=True
        )


        self.classifier = nn.Sequential(

            nn.Linear(
                256,
                128
            ),

            nn.ReLU(),

            nn.Dropout(0.4),

            nn.Linear(
                128,
                num_classes
            )
        )


    def forward(self, x):

        x = self.cnn(x)

        batch_size = x.size(0)

        channels = x.size(1)

        frequency = x.size(2)

        time = x.size(3)

        x = x.permute(
            0,
            3,
            1,
            2
        )

        x = x.contiguous().view(
            batch_size,
            time,
            channels * frequency
        )

        x, _ = self.lstm(x)

        x = x[:, -1, :]

        x = self.classifier(x)

        return x


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device,
        weights_only=False
    )

    model = EmotionCNNLSTM(
        num_classes=len(classes)
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model = model.to(device)

    model.eval()

    return model, device


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_features(audio_source):

    audio, sr = librosa.load(
        audio_source,
        sr=SAMPLE_RATE,
        duration=3
    )

    if len(audio) == 0:

        raise ValueError(
            "Audio file is empty."
        )


    # MFCC

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=NUM_MFCC
    )


    # Delta

    delta = librosa.feature.delta(
        mfcc
    )


    # Delta-Delta

    delta_delta = librosa.feature.delta(
        mfcc,
        order=2
    )


    # Fix length

    def fix_length(feature):

        if feature.shape[1] < MAX_LENGTH:

            padding = (
                MAX_LENGTH -
                feature.shape[1]
            )

            feature = np.pad(
                feature,
                ((0, 0), (0, padding)),
                mode="constant"
            )

        else:

            feature = feature[
                :, :MAX_LENGTH
            ]

        return feature


    mfcc = fix_length(mfcc)

    delta = fix_length(delta)

    delta_delta = fix_length(
        delta_delta
    )


    # Normalize

    def normalize(feature):

        mean = np.mean(feature)

        std = np.std(feature)

        return (
            feature - mean
        ) / (std + 1e-8)


    mfcc = normalize(mfcc)

    delta = normalize(delta)

    delta_delta = normalize(
        delta_delta
    )


    # 3-channel input

    features = np.stack(
        [
            mfcc,
            delta,
            delta_delta
        ],
        axis=0
    )


    features = torch.tensor(
        features,
        dtype=torch.float32
    )


    return features.unsqueeze(0)


# ============================================================
# PREDICTION
# ============================================================

def predict_emotion(
    audio_path,
    model,
    device
):

    features = extract_features(
        audio_path
    )

    features = features.to(device)


    with torch.no_grad():

        output = model(
            features
        )

        probabilities = torch.softmax(
            output,
            dim=1
        )


    probabilities = (
        probabilities[0]
        .cpu()
        .numpy()
    )


    predicted_index = np.argmax(
        probabilities
    )


    emotion = classes[
        predicted_index
    ]


    confidence = (
        probabilities[predicted_index]
        * 100
    )


    return (
        emotion,
        confidence,
        probabilities
    )


# ============================================================
# LOAD MODEL
# ============================================================

try:

    model, device = load_model()

except Exception as e:

    st.error(
        "Could not load the trained model."
    )

    st.code(str(e))

    st.stop()


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">'
    '🎙️ Speech Emotion Recognition'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'AI-powered human emotion detection from speech'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header(
        "🧠 Model Information"
    )

    st.success(
        "Model Loaded"
    )

    st.metric(
        "Test Accuracy",
        "90.62%"
    )

    st.write(
        "**Architecture**"
    )

    st.write(
        "CNN + BiLSTM"
    )

    st.write(
        "**Features**"
    )

    st.write(
        "MFCC + Delta + Delta-Delta"
    )

    st.write(
        "**Dataset**"
    )

    st.write(
        "RAVDESS"
    )

    st.write(
        "**Number of Classes**"
    )

    st.write(
        "8 Emotions"
    )

    st.divider()

    st.write(
        "### Supported Emotions"
    )

    for emotion in classes:

        st.write(
            f"{emotion_icons[emotion]} "
            f"{emotion.capitalize()}"
        )


# ============================================================
# MAIN INPUT SECTION
# ============================================================

st.subheader(
    "🎧 Provide Speech Audio"
)


input_method = st.radio(
    "Choose an input method:",
    [
        "🎤 Record from Microphone",
        "📁 Upload WAV File"
    ],
    horizontal=True
)


audio_input = None


# ============================================================
# MICROPHONE
# ============================================================

if input_method == "🎤 Record from Microphone":

    st.info(
        "🎤 Click the microphone button, "
        "then speak clearly for a few seconds."
    )

    audio_input = st.audio_input(
        "Record your voice"
    )


# ============================================================
# UPLOAD
# ============================================================

else:

    st.info(
        "📁 Upload a WAV speech recording."
    )

    audio_input = st.file_uploader(
        "Choose a WAV file",
        type=["wav"]
    )


# ============================================================
# AUDIO RECEIVED
# ============================================================

if audio_input is not None:

    st.success(
        "✅ Audio received!"
    )


    st.audio(
        audio_input,
        format="audio/wav"
    )


    st.divider()


    # ========================================================
    # ANALYZE BUTTON
    # ========================================================

    if st.button(
        "🔍 Analyze Speech",
        use_container_width=True
    ):

        temp_path = None


        try:

            with st.spinner(
                "🧠 Analyzing your speech..."
            ):

                # Get audio bytes

                if hasattr(
                    audio_input,
                    "getvalue"
                ):

                    audio_bytes = (
                        audio_input.getvalue()
                    )

                else:

                    audio_bytes = (
                        audio_input.read()
                    )


                # Temporary WAV file

                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=".wav"
                ) as temp_file:

                    temp_file.write(
                        audio_bytes
                    )

                    temp_path = (
                        temp_file.name
                    )


                # Prediction

                (
                    emotion,
                    confidence,
                    probabilities
                ) = predict_emotion(
                    temp_path,
                    model,
                    device
                )


            # ==================================================
            # RESULT
            # ==================================================

            st.divider()


            st.subheader(
                "🎯 Emotion Detection Result"
            )


            icon = emotion_icons[
                emotion
            ]


            st.markdown(
                f"""
                <div class="result-box">

                <div class="emotion">
                {icon} {emotion.upper()}
                </div>

                <div class="confidence">
                Confidence: {confidence:.2f}%
                </div>

                </div>
                """,
                unsafe_allow_html=True
            )


            # ==================================================
            # TOP 3
            # ==================================================

            st.subheader(
                "🏆 Top 3 Predictions"
            )


            sorted_indices = np.argsort(
                probabilities
            )[::-1]


            top_three = (
                sorted_indices[:3]
            )


            for rank, index in enumerate(
                top_three,
                start=1
            ):

                emotion_name = classes[
                    index
                ]

                probability = (
                    probabilities[index]
                    * 100
                )

                st.write(
                    f"**{rank}. "
                    f"{emotion_icons[emotion_name]} "
                    f"{emotion_name.capitalize()}**"
                )

                st.progress(
                    int(probability)
                )

                st.caption(
                    f"{probability:.2f}%"
                )


            # ==================================================
            # ALL EMOTIONS
            # ==================================================

            st.subheader(
                "📊 Emotion Probability"
            )


            chart_data = pd.DataFrame(
                {
                    "Emotion": [
                        emotion.capitalize()
                        for emotion in classes
                    ],

                    "Probability": [
                        round(
                            probability * 100,
                            2
                        )
                        for probability in probabilities
                    ]
                }
            )


            chart_data = chart_data.sort_values(
                "Probability",
                ascending=False
            )


            st.bar_chart(
                chart_data.set_index(
                    "Emotion"
                )
            )


            # ==================================================
            # MODEL INFO
            # ==================================================

            st.divider()


            col1, col2, col3 = st.columns(3)


            with col1:

                st.metric(
                    "Model Accuracy",
                    "90.62%"
                )


            with col2:

                st.metric(
                    "Emotion Classes",
                    "8"
                )


            with col3:

                st.metric(
                    "Model",
                    "CNN + BiLSTM"
                )


        except Exception as e:

            st.error(
                "❌ Could not analyze the audio."
            )

            st.exception(e)


        finally:

            if temp_path is not None:

                try:

                    os.remove(
                        temp_path
                    )

                except:

                    pass


# ============================================================
# PROJECT INFORMATION
# ============================================================

st.divider()


st.subheader(
    "📚 About the Project"
)


col1, col2 = st.columns(2)


with col1:

    st.write(
        """
### 🎯 Objective

Recognize human emotions from speech
audio using speech signal processing
and deep learning techniques.

### 🔊 Input

- Microphone recording
- WAV audio file
"""
    )


with col2:

    st.write(
        """
### 🧠 Technology

- Python
- PyTorch
- Librosa
- Streamlit
- CNN
- BiLSTM

### 🎵 Dataset

RAVDESS
"""
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()


st.caption(
    "Speech Emotion Recognition • "
    "CNN + BiLSTM • RAVDESS • PyTorch"
)