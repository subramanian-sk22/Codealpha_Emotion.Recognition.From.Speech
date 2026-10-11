import librosa
import numpy as np
import torch
import torch.nn as nn


# ============================================================
# SETTINGS
# ============================================================

MODEL_PATH = "../model/emotion_cnn_mfcc_delta.pth"

SAMPLE_RATE = 22050
NUM_MFCC = 40
MAX_LENGTH = 130


# ============================================================
# EMOTION CLASSES
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


# ============================================================
# CNN + BiLSTM MODEL
# ============================================================

class EmotionCNNLSTM(nn.Module):

    def __init__(self, num_classes):

        super().__init__()

        self.cnn = nn.Sequential(

            # Block 1
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


            # Block 2
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


            # Block 3
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
# DEVICE
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("Using device:", device)


# ============================================================
# LOAD MODEL
# ============================================================

try:

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device,
        weights_only=False
    )

except Exception as e:

    print("\nCould not load model.")
    print("Error:", e)
    exit()


model = EmotionCNNLSTM(
    num_classes=len(classes)
)


model.load_state_dict(
    checkpoint["model_state_dict"]
)


model = model.to(device)

model.eval()


# ============================================================
# AUDIO FILE
# ============================================================

file_path = input(
    "\nEnter path of WAV file: "
).strip()

file_path = file_path.strip('"')


# ============================================================
# LOAD AUDIO
# ============================================================

try:

    audio, sr = librosa.load(
        file_path,
        sr=SAMPLE_RATE,
        duration=3
    )

except Exception as e:

    print("\nCould not load audio.")

    print("Error:", e)

    exit()


if len(audio) == 0:

    print("\nAudio file is empty.")

    exit()


print(
    "\nAudio loaded successfully!"
)


# ============================================================
# MFCC
# ============================================================

mfcc = librosa.feature.mfcc(
    y=audio,
    sr=sr,
    n_mfcc=NUM_MFCC
)


# ============================================================
# DELTA
# ============================================================

delta = librosa.feature.delta(
    mfcc
)


# ============================================================
# DELTA DELTA
# ============================================================

delta_delta = librosa.feature.delta(
    mfcc,
    order=2
)


# ============================================================
# FIX LENGTH
# ============================================================

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


# ============================================================
# NORMALIZE
# ============================================================

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


# ============================================================
# STACK FEATURES
# ============================================================

features = np.stack(
    [
        mfcc,
        delta,
        delta_delta
    ],
    axis=0
)


print(
    "Feature shape:",
    features.shape
)


# ============================================================
# TENSOR
# ============================================================

features = torch.tensor(
    features,
    dtype=torch.float32
)


features = features.unsqueeze(0)

features = features.to(device)


# ============================================================
# PREDICTION
# ============================================================

with torch.no_grad():

    output = model(
        features
    )

    probabilities = torch.softmax(
        output,
        dim=1
    )


# ============================================================
# TOP PREDICTION
# ============================================================

confidence, prediction = torch.max(
    probabilities,
    dim=1
)


emotion = classes[
    prediction.item()
]


confidence = (
    confidence.item() * 100
)


# ============================================================
# TOP 3
# ============================================================

top_probabilities, top_indices = torch.topk(
    probabilities,
    k=3,
    dim=1
)


# ============================================================
# RESULT
# ============================================================

print("\n======================================")

print(
    "       SPEECH EMOTION RESULT"
)

print("======================================")


print(
    f"Predicted Emotion: "
    f"{emotion.upper()}"
)


print(
    f"Confidence: "
    f"{confidence:.2f}%"
)


print("\nTop Predictions:")


for i in range(3):

    index = (
        top_indices[0][i]
        .item()
    )

    probability = (
        top_probabilities[0][i]
        .item()
        * 100
    )

    print(
        f"{i + 1}. "
        f"{classes[index].capitalize():10s}"
        f" {probability:.2f}%"
    )


print(
    "======================================"
)