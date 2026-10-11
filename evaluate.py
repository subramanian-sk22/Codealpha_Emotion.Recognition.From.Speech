import os
import torch
import torch.nn as nn
import librosa
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


# ============================================================
# SETTINGS
# ============================================================

MODEL_PATH = "../model/emotion_cnn_mfcc_delta.pth"
DATASET_PATH = "../dataset"

SAMPLE_RATE = 22050
NUM_MFCC = 40
MAX_LENGTH = 130

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
# MODEL
# ============================================================

class EmotionCNNLSTM(nn.Module):

    def __init__(self, num_classes):

        super().__init__()

        self.cnn = nn.Sequential(

            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.15),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.20),

            nn.Conv2d(64, 128, 3, padding=1),
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

            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes)
        )


    def forward(self, x):

        x = self.cnn(x)

        batch_size = x.size(0)

        channels = x.size(1)

        frequency = x.size(2)

        time = x.size(3)

        x = x.permute(0, 3, 1, 2)

        x = x.contiguous().view(
            batch_size,
            time,
            channels * frequency
        )

        x, _ = self.lstm(x)

        x = x[:, -1, :]

        return self.classifier(x)


# ============================================================
# LOAD MODEL
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

print("Using device:", device)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device,
    weights_only=False
)

model = EmotionCNNLSTM(
    len(classes)
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.to(device)

model.eval()


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_features(file):

    audio, sr = librosa.load(
        file,
        sr=SAMPLE_RATE,
        duration=3
    )

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=NUM_MFCC
    )

    delta = librosa.feature.delta(
        mfcc
    )

    delta_delta = librosa.feature.delta(
        mfcc,
        order=2
    )


    def fix_length(feature):

        if feature.shape[1] < MAX_LENGTH:

            feature = np.pad(
                feature,
                (
                    (0, 0),
                    (
                        0,
                        MAX_LENGTH -
                        feature.shape[1]
                    )
                ),
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


    def normalize(feature):

        return (
            feature - np.mean(feature)
        ) / (
            np.std(feature) + 1e-8
        )


    features = np.stack(
        [
            normalize(mfcc),
            normalize(delta),
            normalize(delta_delta)
        ]
    )


    return torch.tensor(
        features,
        dtype=torch.float32
    ).unsqueeze(0)


# ============================================================
# LOAD DATASET
# ============================================================

files = []

labels = []


print("\nLoading dataset...")


for actor in os.listdir(DATASET_PATH):

    actor_path = os.path.join(
        DATASET_PATH,
        actor
    )

    if not os.path.isdir(actor_path):
        continue


    for filename in os.listdir(
        actor_path
    ):

        if not filename.endswith(".wav"):
            continue


        path = os.path.join(
            actor_path,
            filename
        )


        # RAVDESS:
        # 03-01-01-01-01-01-01.wav
        #
        # Third number = emotion

        emotion_code = int(
            filename.split("-")[2]
        )


        label = emotion_code - 1


        files.append(path)

        labels.append(label)


print(
    "Total files:",
    len(files)
)


# ============================================================
# PREDICTIONS
# ============================================================

y_true = []

y_pred = []


print("\nRunning predictions...")


with torch.no_grad():

    for i, file in enumerate(files):

        try:

            features = extract_features(
                file
            )

            features = features.to(
                device
            )


            output = model(
                features
            )


            prediction = torch.argmax(
                output,
                dim=1
            ).item()


            y_true.append(
                labels[i]
            )

            y_pred.append(
                prediction
            )


        except Exception as e:

            print(
                "Skipped:",
                file,
                e
            )


        if (i + 1) % 100 == 0:

            print(
                f"Processed {i + 1}/{len(files)}"
            )


# ============================================================
# ACCURACY
# ============================================================

accuracy = np.mean(
    np.array(y_true) ==
    np.array(y_pred)
) * 100


print(
    f"\nDataset Accuracy: {accuracy:.2f}%"
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_true,
    y_pred
)


fig, ax = plt.subplots(
    figsize=(9, 8)
)


display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=[
        x.capitalize()
        for x in classes
    ]
)


display.plot(
    ax=ax,
    xticks_rotation=45
)


plt.title(
    "Speech Emotion Recognition - Confusion Matrix"
)

plt.tight_layout()


plt.savefig(
    "../results/confusion_matrix.png",
    dpi=300
)


plt.close()


print(
    "\nConfusion matrix saved:"
)

print(
    "../results/confusion_matrix.png"
)


print(
    "\nEvaluation completed!"
)