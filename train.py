import os
import random

import librosa
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix


# ============================================================
# SETTINGS
# ============================================================

DATASET_PATH = "../dataset"
MODEL_PATH = "../model/emotion_cnn_mfcc_delta.pth"

SAMPLE_RATE = 22050
NUM_MFCC = 40
MAX_LENGTH = 130

BATCH_SIZE = 32
EPOCHS = 40

LEARNING_RATE = 0.001

PATIENCE = 7

RANDOM_STATE = 42


# ============================================================
# EMOTION MAPPING - RAVDESS
# ============================================================

emotion_map = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}


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


label_to_index = {
    emotion: i
    for i, emotion in enumerate(classes)
}


# ============================================================
# RANDOM SEED
# ============================================================

random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_features(file_path):

    audio, sr = librosa.load(
        file_path,
        sr=SAMPLE_RATE,
        duration=3
    )

    if len(audio) == 0:
        return None

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

    # --------------------------------------------------------
    # Fix time dimension
    # --------------------------------------------------------

    def fix_length(feature):

        if feature.shape[1] < MAX_LENGTH:

            padding = MAX_LENGTH - feature.shape[1]

            feature = np.pad(
                feature,
                ((0, 0), (0, padding)),
                mode="constant"
            )

        else:

            feature = feature[:, :MAX_LENGTH]

        return feature


    mfcc = fix_length(mfcc)
    delta = fix_length(delta)
    delta_delta = fix_length(delta_delta)


    # --------------------------------------------------------
    # Normalize each feature
    # --------------------------------------------------------

    def normalize(feature):

        mean = np.mean(feature)
        std = np.std(feature)

        return (
            feature - mean
        ) / (std + 1e-8)


    mfcc = normalize(mfcc)
    delta = normalize(delta)
    delta_delta = normalize(delta_delta)


    # --------------------------------------------------------
    # 3 CHANNELS
    # --------------------------------------------------------

    features = np.stack(
        [
            mfcc,
            delta,
            delta_delta
        ],
        axis=0
    )

    return features.astype(np.float32)


# ============================================================
# LOAD DATASET
# ============================================================

print("\nLoading dataset...")

X = []
y = []

files = []

for root, dirs, filenames in os.walk(DATASET_PATH):

    for filename in filenames:

        if not filename.lower().endswith(".wav"):
            continue

        file_path = os.path.join(
            root,
            filename
        )

        # RAVDESS filename
        parts = filename.split("-")

        if len(parts) < 3:
            continue

        emotion_code = parts[2]

        if emotion_code not in emotion_map:
            continue

        emotion = emotion_map[
            emotion_code
        ]

        label = label_to_index[
            emotion
        ]

        files.append(
            (file_path, label)
        )


print(
    "Total speech samples:",
    len(files)
)


# ============================================================
# EXTRACT FEATURES
# ============================================================

for i, (file_path, label) in enumerate(files):

    features = extract_features(
        file_path
    )

    if features is not None:

        X.append(features)
        y.append(label)

    if (i + 1) % 100 == 0:

        print(
            f"Processed {i + 1}/{len(files)}"
        )


X = np.array(X)

y = np.array(y)


print(
    "\nFeature shape:",
    X.shape
)


print("\nEmotion classes:")

for i, emotion in enumerate(classes):

    print(
        f"{i} = {emotion}"
    )


# ============================================================
# TRAIN / VALIDATION / TEST SPLIT
# ============================================================

X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y
)


X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=RANDOM_STATE,
    stratify=y_temp
)


print(
    "\nTraining data:",
    X_train.shape
)

print(
    "Validation data:",
    X_val.shape
)

print(
    "Testing data:",
    X_test.shape
)


# ============================================================
# DATASET CLASS
# ============================================================

class SpeechDataset(Dataset):

    def __init__(
        self,
        X,
        y,
        augment=False
    ):

        self.X = X
        self.y = y
        self.augment = augment


    def __len__(self):

        return len(self.X)


    def __getitem__(self, index):

        feature = self.X[index].copy()

        label = self.y[index]


        # ====================================================
        # DATA AUGMENTATION
        # ====================================================

        if self.augment:

            # Add small random noise
            if random.random() < 0.30:

                noise = np.random.normal(
                    0,
                    0.03,
                    feature.shape
                ).astype(np.float32)

                feature += noise


            # Time masking
            if random.random() < 0.30:

                width = random.randint(
                    3,
                    12
                )

                start = random.randint(
                    0,
                    MAX_LENGTH - width
                )

                feature[:, :, start:start + width] = 0


            # Frequency masking
            if random.random() < 0.30:

                width = random.randint(
                    2,
                    8
                )

                start = random.randint(
                    0,
                    NUM_MFCC - width
                )

                feature[:, start:start + width, :] = 0


            # Small gain variation
            if random.random() < 0.30:

                gain = random.uniform(
                    0.9,
                    1.1
                )

                feature *= gain


        feature = torch.tensor(
            feature,
            dtype=torch.float32
        )

        label = torch.tensor(
            label,
            dtype=torch.long
        )

        return feature, label


# ============================================================
# CREATE DATASETS
# ============================================================

train_dataset = SpeechDataset(
    X_train,
    y_train,
    augment=True
)


val_dataset = SpeechDataset(
    X_val,
    y_val,
    augment=False
)


test_dataset = SpeechDataset(
    X_test,
    y_test,
    augment=False
)


# ============================================================
# DATALOADERS
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)


val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# ============================================================
# CNN + BiLSTM MODEL
# ============================================================

class EmotionCNNLSTM(nn.Module):

    def __init__(
        self,
        num_classes
    ):

        super().__init__()


        # ====================================================
        # CNN
        # ====================================================

        self.cnn = nn.Sequential(

            # 3 → 32
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


            # 32 → 64
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


            # 64 → 128
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


        # ====================================================
        # BiLSTM
        # ====================================================

        # Input:
        # 128 channels
        # frequency dimension = 5
        #
        # 128 * 5 = 640

        self.lstm = nn.LSTM(
            input_size=640,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            dropout=0.3,
            bidirectional=True
        )


        # ====================================================
        # CLASSIFIER
        # ====================================================

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

        # CNN

        x = self.cnn(x)


        # ----------------------------------------------------
        # Shape:
        # Batch × Channel × Frequency × Time
        # ----------------------------------------------------

        batch_size = x.size(0)

        channels = x.size(1)

        frequency = x.size(2)

        time = x.size(3)


        # ----------------------------------------------------
        # Convert to:
        # Batch × Time × Features
        # ----------------------------------------------------

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


        # ----------------------------------------------------
        # BiLSTM
        # ----------------------------------------------------

        x, _ = self.lstm(x)


        # Take final timestep

        x = x[:, -1, :]


        # Classifier

        x = self.classifier(x)


        return x


# ============================================================
# CREATE MODEL
# ============================================================

model = EmotionCNNLSTM(
    num_classes=len(classes)
)

model = model.to(device)


print("\nModel architecture:")

print(model)


# ============================================================
# CLASS WEIGHTS
# ============================================================

class_counts = np.bincount(
    y_train,
    minlength=len(classes)
)


class_weights = (
    len(y_train)
    /
    (
        len(classes)
        * class_counts
    )
)


class_weights = torch.tensor(
    class_weights,
    dtype=torch.float32
)


class_weights = class_weights.to(
    device
)


print("\nClass weights:")

for i, emotion in enumerate(classes):

    print(
        emotion,
        ":",
        round(
            class_weights[i].item(),
            3
        )
    )


# ============================================================
# LOSS
# ============================================================

criterion = nn.CrossEntropyLoss(
    weight=class_weights,
    label_smoothing=0.05
)


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=1e-4
)


# ============================================================
# LR SCHEDULER
# ============================================================

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="max",
    factor=0.5,
    patience=3
)


# ============================================================
# TRAINING
# ============================================================

best_val_accuracy = 0

epochs_without_improvement = 0


print("\n======================================")
print("Starting CNN + BiLSTM training")
print("======================================\n")


for epoch in range(EPOCHS):

    # ========================================================
    # TRAIN
    # ========================================================

    model.train()

    running_loss = 0

    correct = 0

    total = 0


    for features, labels in train_loader:

        features = features.to(device)

        labels = labels.to(device)


        optimizer.zero_grad()


        outputs = model(
            features
        )


        loss = criterion(
            outputs,
            labels
        )


        loss.backward()


        # Prevent exploding gradients

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=2.0
        )


        optimizer.step()


        running_loss += (
            loss.item()
            * labels.size(0)
        )


        _, predicted = torch.max(
            outputs,
            1
        )


        total += labels.size(0)

        correct += (
            predicted == labels
        ).sum().item()


    train_loss = (
        running_loss / total
    )


    train_accuracy = (
        100 * correct / total
    )


    # ========================================================
    # VALIDATION
    # ========================================================

    model.eval()

    val_correct = 0

    val_total = 0

    val_loss_total = 0


    with torch.no_grad():

        for features, labels in val_loader:

            features = features.to(device)

            labels = labels.to(device)


            outputs = model(
                features
            )


            loss = criterion(
                outputs,
                labels
            )


            val_loss_total += (
                loss.item()
                * labels.size(0)
            )


            _, predicted = torch.max(
                outputs,
                1
            )


            val_total += labels.size(0)

            val_correct += (
                predicted == labels
            ).sum().item()


    val_loss = (
        val_loss_total / val_total
    )


    val_accuracy = (
        100 * val_correct / val_total
    )


    # Scheduler

    scheduler.step(
        val_accuracy
    )


    current_lr = optimizer.param_groups[0]["lr"]


    print(
        f"Epoch [{epoch + 1}/{EPOCHS}] "
        f"Loss: {train_loss:.4f} "
        f"Train Acc: {train_accuracy:.2f}% "
        f"Val Loss: {val_loss:.4f} "
        f"Val Acc: {val_accuracy:.2f}% "
        f"LR: {current_lr:.6f}"
    )


    # ========================================================
    # SAVE BEST MODEL
    # ========================================================

    if val_accuracy > best_val_accuracy:

        best_val_accuracy = val_accuracy

        epochs_without_improvement = 0


        os.makedirs(
            "../model",
            exist_ok=True
        )


        torch.save(
            {
                "model_state_dict":
                    model.state_dict(),

                "classes":
                    classes,

                "best_val_accuracy":
                    best_val_accuracy
            },
            MODEL_PATH
        )


        print(
            "   ✓ Best model saved!"
        )


    else:

        epochs_without_improvement += 1


    # ========================================================
    # EARLY STOPPING
    # ========================================================

    if epochs_without_improvement >= PATIENCE:

        print(
            "\nEarly stopping triggered."
        )

        break


# ============================================================
# LOAD BEST MODEL
# ============================================================

print(
    "\nLoading best model..."
)


checkpoint = torch.load(
    MODEL_PATH,
    map_location=device,
    weights_only=False
)


model.load_state_dict(
    checkpoint["model_state_dict"]
)


model.eval()


# ============================================================
# TEST
# ============================================================

print(
    "\n======================================"
)

print(
    "Testing best model..."
)

print(
    "======================================"
)


all_predictions = []

all_labels = []


with torch.no_grad():

    for features, labels in test_loader:

        features = features.to(device)

        outputs = model(
            features
        )


        predictions = torch.argmax(
            outputs,
            dim=1
        )


        all_predictions.extend(
            predictions.cpu().numpy()
        )


        all_labels.extend(
            labels.numpy()
        )


all_predictions = np.array(
    all_predictions
)


all_labels = np.array(
    all_labels
)


test_accuracy = (
    np.mean(
        all_predictions == all_labels
    )
    * 100
)


print(
    f"\nTest Accuracy: "
    f"{test_accuracy:.2f}%"
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print(
    "\nClassification Report:\n"
)


print(
    classification_report(
        all_labels,
        all_predictions,
        labels=list(
            range(len(classes))
        ),
        target_names=classes,
        zero_division=0
    )
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

print(
    "\nConfusion Matrix:\n"
)


cm = confusion_matrix(
    all_labels,
    all_predictions
)


print(cm)


# ============================================================
# FINISHED
# ============================================================

print(
    "\n======================================"
)

print(
    "Training completed!"
)

print(
    "======================================"
)

print(
    f"Best Validation Accuracy: "
    f"{best_val_accuracy:.2f}%"
)

print(
    f"Test Accuracy: "
    f"{test_accuracy:.2f}%"
)

print(
    "\nModel saved at:"
)

print(
    MODEL_PATH
)