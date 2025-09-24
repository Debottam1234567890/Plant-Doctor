"""This is the engine file for training and testing a ResNet18 model on the Plant Disease Detection dataset.
It includes data preprocessing, model definition, training, and testing functions."""

# Import necessary libraries
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torchvision.models as models
from torch import nn
import os
import sys
from sklearn.utils.class_weight import compute_class_weight
from collections import Counter
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix
import wandb
from torch.utils.tensorboard import SummaryWriter
import seaborn as sns
from pathlib import Path
from torchvision.datasets.folder import default_loader
from PIL import UnidentifiedImageError, ImageFile
from torch.cuda.amp import GradScaler

# Allow truncated images instead of crashing
ImageFile.LOAD_TRUNCATED_IMAGES = True  

def safe_loader(path):
    try:
        return default_loader(path)
    except (OSError, UnidentifiedImageError) as e:
        print(f"[SKIPPED CORRUPT FILE] {path} | Error: {e}")
        return None

class SafeImageFolder(datasets.ImageFolder):
    def __init__(self, *args, delete_bad=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.delete_bad = delete_bad

    def __getitem__(self, index):
        while True:
            path, target = self.samples[index]
            sample = safe_loader(path)
            if sample is not None:
                if self.transform is not None:
                    sample = self.transform(sample)
                if self.target_transform is not None:
                    target = self.target_transform(target)
                return sample, target
            else:
                if self.delete_bad:
                    try:
                        os.remove(path)
                        print(f"[DELETED CORRUPT FILE] {path}")
                    except Exception as e:
                        print(f"[ERROR DELETING] {path}: {e}")
                index = (index + 1) % len(self.samples)

from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
RUNS_DIR = BASE_DIR / "runs"
MODEL_PATH = MODEL_DIR / "resnet18_plant_disease_detection.pth"

from torchvision.models import resnet18, ResNet18_Weights
resnet18 = models.resnet18(weights=ResNet18_Weights.DEFAULT)
num_classes = 38
resnet18.fc = nn.Linear(resnet18.fc.in_features, num_classes)
for param in resnet18.parameters():
    param.requires_grad = True

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
resnet18 = resnet18.to(device)

if __name__ == "__main__":
    TRAIN_DIR = BASE_DIR / "CombinedDataset" / "train"
    TEST_DIR = BASE_DIR / "CombinedDataset" / "test"

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
    ])

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
    ])

    train_data = SafeImageFolder(root=str(TRAIN_DIR), transform=train_transform, delete_bad=True)
    test_data = SafeImageFolder(root=str(TEST_DIR), transform=test_transform, delete_bad=True)

    train_dataloader = DataLoader(train_data, batch_size=32, shuffle=True, num_workers=0)
    test_dataloader = DataLoader(test_data, batch_size=32, shuffle=False, num_workers=0)
    classes = train_data.classes

    class_counts = train_data.targets
    all_classes = np.arange(num_classes)
    class_weights = np.zeros(num_classes, dtype=np.float32)
    unique_classes = np.unique(class_counts)
    computed_weights = compute_class_weight(class_weight="balanced", classes=unique_classes, y=class_counts)
    for i, cls in enumerate(unique_classes):
        class_weights[cls] = computed_weights[i]
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

    loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(resnet18.parameters(), lr=1e-4)

    from timeit import default_timer as timer
    from torchmetrics import Accuracy
    accuracy_fn = Accuracy(task="multiclass", num_classes=num_classes).to(device)
    scaler = GradScaler()

    def print_train_time(start: float, end: float, device: torch.device):
        total_time = end - start
        print(f"Train time on {device}: {total_time:.3f} seconds")
        return total_time

    def train_step(model, data_loader, loss_fn, optimizer, accuracy_fn, device):
        model.train()
        total_loss = 0
        total_correct = 0
        total_samples = 0
        from tqdm.auto import tqdm
        for X, y in tqdm(data_loader, desc="Training", leave=False):
            X, y = X.to(device).float(), y.to(device)
            with torch.autocast(device_type=device, dtype=torch.float16):
                y_pred = model(X)
                loss = loss_fn(y_pred, y)
            total_loss += loss.item()
            total_correct += (y_pred.argmax(dim=1) == y).sum().item()
            total_samples += len(y)
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        avg_loss = total_loss / len(data_loader)
        acc = total_correct / total_samples * 100
        return avg_loss, acc

    def test_step(model, data_loader, loss_fn, accuracy_fn, device):
        model.eval()
        total_loss = 0
        total_correct = 0
        total_samples = 0
        from tqdm.auto import tqdm
        with torch.inference_mode():
            for X, y in tqdm(data_loader, desc="Testing", leave=False):
                X, y = X.to(device).float(), y.to(device)
                y_pred = model(X)
                loss = loss_fn(y_pred, y)
                total_loss += loss.item()
                total_correct += (y_pred.argmax(dim=1) == y).sum().item()
                total_samples += len(y)
        avg_loss = total_loss / len(data_loader)
        acc = total_correct / total_samples * 100
        return avg_loss, acc

    writer = SummaryWriter(log_dir=str(RUNS_DIR / "plant_disease_resnet18"))
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=3)

    sample_input = torch.randn(*(1, 3, 224, 224)).to(device)
    writer.add_graph(resnet18, sample_input)

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    train_time_start_model = timer()
    epochs = 20
    train_losses, test_losses = [], []
    train_accuracies, test_accuracies = [], []

    try:
        print(f"🚀 Training started!")
        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}\n{'-'*30}")
            train_loss, train_acc = train_step(resnet18, train_dataloader, loss_fn, optimizer, accuracy_fn, device)
            test_loss, test_acc = test_step(resnet18, test_dataloader, loss_fn, accuracy_fn, device)
            scheduler.step(test_loss)
            train_losses.append(train_loss)
            train_accuracies.append(train_acc)
            test_losses.append(test_loss)
            test_accuracies.append(test_acc)
            writer.add_scalar("Loss/Train", train_loss, epoch)
            writer.add_scalar("Loss/Test", test_loss, epoch)
            writer.add_scalar("Accuracy/Train", train_acc, epoch)
            writer.add_scalar("Accuracy/Test", test_acc, epoch)
            from tqdm.auto import tqdm
            tqdm.write(f"Epoch {epoch + 1} completed.")
    except KeyboardInterrupt:
        print("\n⛔ Training interrupted by user.")
    finally:
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        torch.save(resnet18.state_dict(), str(MODEL_PATH))
        print(f"💾 Final model saved to {MODEL_PATH}")
        writer.close()
        train_time_end_model = timer()
        total_train_time_model = print_train_time(start=train_time_start_model, end=train_time_end_model, device=device)
        print(f"🏁 Training completed in {total_train_time_model}")

        def plot_loss_curves(train_loss, test_loss):
            plt.plot(train_loss, label='Train Loss')
            plt.plot(test_loss, label='Test Loss')
            plt.title('Loss Curves')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.legend()
            plt.show()

        def plot_accuracy_curves(train_acc, test_acc):
            plt.plot(train_acc, label='Train Accuracy')
            plt.plot(test_acc, label='Test Accuracy')
            plt.title('Accuracy Curves')
            plt.xlabel('Epochs')
            plt.ylabel('Accuracy')
            plt.legend()
            plt.show()

        plot_loss_curves(train_losses, test_losses)
        plot_accuracy_curves(train_accuracies, test_accuracies)

        def generate_classification_report(model, data_loader, device, class_names):
            y_true = []
            y_pred = []
            model.eval()
            with torch.inference_mode():
                for X, y in data_loader:
                    X, y = X.to(device).float(), y.to(device)
                    preds = model(X)
                    y_true.extend(y.cpu().numpy())
                    y_pred.extend(preds.argmax(dim=1).cpu().numpy())
            print(classification_report(y_true, y_pred))
            cm = confusion_matrix(y_true, y_pred)
            plt.figure(figsize=(12, 10))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
            plt.xlabel("Predicted Label")
            plt.ylabel("True Label")
            plt.title("Confusion Matrix")
            plt.tight_layout()
            plt.show()

        print("Generating classification report and confusion matrix...")
        generate_classification_report(resnet18, test_dataloader, device, train_data.classes)

        def load_model(model, model_path, device):
            model.load_state_dict(torch.load(model_path, map_location=device), strict=True)
            model.to(device)
            return model

        loaded_resnet18 = load_model(resnet18, MODEL_PATH, device)
        print(f"✅ Model loaded from {MODEL_PATH}")

        def evaluate_model(model, data_loader, loss_fn, accuracy_fn, device):
            model.eval()
            total_loss = 0
            total_correct = 0
            total_samples = 0
            with torch.inference_mode():
                for X, y in data_loader:
                    X, y = X.to(device).float(), y.to(device)
                    y_pred = model(X)
                    loss = loss_fn(y_pred, y)
                    total_loss += loss.item()
                    total_correct += (y_pred.argmax(dim=1) == y).sum().item()
                    total_samples += len(y)
            avg_loss = total_loss / len(data_loader)
            acc = total_correct / total_samples * 100
            return avg_loss, acc

        loaded_model_loss, loaded_model_acc = evaluate_model(loaded_resnet18, test_dataloader, loss_fn, accuracy_fn, device)
        print(f"Loaded Model - Test Loss: {loaded_model_loss:.4f} | Test Acc: {loaded_model_acc:.2f}%")