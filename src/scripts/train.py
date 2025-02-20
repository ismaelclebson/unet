"""
Training Script for U-Net Model with Checkpointing and Validation

This script trains a U-Net model for image segmentation tasks. It supports:
- Loading datasets for training and validation.
- Training with a custom loss function combining Focal Loss and Jaccard Loss.
- Saving the best checkpoints based on validation loss.
- Resuming training from the last checkpoint if interrupted.

Usage:
    Run the script to start training. Ensure the dataset paths and hyperparameters are correctly configured.

Dependencies:
    - torch
    - torchmetrics
    - segmentation_models_pytorch
    - Custom modules: UNet, JointLoss, ModelCheckpoint, and dataset utilities.

Hyperparameters:
    - BATCH_SIZE: Batch size for training and validation.
    - LEARNING_RATE: Learning rate for the optimizer.
    - NUM_EPOCHS: Total number of training epochs.
    - DEVICE: Device to use for training (CUDA if available, else CPU).
    - CHECKPOINT_DIR: Directory to save model checkpoints.

Example:
    python train.py
"""


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
from src.models.model import UNet
from src.data.uniqueDataset import get_dataloaders
from torchmetrics.classification import F1Score, BinaryJaccardIndex
import segmentation_models_pytorch as smp
from src.scripts.utils import JointLoss, ModelCheckpoint


# Hiperparâmetros
BATCH_SIZE = 8
LEARNING_RATE = 0.0001 # 0.001 (antes)
NUM_EPOCHS = 100
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_DIR = "src/models/checkpoints"  # Pasta para salvar os pesos

# Criar pasta para checkpoints
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Função para salvar o último checkpoint
def save_last_checkpoint(model, optimizer, epoch, checkpoint_dir):
    checkpoint_path = os.path.join(checkpoint_dir, "checkpoint_last.pth")
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, checkpoint_path)



# Carregar dados
train_loader, val_loader = get_dataloaders(
    train_img_dir="/home/clebson/Documentos/dataset_palmls4claymodel/data/train/aug/img/",
    train_gt_dir="/home/clebson/Documentos/dataset_palmls4claymodel/data/train/aug/gt/",
    val_img_dir="/home/clebson/Documentos/dataset_palmls4claymodel/data/val/aug/img/",
    val_gt_dir="/home/clebson/Documentos/dataset_palmls4claymodel/data/val/aug/gt/",
    batch_size=BATCH_SIZE
)

# Inicializar modelo, função de perda e otimizador
model = UNet(in_channels=6).to(DEVICE)
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
#criterion = smp.losses.FocalLoss(mode="binary").to(DEVICE)

# Inicializa as funções de perda
focal_loss = smp.losses.FocalLoss(mode="binary")
#dice_loss = smp.losses.DiceLoss(mode="binary")
iou_loss = smp.losses.JaccardLoss(mode="binary")

# Combina as perdas com pesos iguais
criterion = JointLoss(focal_loss, iou_loss, weight1=0.7, weight2=0.3).to(DEVICE)

metric = BinaryJaccardIndex().to(DEVICE)

# Verificar se existe último checkpoint
start_epoch = 0  # Época inicial padrão
last_checkpoint_path = os.path.join(CHECKPOINT_DIR, "checkpoint_last.pth")
if os.path.exists(last_checkpoint_path):
    checkpoint = torch.load(last_checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']
    print(f"Carregado último checkpoint da época {start_epoch}")

# Inicializar o callback de checkpoint
checkpoint_callback = ModelCheckpoint(CHECKPOINT_DIR, max_saves=1)

# Treinamento
best_val_loss = float("inf")
for epoch in range(start_epoch, NUM_EPOCHS):  # Alterado para começar da start_epoch
    model.train()
    train_loss = 0.0
    train_acc = 0.0
    for images, masks in train_loader:
        images, masks = images.to(DEVICE), masks.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_acc += metric(outputs, masks).item()

    avg_train_loss = train_loss / len(train_loader)
    avg_train_acc = train_acc / len(train_loader)

    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Train Jaccard: {avg_train_acc:.4f}, Train Loss: {avg_train_loss:.4f}")

    # Validação
    model.eval()
    val_loss = 0.0
    jaccard_score = 0.0
    with torch.no_grad():
        for images, masks in val_loader:
            images, masks = images.to(DEVICE), masks.to(DEVICE)
            outputs = model(images)
            val_loss += criterion(outputs, masks).item()
            jaccard_score += metric(outputs, masks).item()

    avg_val_loss = val_loss / len(val_loader)
    avg_val_acc = jaccard_score / len(val_loader)

    print(f"Validation Jaccard: {avg_val_acc}, Validation Loss: {avg_val_loss:.4f}")

    # Salvar checkpoint se for um dos dois melhores
    checkpoint_callback(model, avg_val_acc, avg_val_loss, epoch + 1)

    # Salvar último checkpoint após cada época
    save_last_checkpoint(model, optimizer, epoch + 1, CHECKPOINT_DIR)

# Carregar o melhor modelo ao final do treinamento (opcional)
best_checkpoint = sorted(checkpoint_callback.best_losses)[0][1]
model.load_state_dict(torch.load(best_checkpoint))
print(f"Melhor modelo carregado: {best_checkpoint}")