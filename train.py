import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
from model import UNet
from utils import get_dataloaders
from torchmetrics.classification import F1Score, BinaryJaccardIndex
import segmentation_models_pytorch as smp

# Hiperparâmetros
BATCH_SIZE = 8
LEARNING_RATE = 0.0001 # 0.001 (antes)
NUM_EPOCHS = 100
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_DIR = "checkpoints"  # Pasta para salvar os pesos

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

class ModelCheckpoint:
    def __init__(self, checkpoint_dir, max_saves=1):
        """
        Inicializa o callback para salvar apenas o melhor checkpoint.
        
        Args:
            checkpoint_dir (str): Diretório para salvar os checkpoints.
            max_saves (int): Número máximo de checkpoints a serem mantidos.
        """
        self.checkpoint_dir = checkpoint_dir
        self.max_saves = max_saves
        self.best_losses = []  # Lista de tuplas (val_loss, checkpoint_path)
        os.makedirs(checkpoint_dir, exist_ok=True)  # Cria o diretório se não existir

    def __call__(self, model, avg_val_acc, val_loss, epoch):
        """
        Salva apenas o melhor checkpoint e remove os anteriores.
        
        Args:
            model (torch.nn.Module): Modelo a ser salvo.
            val_loss (float): Valor da perda de validação.
            epoch (int): Número da época atual.
        """
        checkpoint_path = os.path.join(
            self.checkpoint_dir, f"checkpoint_epoch_{epoch}_acc_{avg_val_acc:.4f}_loss_{val_loss:.4f}.pth"
        )

        # Se ainda não há checkpoints, salva o primeiro
        if not self.best_losses:
            self.best_losses.append((val_loss, checkpoint_path))
            torch.save(model.state_dict(), checkpoint_path)
            return

        # Obtém a menor perda já salva
        best_loss, best_path = self.best_losses[0]

        if val_loss < best_loss:  # Apenas salva se for melhor
            # Remove o checkpoint anterior
            if os.path.exists(best_path):
                os.remove(best_path)
            
            # Atualiza para o novo melhor
            self.best_losses = [(val_loss, checkpoint_path)]
            torch.save(model.state_dict(), checkpoint_path)
            print(f'Salvando: "checkpoint_epoch_{epoch}_acc_{avg_val_acc:.4f}_loss_{val_loss:.4f}')

class JointLoss(nn.Module):
    def __init__(self, loss1, loss2, weight1=0.5, weight2=0.5):
        """
        Combina duas funções de perda com pesos personalizados.

        Args:
            loss1 (nn.Module): Primeira função de perda (ex: FocalLoss).
            loss2 (nn.Module): Segunda função de perda (ex: DiceLoss).
            weight1 (float): Peso para a primeira função de perda.
            weight2 (float): Peso para a segunda função de perda.
        """
        super(JointLoss, self).__init__()
        self.loss1 = loss1
        self.loss2 = loss2
        self.weight1 = weight1
        self.weight2 = weight2

    def forward(self, outputs, targets):
        # Calcula as duas perdas
        loss1 = self.loss1(outputs, targets)
        loss2 = self.loss2(outputs, targets)
        
        # Combina as perdas com os pesos
        total_loss = self.weight1 * loss1 + self.weight2 * loss2
        return total_loss


# Carregar dados
train_loader, val_loader = get_dataloaders(
    train_img_dir="/home/clebson/Documentos/dataset_palmls4claymodel/data/train/aug/img/",
    train_gt_dir="/home/clebson/Documentos/dataset_palmls4claymodel/data/train/aug/gt/",
    val_img_dir="/home/clebson/Documentos/dataset_palmls4claymodel/data/val/img/",
    val_gt_dir="/home/clebson/Documentos/dataset_palmls4claymodel/data/val/gt/",
    batch_size=BATCH_SIZE
)

# Inicializar modelo, função de perda e otimizador
model = UNet(in_channels=6).to(DEVICE)
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-2)
#criterion = smp.losses.FocalLoss(mode="binary").to(DEVICE)

# Inicializa as funções de perda
focal_loss = smp.losses.FocalLoss(mode="binary")
dice_loss = smp.losses.DiceLoss(mode="binary")

# Combina as perdas com pesos iguais
criterion = JointLoss(focal_loss, dice_loss, weight1=0.5, weight2=0.5).to(DEVICE)

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