import torch
from model import UNet
import matplotlib.pyplot as plt
from utils import get_dataloaders

# Carregar dados
train_loader, val_loader = get_dataloaders(
    train_img_dir="/home/clebson/Documentos/dataset_palmls4claymodel/data/train/img/",
    train_gt_dir="/home/clebson/Documentos/dataset_palmls4claymodel/data/train/gt/",
    val_img_dir="/home/clebson/Documentos/dataset_palmls4claymodel/data/val/img/",
    val_gt_dir="/home/clebson/Documentos/dataset_palmls4claymodel/data/val/gt/",
    batch_size=8  # Definindo o batch size como 16
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Carregar o modelo
model = UNet(in_channels=6).to(DEVICE)
model.load_state_dict(torch.load("checkpoints/checkpoint_epoch_77_loss_0.0549_c_dropout.pth"))

# Inferência em um batch de imagens de teste
model.eval()
with torch.no_grad():
    for test_images, test_masks in val_loader:
        test_images = test_images.to(DEVICE)
        test_masks = test_masks.to(DEVICE)
        outputs = model(test_images)
        predicted_masks = (outputs > 0.5).float()  # Binariza a saída
        break  # Para sair após a primeira iteração

print(test_masks.shape)

# Visualização
num_images = test_images.size(0)  # Número de imagens no batch
fig, axes = plt.subplots(num_images, 3, figsize=(12, num_images * 4))

for i in range(num_images):
    # Imagem Original
    axes[i, 0].imshow(test_images[i, :3, :, :].cpu().permute(1, 2, 0))
    axes[i, 0].set_title("Imagem Original")
    axes[i, 0].axis('off')

    # Máscara Real
    axes[i, 1].imshow(test_masks[i].squeeze().cpu(), cmap="gray")
    axes[i, 1].set_title("Máscara Real")
    axes[i, 1].axis('off')

    # Máscara Predita
    axes[i, 2].imshow(predicted_masks[i].squeeze().cpu(), cmap="gray")
    axes[i, 2].set_title("Máscara Predita")
    axes[i, 2].axis('off')

plt.tight_layout()
plt.show()

