import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class UniqueDataset(Dataset):
    def __init__(self, img_dir, gt_dir, transform=None):
        """
        Construtor do dataset.
        :param img_dir: Diretório com as imagens de entrada (7 bandas do Landsat).
        :param gt_dir: Diretório com as máscaras de ground truth.
        :param transform: Transformações a serem aplicadas.
        """
        self.img_dir = img_dir
        self.gt_dir = gt_dir
        self.transform = transform
        self.img_names = os.listdir(img_dir)  # Lista de nomes das imagens


    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        # Caminhos para a imagem e a máscara
        img_path = os.path.join(self.img_dir, self.img_names[idx])
        gt_path = os.path.join(self.gt_dir, self.img_names[idx])  # Assume que as máscaras são .png

        # Carrega as 7 bandas do Landsat (salvas como .npy)
        image = np.load(img_path).astype(np.float32)  # (7, 256, 256)
        #print(image.shape)

        # Normaliza cada banda individualmente para [0, 255]
        for i in range(image.shape[0]):  # Itera sobre cada banda
            min_val = image[i].min()
            max_val = image[i].max()
            if max_val > min_val:  # Evita divisão por zero
                image[i] = ((image[i] - min_val)) / (max_val - min_val)
            else:
                image[i] = 0  # Se min == max, define como 0


        #print(image.shape)

        # Carrega a máscara de ground truth (1 canal, 256x256)
        mask = np.load(gt_path).astype(np.uint8)  # (1, 256, 256)
        #print(mask.shape)

        # Aplica transformações, se fornecidas
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        # Converte a máscara para binária (0 ou 1)
        #mask = (mask > 0).float()

        return torch.from_numpy(image), torch.from_numpy(mask)

def get_dataloaders(train_img_dir, train_gt_dir, val_img_dir, val_gt_dir, batch_size=8, n_workers=4):
    """
    Cria dataloaders para treino e validação.
    :param train_img_dir: Diretório das imagens de treino.
    :param train_gt_dir: Diretório das máscaras de treino.
    :param val_img_dir: Diretório das imagens de validação.
    :param val_gt_dir: Diretório das máscaras de validação.
    :param batch_size: Tamanho do batch.
    :return: Dataloaders para treino e validação.
    """
    # Transformação para converter numpy arrays em tensores
    transform = transforms.Compose([
        transforms.ToTensor(),  # Converte numpy array para tensor
    ])

    # Cria datasets para treino e validação
    train_dataset = UniqueDataset(train_img_dir, train_gt_dir)#, transform=transform)
    val_dataset = UniqueDataset(val_img_dir, val_gt_dir)#, transform=transform)

    # Cria dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=n_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=n_workers)

    return train_loader, val_loader