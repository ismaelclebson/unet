import torch.nn as nn
import os
import torch

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