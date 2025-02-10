import rasterio
import numpy as np
import torch
import torch.nn.functional as F
from rasterio.windows import Window
from pathlib import Path
from tqdm import tqdm
from model import UNet

# class GeoTIFFPredictor:
#     def __init__(self, model, device, window_size=256, overlap=64):
#         self.model = model
#         self.device = device
#         self.window_size = window_size
#         self.overlap = overlap
#         self.stride = window_size - overlap
        
#         # Cria kernel de pesos para blend (Gaussian-like)
#         self.blend_weights = self.create_blend_weights(window_size, overlap)

#     @staticmethod
#     def create_blend_weights(size, overlap):
#         """Cria pesos para blend suave nas bordas"""
#         weights = np.ones((size, size), dtype=np.float32)
#         ramp = np.linspace(0, 1, overlap//2)
        
#         # Bordas horizontais
#         weights[:overlap//2, :] *= ramp[:, np.newaxis]
#         weights[-overlap//2:, :] *= ramp[::-1, np.newaxis]
        
#         # Bordas verticais
#         weights[:, :overlap//2] *= ramp[np.newaxis, :]
#         weights[:, -overlap//2:] *= ramp[np.newaxis, ::-1]
        
#         return torch.from_numpy(weights).to(device).cpu()

#     def normalize_band(self, band):
#         """Normalização individual de cada banda (igual ao dataset)"""
#         min_val = band.min()
#         max_val = band.max()
#         if max_val > min_val:
#             return (band - min_val) / (max_val - min_val)
#         return np.zeros_like(band, dtype=np.float32)

#     def predict_geotiff(self, input_path, output_path, return_probs=True):
#         """Executa a predição completa em um arquivo GeoTIFF"""
#         with rasterio.open(input_path) as src:
#             # Cria matrizes para acumulação
#             full_pred = np.zeros((src.height, src.width), dtype=np.float32)
#             full_count = np.zeros((src.height, src.width), dtype=np.float32)
            
#             # Gera coordenadas das janelas
#             offsets = []
#             for y in range(0, src.height, self.stride):
#                 for x in range(0, src.width, self.stride):
#                     y_start = max(0, y)
#                     x_start = max(0, x)
#                     y_end = min(src.height, y_start + self.window_size)
#                     x_end = min(src.width, x_start + self.window_size)
#                     offsets.append((y_start, x_start, y_end, x_end))

#             # Processa cada janela
#             for y_start, x_start, y_end, x_end in tqdm(offsets, desc="Processando janelas"):
#                 # Lê a janela
#                 window = Window(x_start, y_start, x_end - x_start, y_end - y_start)
#                 chip = src.read(window=window).astype(np.float32)
                
#                 # Normaliza cada banda
#                 for c in range(chip.shape[0]):
#                     chip[c] = self.normalize_band(chip[c])
                
#                 # Converte para tensor e move para GPU
#                 input_tensor = torch.from_numpy(chip).unsqueeze(0).to(self.device)
                
#                 # Predição
#                 with torch.no_grad():
#                     output = self.model(input_tensor)
#                     pred = torch.sigmoid(output).squeeze().cpu().numpy()
                
#                 # Aplica pesos de blend
#                 h, w = pred.shape
#                 weights = self.blend_weights[:h, :w]
#                 weighted_pred = pred * weights.numpy()
                
#                 # Atualiza a matriz de acumulação
#                 full_pred[y_start:y_end, x_start:x_end] += weighted_pred
#                 full_count[y_start:y_end, x_start:x_end] += weights.numpy()

#             # Calcula média ponderada
#             full_pred = np.divide(full_pred, full_count, where=full_count>0)
            
#             # Binariza se necessário
#             if not return_probs:
#                 full_pred = (full_pred > 0.5).astype(np.uint8)

#             # Salva o resultado
#             self.save_geotiff(output_path, full_pred, src.profile, return_probs)

#     def save_geotiff(self, output_path, data, profile, return_probs):
#         """Salva o resultado mantendo a projeção original"""
#         # Se for probabilidade, atualiza count e dtype
#         if return_probs:
#             profile.update({
#                 'driver': 'GTiff',
#                 'height': data.shape[0],
#                 'width': data.shape[1],
#                 'count': 1,  # A probabilidade é salva em uma única camada
#                 'dtype': 'float32',  # Tipo de dado para probabilidade
#                 'nodata': None,
#                 'compress': 'lzw'
#             })
#         else:
#             # Se for classe binária, muda para inteiro
#             profile.update({
#                 'driver': 'GTiff',
#                 'height': data.shape[0],
#                 'width': data.shape[1],
#                 'count': 1,  # Uma camada para as classes binárias
#                 'dtype': 'int8',  # Tipo de dado para classe binária
#                 'nodata': None,
#                 'compress': 'lzw'
#             })
        
#         # Salva o arquivo com o perfil modificado
#         with rasterio.open(output_path, 'w', **profile) as dst:
#             dst.write(data, 1)

# if __name__ == "__main__":
#     # Configurações
#     checkpoint_path = "checkpoints/checkpoint_epoch_16_loss_0.0299.pth"
#     input_tif = "image/landsat2024_0 (4).tif"
#     output_tif = "image/landsat2024_0 (4)_pred.tif"
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     # Carrega o modelo
#     model = UNet(in_channels=6).to(device)  # Ajustar para número correto de bandas
#     model.load_state_dict(torch.load(checkpoint_path))
#     model.eval()

#     # Executa predição
#     predictor = GeoTIFFPredictor(model, device, window_size=256, overlap=32)
#     predictor.predict_geotiff(input_tif, output_tif, return_probs=True)



class GeoTIFFPredictor:
    def __init__(self, model, device, window_size=256, overlap=64):
        self.model = model
        self.device = device
        self.window_size = window_size
        self.overlap = overlap
        self.stride = window_size - overlap
        self.ramp = np.linspace(0, 1, overlap//2)

    def get_blend_weights(self, window_height, window_width, y_start, x_start, src_height, src_width):
        """Gera pesos adaptativos considerando bordas da imagem"""
        weights = np.ones((window_height, window_width), dtype=np.float32)
        
        # Aplica rampa apenas se não estiver na borda correspondente
        # Topo
        if y_start > 0:
            top_ramp = self.ramp[:, np.newaxis]
            weights[:self.overlap//2, :] *= top_ramp[:window_height, :]
        
        # Base
        if y_start + window_height < src_height:
            bottom_ramp = self.ramp[::-1, np.newaxis]
            weights[-self.overlap//2:, :] *= bottom_ramp[:window_height, :]
        
        # Esquerda
        if x_start > 0:
            left_ramp = self.ramp[np.newaxis, :]
            weights[:, :self.overlap//2] *= left_ramp[:, :window_width]
        
        # Direita
        if x_start + window_width < src_width:
            right_ramp = self.ramp[np.newaxis, ::-1]
            weights[:, -self.overlap//2:] *= right_ramp[:, :window_width]
            
        return torch.from_numpy(weights).to(self.device).cpu()

    def normalize_band(self, band):
        """Normalização individual de cada banda (igual ao dataset)"""
        min_val = band.min()
        max_val = band.max()
        if max_val > min_val:
            return (band - min_val) / (max_val - min_val)
        return np.zeros_like(band, dtype=np.float32)

    def predict_geotiff(self, input_path, output_path, return_probs=True):
        """Executa a predição completa em um arquivo GeoTIFF"""
        with rasterio.open(input_path) as src:
            full_pred = np.zeros((src.height, src.width), dtype=np.float32)
            full_count = np.zeros((src.height, src.width), dtype=np.float32)
            
            offsets = []
            # Gera coordenadas com overlap negativo para cobrir bordas
            for y in range(-self.overlap//2, src.height, self.stride):
                for x in range(-self.overlap//2, src.width, self.stride):
                    y_start = max(0, y)
                    x_start = max(0, x)
                    y_end = min(src.height, y_start + self.window_size)
                    x_end = min(src.width, x_start + self.window_size)
                    
                    # Apenas adiciona janelas válidas
                    if (y_end - y_start) > self.overlap//2 and (x_end - x_start) > self.overlap//2:
                        offsets.append((y_start, x_start, y_end, x_end))

            for y_start, x_start, y_end, x_end in tqdm(offsets, desc="Processando janelas"):
                window = Window(x_start, y_start, x_end - x_start, y_end - y_start)
                chip = src.read(window=window).astype(np.float32)
                
                # Normaliza cada banda
                for c in range(chip.shape[0]):
                    chip[c] = self.normalize_band(chip[c])
                
                # Converte para tensor
                input_tensor = torch.from_numpy(chip).unsqueeze(0).to(self.device)
                
                # Predição
                with torch.no_grad():
                    output = self.model(input_tensor)
                    pred = torch.sigmoid(output).squeeze().cpu().numpy()
                
                # Obtém dimensões reais
                h, w = pred.shape
                
                # Gera pesos adaptativos
                weights = self.get_blend_weights(
                    window_height=h,
                    window_width=w,
                    y_start=y_start,
                    x_start=x_start,
                    src_height=src.height,
                    src_width=src.width
                )
                
                # Aplica pesos
                weighted_pred = pred * weights.numpy()
                
                # Atualiza acumuladores
                full_pred[y_start:y_end, x_start:x_end] += weighted_pred
                full_count[y_start:y_end, x_start:x_end] += weights.numpy()

            # Calcula média ponderada
            full_pred = np.divide(full_pred, full_count, where=full_count>0)
            
            # Binarização
            if not return_probs:
                full_pred = (full_pred > 0.5).astype(np.uint8)

            # Salva resultado
            self.save_geotiff(output_path, full_pred, src.profile, return_probs)

    def save_geotiff(self, output_path, data, profile, return_probs):
        """Salva o resultado mantendo a projeção original"""
        profile.update({
            'driver': 'GTiff',
            'height': data.shape[0],
            'width': data.shape[1],
            'count': 1,
            'dtype': 'float32' if return_probs else 'uint8',
            'nodata': None,
            'compress': 'lzw'
        })
        
        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(data, 1)

if __name__ == "__main__":
    checkpoint_path = "checkpoints/checkpoint_epoch_16_loss_0.0299.pth"
    input_tif = "image/landsat2024_0 (4).tif"
    output_tif = "image/landsat2024_0 (4)_pred.tif"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = UNet(in_channels=6).to(device)
    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()

    predictor = GeoTIFFPredictor(model, device, window_size=256, overlap=64)
    predictor.predict_geotiff(input_tif, output_tif, return_probs=True)