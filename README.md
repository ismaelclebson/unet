# UNet para segmentação semântica na agricultura
Este repositório contém a implementação de um modelo UNet para segmentação semântica aplicada à agricultura. O modelo foi desenvolvido para identificar e segmentar áreas de interesse em imagens de satélite, com foco em aplicações como mapeamento de culturas, detecção de áreas de plantio e monitoramento de vegetação.

# Visão Geral
O projeto utiliza imagens multiespectrais (7 bandas do Landsat) e máscaras de ground truth para treinar um modelo UNet capaz de realizar segmentação semântica em imagens agrícolas. O modelo é treinado para identificar áreas específicas de interesse, como plantações, solo exposto e vegetação.

## Funcionalidades Principais
Modelo UNet: Arquitetura de rede neural convolucional para segmentação semântica.

Data Augmentation: Técnicas de aumento de dados, como rotação e flip, para melhorar a generalização do modelo.

Treinamento com Checkpoints: Salvamento automático dos melhores modelos durante o treinamento.

Métricas de Avaliação: Uso de métricas como Jaccard Index (IoU) para avaliação do desempenho do modelo.

Integração com PyTorch: Implementação utilizando o framework PyTorch para flexibilidade e desempenho.

# Requisitos
Python 3.11

PyTorch 2.4.0+ (com suporte a CUDA, se disponível)

Então recomendamos [usar mamba](https://mamba.readthedocs.io/en/latest/installation/mamba-installation.html)
para instalar as dependências.

    mamba env create --file environment.yml

> [!NOTE]
> O comando acima foi testado em dispositivos Linux com GPUs CUDA.

Para executar o script de treinamento.

    python -m src.scripts.train