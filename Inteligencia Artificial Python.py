import os
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from tqdm import tqdm
from PIL import Image

# Parte 1: Creación del conjunto de datos

class ConjuntoDeDatosPersonalizado(Dataset):
    def __init__(self, rutas_archivos, etiquetas, transformaciones=None):
        self.rutas_archivos = rutas_archivos
        self.etiquetas = etiquetas
        self.transformaciones = transformaciones

    def __len__(self):
        return len(self.rutas_archivos)

    def __getitem__(self, indice):
        ruta_imagen = self.rutas_archivos[indice]
        imagen = Image.open(ruta_imagen).convert('RGB')
        etiqueta = torch.tensor(self.etiquetas[indice], dtype=torch.long)

        if self.transformaciones:
            imagen = self.transformaciones(imagen)
        
        return imagen, etiqueta

# Parte 2: Creación de la red neuronal

class RedNeuronal(nn.Module):
    def __init__(self, num_clases):
        super(RedNeuronal, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3,3), stride=2)
        self.conv2 = torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3,3), stride=2)
        self.conv3 = torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3,3), stride=2)
        self.aplanar = torch.nn.Flatten()
        self.fc1 = torch.nn.Linear(128*27*27, 128)
        self.fc2 = torch.nn.Linear(128, num_clases)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.aplanar(x)
        x = F.relu(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x

# Parte 3: Configuración de hiperparámetros y carga de datos

velocidad_aprendizaje = 1e-4
lote_tamano = 32
numero_epocas = 10

transformaciones = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
])

conjunto_entrenamiento = ImageFolder(root='pruebas/dataset/train', transform=transformaciones)
conjunto_prueba = ImageFolder(root='pruebas/dataset/test', transform=transformaciones)

cargador_entrenamiento = DataLoader(conjunto_entrenamiento, batch_size=lote_tamano, shuffle=True)
cargador_prueba = DataLoader(conjunto_prueba, batch_size=lote_tamano, shuffle=False)

# Parte 4: Inicializar el modelo y el optimizador

num_clases = 3
modelo = RedNeuronal(num_clases)
perdida = nn.CrossEntropyLoss()
optimizador = optim.Adam(modelo.parameters(), lr=velocidad_aprendizaje)

# Parte 5: Entrenar al modelo

dispositivo = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
modelo.to(dispositivo)

for epoca in range(numero_epocas):
    modelo.train()
    perdida_acumulada = 0.0
    
    for entradas, etiquetas in tqdm(cargador_entrenamiento, desc=f'Época {epoca + 1}/{numero_epocas}'):
        entradas, etiquetas = entradas.to(dispositivo), etiquetas.to(dispositivo)
        
        optimizador.zero_grad()
        salidas = modelo(entradas)
        perdida_actual = perdida(salidas, etiquetas)
        perdida_actual.backward()
        optimizador.step()
        
        perdida_acumulada += perdida_actual.item()
    
    print(f'Pérdida de entrenamiento: {perdida_acumulada / len(cargador_entrenamiento)}')

# Parte 6: Evaluar el modelo

modelo.eval()
predicciones_correctas = 0
total_muestras = 0

with torch.no_grad():
    for entradas, etiquetas in tqdm(cargador_prueba, desc='Prueba'):
        entradas, etiquetas = entradas.to(dispositivo), etiquetas.to(dispositivo)
        salidas = modelo(entradas)
        _, predicciones = torch.max(salidas, 1)
        predicciones_correctas += (predicciones == etiquetas).sum().item()
        total_muestras += etiquetas.size(0)

precision = predicciones_correctas / total_muestras
print(f'Precisión del modelo: {precision * 100:.2f}%')

# Parte 7: Clasificación de imágenes individuales

def predecir_imagen(ruta_imagen):
    modelo.eval()
    with torch.no_grad():
        imagen = Image.open(ruta_imagen).convert('RGB')
        imagen = transformaciones(imagen).unsqueeze(0).to(dispositivo)
        salidas = modelo(imagen)
        _, clase_predicha = torch.max(salidas, 1)
        nombres_clases = conjunto_entrenamiento.classes
        etiqueta_predicha = nombres_clases[clase_predicha.item()]
        print(f'La imagen seleccionada fue reconocida como: {etiqueta_predicha}')

nueva_ruta_imagen = 'pruebas/imgtest/tcarro.jpg'
predecir_imagen(nueva_ruta_imagen)

nueva_ruta_imagen = 'pruebas/imgtest/tdiamante.jpg'
predecir_imagen(nueva_ruta_imagen)

nueva_ruta_imagen = 'pruebas/imgtest/tpato.jpeg'
predecir_imagen(nueva_ruta_imagen)