import os
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torchaudio.transforms import MelSpectrogram
from torch.nn.functional import pad
import gradio as gr

# Configuración
AUDIO_DIR = os.path.join(".", "motors")  # Ruta al dataset descomprimido
print(f"Ruta del dataset: {AUDIO_DIR}")  # Depuración

SAMPLE_RATE = 16000
N_FFT = 1024
N_MELS = 128
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 0.001

# Clase para cargar el dataset
class EngineSoundDataset(Dataset):
    def __init__(self, audio_dir, transform=None):
        self.audio_dir = audio_dir
        self.transform = transform
        self.classes = ['V8', 'V12']  # Asegúrate de que los nombres coincidan con las carpetas
        self.filepaths = []
        self.labels = []

        for label, class_name in enumerate(self.classes):
            class_dir = os.path.join(audio_dir, class_name)
            if not os.path.exists(class_dir):
                print(f"Advertencia: La carpeta {class_dir} no existe.")
                continue
            
            # Buscar archivos .wav en las subcarpetas 'test' y 'train'
            for subdir in ['test', 'train']:
                subdir_path = os.path.join(class_dir, subdir)
                if not os.path.exists(subdir_path):
                    print(f"Advertencia: La subcarpeta {subdir_path} no existe.")
                    continue
                
                print(f"Archivos en {subdir_path}: {os.listdir(subdir_path)}")  # Depuración
                for filename in os.listdir(subdir_path):
                    if filename.endswith(".wav"):  # Solo procesar archivos .wav
                        self.filepaths.append(os.path.join(subdir_path, filename))
                        self.labels.append(label)
            
            print(f"Encontrados {len(self.filepaths)} archivos en {class_dir}")  # Depuración

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        audio_path = self.filepaths[idx]
        audio_path = os.path.normpath(audio_path)  # Normaliza la ruta
        print(f"Cargando archivo: {audio_path}")  # Depuración
        
        try:
            waveform, sample_rate = torchaudio.load(audio_path)
            print(f"Archivo cargado correctamente: {audio_path}")
        except Exception as e:
            print(f"Error al cargar {audio_path}: {e}")
            raise e
        
        # Convertir a mono si es estéreo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Cambiar la tasa de muestreo si es necesario
        if sample_rate != SAMPLE_RATE:
            transform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=SAMPLE_RATE)
            waveform = transform(waveform)
        
        # Aplicar transformación (espectrograma Mel)
        if self.transform:
            waveform = self.transform(waveform)
        
        label = self.labels[idx]
        return waveform, label

# Función para ajustar la longitud de los espectrogramas
def collate_fn(batch):
    # Encuentra la longitud máxima en el batch
    max_length = max(item[0].shape[2] for item in batch)
    
    # Rellena o recorta todos los espectrogramas para que tengan la misma longitud
    padded_batch = []
    for waveform, label in batch:
        if waveform.shape[2] < max_length:
            # Rellenar con ceros
            padding = (0, max_length - waveform.shape[2])
            waveform = pad(waveform, padding)
        else:
            # Recortar
            waveform = waveform[:, :, :max_length]
        padded_batch.append((waveform, label))
    
    # Apila los tensores
    waveforms = torch.stack([item[0] for item in padded_batch])
    labels = torch.tensor([item[1] for item in padded_batch])
    
    return waveforms, labels

# Transformación para convertir el audio en un espectrograma Mel
transform = MelSpectrogram(sample_rate=SAMPLE_RATE, n_fft=N_FFT, n_mels=N_MELS)

# Cargar el dataset
dataset = EngineSoundDataset(AUDIO_DIR, transform=transform)
print(f"Total de muestras en el dataset: {len(dataset)}")

# Verificar si hay muestras
if len(dataset) == 0:
    raise ValueError("No se encontraron archivos de audio válidos en el dataset.")

# Dividir en entrenamiento y prueba
train_data, test_data = train_test_split(dataset, test_size=0.2, random_state=42)

# Usar el collate_fn personalizado en el DataLoader
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

# Definir el modelo
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.relu = nn.ReLU()  # Definir self.relu antes de usarlo
        
        # Calcular el tamaño de la salida después de las capas convolucionales y de pooling
        self.fc1_input_size = self._calculate_fc1_input_size()
        
        self.fc1 = nn.Linear(self.fc1_input_size, 128)
        self.fc2 = nn.Linear(128, 2)  # 2 clases: V8 y V12
        self.dropout = nn.Dropout(0.5)

    def _calculate_fc1_input_size(self):
        # Crear un tensor de prueba con las dimensiones esperadas del espectrograma
        # [batch_size, 1, n_mels, time_steps]
        x = torch.randn(1, 1, N_MELS, 181)  # Usar el tamaño real del espectrograma
        x = self.pool(self.relu(self.conv1(x)))  # [batch_size, 32, 64, 90]
        x = self.pool(self.relu(self.conv2(x)))  # [batch_size, 64, 32, 45]
        return x.view(x.size(0), -1).size(1)  # Aplanar y obtener el tamaño

    def forward(self, x):
        print(f"Input shape: {x.shape}")  # Verificar la forma de entrada
        x = self.pool(self.relu(self.conv1(x)))  # [batch_size, 32, 64, 90]
        print(f"After conv1 and pool1: {x.shape}")  # Verificar la forma después de conv1 y pool1
        x = self.pool(self.relu(self.conv2(x)))  # [batch_size, 64, 32, 45]
        print(f"After conv2 and pool2: {x.shape}")  # Verificar la forma después de conv2 y pool2
        x = x.view(x.size(0), -1)  # [batch_size, 64 * 32 * 45 = 92160]
        print(f"After flattening: {x.shape}")  # Verificar la forma después de aplanar
        x = self.relu(self.fc1(x))
        print(f"After fc1: {x.shape}")  # Verificar la forma después de fc1
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Instanciar el modelo, la función de pérdida y el optimizador
model = CNNModel()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Entrenamiento
def train(model, train_loader, criterion, optimizer, epochs):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader)}")

# Evaluación
def evaluate(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    print(f"Accuracy on test set: {accuracy:.2f}%")

# Entrenar y evaluar el modelo
train(model, train_loader, criterion, optimizer, EPOCHS)
evaluate(model, test_loader)

# Guardar el modelo entrenado
torch.save(model.state_dict(), "engine_sound_model.pth")

# Función para preprocesar el audio y hacer la predicción
def predict_engine(audio_path):
    try:
        # Cargar el archivo de audio
        waveform, sample_rate = torchaudio.load(audio_path)
        
        # Convertir a mono si es estéreo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Cambiar la tasa de muestreo si es necesario
        if sample_rate != SAMPLE_RATE:
            resample = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=SAMPLE_RATE)
            waveform = resample(waveform)
        
        # Aplicar transformación (espectrograma Mel)
        waveform = transform(waveform)
        
        # Ajustar la longitud del espectrograma
        target_length = 181  # Longitud del espectrograma utilizada durante el entrenamiento
        if waveform.shape[2] < target_length:
            # Rellenar con ceros
            padding = (0, target_length - waveform.shape[2])
            waveform = torch.nn.functional.pad(waveform, padding)
        else:
            # Recortar
            waveform = waveform[:, :, :target_length]
        
        # Verificar las dimensiones del espectrograma
        print(f"Espectrograma shape: {waveform.shape}")  # Debería ser [1, 128, 181]
        
        # Añadir una dimensión de batch
        waveform = waveform.unsqueeze(0)  # [1, 1, 128, 181]
        
        # Hacer la predicción
        with torch.no_grad():
            outputs = model(waveform)
            _, predicted = torch.max(outputs, 1)
        
        # Convertir la predicción a un nombre de clase
        class_names = ["V8", "V12"]
        prediction = class_names[predicted.item()]
        
        return prediction
    except Exception as e:
        return f"Error: {str(e)}"

# Crear la interfaz de Gradio
interface = gr.Interface(
    fn=predict_engine,  # Función que realiza la predicción
    inputs=gr.Audio(type="filepath"),  # Entrada: archivo de audio
    outputs="text",  # Salida: texto (V8 o V12)
    live=False,  # No actualizar en tiempo real
    title="V8 or V12",
    description="upload a .wav file to predict if the motor is a V8 or V12."
)

# Lanzar la interfaz
interface.launch()
