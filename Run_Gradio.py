import torch
import torch.nn as nn
import gradio as gr
from PIL import Image
import numpy as np

def cnn_model():
    return nn.Sequential(
        nn.Conv2d(1, 32, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),
        nn.Conv2d(32, 64, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),
        nn.Flatten(),
        nn.Linear(64 * 7 * 7, 128),
        nn.ReLU(),
        nn.Linear(128, 6) 
    )

def classify_image(image):
    if image is None:
        # Jeśli obraz nie został dostarczony, zwracamy wartość domyślną
        return "no image"
    
    if isinstance(image, Image.Image):
        image = image.convert('L')  # Konwertujemy na skalę szarości
        image = image.resize((28, 28))  # Zmieniamy rozmiar na 28x28
        image = np.array(image).astype(np.float32) / 255.0  # Normalizujemy
        image = torch.tensor(image).unsqueeze(0).unsqueeze(0)  # Konwertujemy na tensor, dodajemy wymiary batch i kanału
    else:
        # Jeśli image nie jest obrazem typu PIL, zwróć odpowiednią odpowiedź
        return -1

    model = cnn_model()
    model.load_state_dict(torch.load('dice_cnn.pth'))
    model.eval()
    
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        return predicted.item() + 1  # Dodajemy 1, ponieważ etykiety zostały zmniejszone o 1

# Tworzymy interfejs Gradio
iface = gr.Interface(
    fn=classify_image,
    inputs=gr.Image(type="pil"),  # Używamy gr.inputs.Image z type="pil"
    outputs=gr.Label(),
    live=True
)



# Uruchamiamy interfejs
iface.launch()
