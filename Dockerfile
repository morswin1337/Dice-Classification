# Użyj obrazu bazowego PyTorch
FROM pytorch/pytorch:latest

# Skopiuj kod aplikacji do kontenera
COPY . /app
COPY dice.csv /app

# Ustaw katalog roboczy
WORKDIR /app

# Instalacja zależności Pythona
RUN pip install torch torchvision pandas numpy matplotlib gradio

# Otwórz port 7860
EXPOSE 7860

# Uruchom aplikację
CMD ["python", "Dice CNN.py"]
