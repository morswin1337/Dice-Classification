# Użyj obrazu bazowego PyTorch
FROM pytorch/pytorch:latest

# Skopiuj kod aplikacji do kontenera
COPY Run_Gradio.py /app
COPY dice_cnn.pth /app

# Instalacja zależności Pythona
RUN pip install pandas numpy matplotlib gradio

# Otwórz port 7860
EXPOSE 7860

# Uruchom aplikację
CMD ["python", "Run_Gradio.py"]
