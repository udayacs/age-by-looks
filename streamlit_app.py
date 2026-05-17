import streamlit as st
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from PIL import Image
import numpy as np
import torch.nn.functional as F

IMG_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AGE_LABELS = [
    "0-2", "3-9", "10-19", "20-29", "30-39",
    "40-49", "50-59", "60-69", "more than 70"
    ]


class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), # Input: 3 channels (RGB), Output: 32 channels
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # (IMG_SIZE/2) x (IMG_SIZE/2)

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # (IMG_SIZE/4) x (IMG_SIZE/4)

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  # (IMG_SIZE/8) x (IMG_SIZE/8)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            # Calculate the input features for the first linear layer
            # IMG_SIZE/8 * IMG_SIZE/8 * 128 (channels)
            nn.Linear(128 * (IMG_SIZE // 8) * (IMG_SIZE // 8), 1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
    
def initialize():
    
    BEST_MODEL_PATH = 'best_model.pt'
    NUM_AGE_CATEGORIES = len(AGE_LABELS)
    
    model = SimpleCNN(NUM_AGE_CATEGORIES).to(DEVICE)
    model.eval()
    map_location = torch.device('cpu') if not torch.cuda.is_available() else None
    checkpoint = torch.load(BEST_MODEL_PATH, map_location=map_location)
    model.load_state_dict(checkpoint['model_state_dict'])
    st.session_state["model"] = model
    
def predict_age(image):

    model = st.session_state["model"]
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.48263129591941833, 0.35801640152931213, 0.3045556843280792], std=[0.25709637999534607, 0.2242567241191864, 0.21827450394630432])
    ])
    pilImage = Image.open(image).convert('RGB')
    image = transform(pilImage).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        output = model(image)
        val, ind = torch.topk(output, k=3)
        probabilities = F.softmax(val[0], dim=0)
        print(f"Top 3 predictions: {[(AGE_LABELS[i], probabilities[j].item()) for j, i in enumerate(ind[0])]}")
        st.write(f"We have three guesses for your age category:")
        for i in range(3):
            st.write(f"Guess #{i+1}:&nbsp;&nbsp;&nbsp;**{AGE_LABELS[ind[0][i]]}**&nbsp;&nbsp;&nbsp;&nbsp;with probability:&nbsp;{probabilities[i]*100:.2f}%")


# --- INITIALIZE BLOCK ---
if "my_variable" not in st.session_state:
    st.session_state["my_variable"] = "Initial Value"
    initialize()

# --- APP CODE ---
st.image("banner.png", width=400)
st.title("Age By Looks")
st.write(
    " Please upload a photo of yourself, and we will guess your age."
)
uploaded_file = st.file_uploader("Choose a photo...", type=["jpg", "jpeg", "png"])  
if uploaded_file is not None:
    st.image(uploaded_file, caption='Uploaded Photo', width=200)
    st.write("Processing your photo...")
    predict_age(uploaded_file)
    st.write("Thank you for using the Age By Looks app!")
else:
    st.write("Please upload a photo to get started.")



