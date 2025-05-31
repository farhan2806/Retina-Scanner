import streamlit as st
from PIL import Image
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision import models

slope = pickle.load(open("slope.pkl","rb"))
intercept = pickle.load(open("intercept.pkl","rb"))
 
def point_position_parallel(m, c, point):
    x, y = point
    y_on_line = m * x + c

    if y > y_on_line:
        return "Not autistic"
    elif y < y_on_line:
        return "Autistic"
    else:
        return "Not autistic"

class RetinaModel(nn.Module):
    def __init__(self):
        super(RetinaModel, self).__init__()
        self.resnet = models.resnet18(pretrained=True)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 4)  # 4 outputs for height/width of optic disc and cup
    
    def forward(self, x):
        return self.resnet(x)
# transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor()
# ])

st.title("Upload an Image")

# Upload image file
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if st.button("Check Autism"):
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Fundus Image", use_container_width=True)
        image = Image.open(uploaded_file).convert("RGB")
        
        transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Resize to match model input
            transforms.ToTensor(),  # Convert to tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        
        image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
        
        model = RetinaModel()
        model.load_state_dict(torch.load('retina_model.pth'))
        model.eval()

        with torch.no_grad():
            output = model(image_tensor)
            disc_height, disc_width, cup_height, cup_width = abs(output[0])
            disc_diameter = (disc_height + disc_width) / 2
            cup_diameter = (cup_height + cup_width) / 2

            result1 = point_position_parallel(slope, intercept, (disc_diameter, cup_diameter))
            st.text(result1)
    else:
        st.text("Please upload the image")

