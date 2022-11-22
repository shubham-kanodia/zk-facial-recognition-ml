import torch

from inference.embeddings import Generator
from PIL import Image

from training.model import ClassifierModel

# Pick any sample image
image = Image.open("../data/processed-images/subject08.happy.jpeg").convert("RGB")

# Generate embeddings
generator = Generator()
embeds = generator.generate(image)

# Load model
model = ClassifierModel()
model.load_state_dict(torch.load("../model/face_classification_model.pt"))
model.eval()

# Inference
output = model.forward(torch.Tensor(embeds).unsqueeze(0)).argmax().detach().item()
print(output + 1)
