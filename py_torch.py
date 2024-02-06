import torch
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt

model = models.resnet50(pretrained=True)
model.eval()


def predict_object_pytorch(image_path):
    img = Image.open(image_path)
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    img = transform(img)
    img = img.unsqueeze(0)

    with torch.no_grad():
        predictions = model(img)

    _, predicted_class = torch.max(predictions, 1)
    return predicted_class.item()


image_path = "screenshots/optical/fboptical/opticalfb_44.png"
result_pytorch = predict_object_pytorch(image_path)
print(result_pytorch)
