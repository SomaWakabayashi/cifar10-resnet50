import torch
import torchvision.transforms as transforms
from PIL import Image
import argparse
import os
import torchvision

def predict(image_path, model_path, device):
    # 1. Model Setup
    model = torchvision.models.resnet50(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, 10)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # 2. Transform (Test時と同じ正規化)
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # 3. Inference
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
        prob = torch.nn.functional.softmax(output, dim=1)

    print(f"Result: {classes[predicted[0]]} ({prob[0][predicted[0]]*100:.2f}%)")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, required=True)
    parser.add_argument('--model', type=str, required=True)
    args = parser.parse_args()
    predict(args.image, args.model, torch.device('cuda' if torch.cuda.is_available() else 'cpu'))