import torch
import torchvision.transforms as transforms
from PIL import Image
from model import cifar10cnn

# CIFAR-10 has 10 class names
CIFAR10_CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer', 
    'dog', 'frog', 'horse', 'ship', 'truck'
]

def predict_image(image_path, model_path='saved_models/cifar10_cnn.pth'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model architecture and weights
    model = cifar10cnn(num_classes=10).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Preprocess the input image to match CIFAR-100 dimensions and normalization
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5071, 0.4867, 0.4408], 
            std=[0.2675, 0.2565, 0.2761]
        )
    ])

    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(device) 

    # Run inference
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted_idx = torch.max(probabilities, 1)

    class_name = CIFAR10_CLASSES[predicted_idx.item()]
    print(f"Prediction: {class_name} (Confidence: {confidence.item() * 100:.2f}%)")

if __name__ == '__main__':
    print("Prediction script ready. Import or run with an image path.")
    predict_image('./images/dog.jpg')