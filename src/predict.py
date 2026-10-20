import torch
import torchvision.transforms as transforms
from PIL import Image
from model import cifar100cnn

# CIFAR-100 has 100 class names
CIFAR100_CLASSES = [
    'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle',
    'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle',
    'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
    'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo', 'keyboard',
    'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain',
    'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree',
    'plain', 'plate', 'poppy', 'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket',
    'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider',
    'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank', 'telephone', 'television', 'tiger', 'tractor',
    'train', 'trout', 'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm'
]

def predict_image(image_path, model_path='saved_models/cifar100_cnn.pth'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model architecture and weights
    model = cifar100cnn(num_classes=100).to(device)
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

    class_name = CIFAR100_CLASSES[predicted_idx.item()]
    print(f"Prediction: {class_name} (Confidence: {confidence.item() * 100:.2f}%)")

if __name__ == '__main__':
    print("Prediction script ready. Import or run with an image path.")
    predict_image('./images/apple.jpg')