import torch
from torchvision import models, transforms
from PIL import Image
import requests

# 事前に訓練されたResNet18モデルをダウンロード
model = models.resnet18(pretrained=True)
model.eval()

# 画像の前処理
def preprocess_image(image_path):
    input_image = Image.open(image_path)
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0)
    return input_batch

# クラスのラベルを予測
def classify_image(image_path):
    input_tensor = preprocess_image(image_path)
    with torch.no_grad():
        output = model(input_tensor)

    # モデルの出力からクラスのラベルを取得
    _, predicted_idx = torch.max(output, 1)
    
    # ImageNetのクラスインデックスからラベルへの変換
    labels_url = 'https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json'
    labels = requests.get(labels_url).json()
    predicted_label = labels[predicted_idx.item()]

    return predicted_label

# 画像のパス
image_path = 'raion_002.jpeg'

# 画像を分類
predicted_class = classify_image(image_path)
print(f'The image is classified as: {predicted_class}')

