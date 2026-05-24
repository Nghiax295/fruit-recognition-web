import os
import numpy as np
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
from PIL import Image
import tensorflow as tf
import torch
import torch.nn as nn
from torchvision import transforms, models

#====================
# EfficientNet-B0
#====================
import torch.nn as nn
import torch

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=4):
        super(SEBlock, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction),
            Swish(),
            nn.Linear(in_channels // reduction, in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class MBConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, expand_ratio, stride, kernel_size):
        super(MBConvBlock, self).__init__()
        hidden_dim = in_channels * expand_ratio
        self.use_residual = (stride == 1 and in_channels == out_channels)
        layers = []
        if expand_ratio != 1:
            layers.extend([
                nn.Conv2d(in_channels, hidden_dim, 1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                Swish()
            ])
        layers.extend([
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, kernel_size//2, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            Swish()
        ])
        layers.append(SEBlock(hidden_dim))
        layers.extend([
            nn.Conv2d(hidden_dim, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        ])
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_residual:
            return x + self.block(x)
        else:
            return self.block(x)

class CustomEfficientNet(nn.Module):
    def __init__(self, num_classes=10):
        super(CustomEfficientNet, self).__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            Swish()
        )
        self.blocks = nn.Sequential(
            MBConvBlock(32, 16, 1, 1, 3),
            MBConvBlock(16, 24, 6, 2, 3),
            MBConvBlock(24, 40, 6, 2, 5),
            MBConvBlock(40, 80, 6, 2, 3),
            MBConvBlock(80, 112, 6, 1, 5),
            MBConvBlock(112, 192, 6, 2, 5),
            MBConvBlock(192, 320, 6, 1, 3)
        )
        self.head = nn.Sequential(
            nn.Conv2d(320, 1280, 1, bias=False),
            nn.BatchNorm2d(1280),
            Swish(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(1280, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

#====================
# MobileNetV2
#====================
class Block(tf.keras.Model):
    """Inverted residual block: Expand → Depthwise → Pointwise."""
    def __init__(self, in_channels, out_channels, expansion, strides):
        super().__init__()
        self.strides = strides
        ch = expansion * in_channels

        self.conv1 = tf.keras.layers.Conv2D(ch, 1, use_bias=False)
        self.bn1   = tf.keras.layers.BatchNormalization()

        self.conv2 = tf.keras.layers.Conv2D(ch, 3, strides=strides, padding='same',
                                            groups=ch, use_bias=False)
        self.bn2   = tf.keras.layers.BatchNormalization()

        self.conv3 = tf.keras.layers.Conv2D(out_channels, 1, use_bias=False)
        self.bn3   = tf.keras.layers.BatchNormalization()

        if strides == 1 and in_channels != out_channels:
            self.shortcut = tf.keras.Sequential([
                tf.keras.layers.Conv2D(out_channels, 1, use_bias=False),
                tf.keras.layers.BatchNormalization(),
            ])
        else:
            self.shortcut = lambda x: x

    def call(self, x, training=False):
        out = tf.nn.relu6(self.bn1(self.conv1(x), training=training))
        out = tf.nn.relu6(self.bn2(self.conv2(out), training=training))
        out = self.bn3(self.conv3(out), training=training)
        if self.strides == 1:
            out = tf.keras.layers.add([self.shortcut(x), out])
        return out


class MobileNetV2(tf.keras.Model):
    # (expansion, out_channels, num_blocks, stride)
    _cfg = [
        (1,  16, 1, 1),
        (6,  24, 2, 2),
        (6,  32, 3, 2),
        (6,  64, 4, 2),
        (6,  96, 3, 1),
        (6, 160, 3, 2),
        (6, 320, 1, 1),
    ]

    def __init__(self, num_classes=10, **kwargs):
        super().__init__(**kwargs)
        self.conv1   = tf.keras.layers.Conv2D(32, 3, strides=2, padding='same', use_bias=False)
        self.bn1     = tf.keras.layers.BatchNormalization()
        self.blocks  = self._make_layers(32)
        self.conv2   = tf.keras.layers.Conv2D(1280, 1, use_bias=False)
        self.bn2     = tf.keras.layers.BatchNormalization()
        self.gap     = tf.keras.layers.GlobalAveragePooling2D()
        self.dropout = tf.keras.layers.Dropout(0.3)
        self.fc      = tf.keras.layers.Dense(num_classes, activation='softmax')

    def call(self, x, training=False):
        out = tf.nn.relu6(self.bn1(self.conv1(x), training=training))
        out = self.blocks(out, training=training)
        out = tf.nn.relu6(self.bn2(self.conv2(out), training=training))
        out = self.gap(out)
        out = self.dropout(out, training=training)
        return self.fc(out)

    def _make_layers(self, in_ch):
        layers = []
        for exp, out_ch, n, s in self._cfg:
            for stride in ([s] + [1] * (n - 1)):
                layers.append(Block(in_ch, out_ch, exp, stride))
                in_ch = out_ch
        return tf.keras.Sequential(layers)

# ====================
# ResNet-34 (PyTorch)
# ====================
from torchvision.models import ResNet34_Weights
class LabelSmoothingCE(nn.Module):
    def __init__(self, classes, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
        self.cls = classes
    def forward(self, pred, target):
        confidence = 1.0 - self.smoothing
        smooth_val  = self.smoothing / (self.cls - 1)
        one_hot = torch.zeros_like(pred).scatter_(1, target.unsqueeze(1), 1)
        smooth_lbl = one_hot * confidence + (1 - one_hot) * smooth_val
        log_prob = nn.functional.log_softmax(pred, dim=1)
        return -(smooth_lbl * log_prob).sum(dim=1).mean()

def build_resnet34(num_classes: int, pretrained: bool = True) -> nn.Module:
    weights = ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.resnet34(weights=weights)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(in_features, num_classes),
    )
    return model

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# 10 lớp quả
CLASS_NAMES = ['apple', 'cantaloupe', 'cherry', 'lemon', 'lime', 'mandarine', 'orange', 'pear', 'plum', 'watermelon']


IMG_SIZE = 224

#====================
# Load Model
#====================
# Keras models
mobilenet_model = None
resnet_model = None
# PyTorch model
efficientnet_model = None
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_keras_model(path, model_type):
    if not os.path.exists(path):
        print(f"Warning: {path} not found.")
        return None
    try:
        if model_type == 'MobileNet':
            model = tf.keras.models.load_model(
                path,
                custom_objects={'Block': Block, 'MobileNetV2': MobileNetV2},
                compile=False  # Quan trọng: tránh lỗi optimizer state
            )
        else:
            model = tf.keras.models.load_model(path, compile=False)
        print(f"Loaded {model_type} from {path}")
        return model
    except Exception as e:
        print(f"Error loading {model_type} from {path}: {str(e)}")
        return None

def load_efficientnet():
    path = 'models/best_custom_efficientnet.pth'   # giữ nguyên đường dẫn file của bạn
    if not os.path.exists(path):
        print("Warning: EfficientNet model not found.")
        return None
    try:
        model = CustomEfficientNet(num_classes=len(CLASS_NAMES))
        state_dict = torch.load(path, map_location=device)
        model.load_state_dict(state_dict)
        model = model.to(device)
        model.eval()
        print(f"Loaded CustomEfficientNet from {path}")
        return model
    except Exception as e:
        print(f"Error loading EfficientNet model: {e}")
        return None

def load_resnet34():
    path = 'models/resnet34_scratch_final.pt'
    if not os.path.exists(path):
        print("ResNet34 model file not found. ResNet will be disabled.")
        return None
    try:
        # File này là TorchScript, load trực tiếp
        model = torch.jit.load(path, map_location=device)
        model = model.to(device)
        model.eval()
        print(f"Loaded ResNet34 (TorchScript) from {path}")
        return model
    except Exception as e:
        print(f"❌ Error loading ResNet34: {e}")
        return None

# Hàm tiền xử lý ảnh cho Keras
def preprocess_keras(img, model_type='mobilenet'):
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Tiền xử lý cho PyTorch EfficientNet
transform_efficientnet = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Transform cho ResNet (giống EfficientNet)
transform_resnet = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def predict_mobilenet(img):
    global mobilenet_model
    if mobilenet_model is None:
        return "MobileNet model not loaded"
    processed = preprocess_keras(img, 'mobilenet')
    preds = mobilenet_model.predict(processed, verbose=0)
    idx = np.argmax(preds[0])
    return CLASS_NAMES[idx]

def predict_resnet(img):
    global resnet_model
    if resnet_model is None:
        return "ResNet model not loaded"
    img_tensor = transform_resnet(img).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = resnet_model(img_tensor)
        _, predicted = torch.max(outputs, 1)
        return CLASS_NAMES[predicted.item()]

def predict_efficientnet(img):
    global efficientnet_model
    if efficientnet_model is None:
        return "Efficient model not loaded"
    img_tensor = transform_efficientnet(img).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = efficientnet_model(img_tensor)
        _, predicted = torch.max(outputs, 1)
        return CLASS_NAMES[predicted.item()]

# Load models khi khởi động
mobilenet_model = load_keras_model('models/mobilenetv2_fruit_final.keras', 'MobileNet')
resnet_model = load_resnet34()
efficientnet_model = load_efficientnet()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400
    model_choice = request.form.get('model', 'mobilenet')

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    img = Image.open(filepath).convert('RGB')

    if model_choice == 'mobilenet':
        result = predict_mobilenet(img)
    elif model_choice == 'resnet':
        result = predict_resnet(img)
    elif model_choice == 'efficientnet':
        result = predict_efficientnet(img)
    else:
        result = 'Unknown model'

    os.remove(filepath)
    return jsonify({'prediction': result})

# if __name__ == '__main__':
#     app.run(debug=True, host='0.0.0.0', port=5000)

if __name__ == '__main__':
    # Lấy cổng từ biến môi trường (Render sẽ cung cấp), mặc định 5000 cho local
    port = int(os.environ.get('PORT', 5000))
    # Tắt debug khi chạy production, bật khi cần debug local
    debug_mode = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    app.run(debug=debug_mode, host='0.0.0.0', port=port)