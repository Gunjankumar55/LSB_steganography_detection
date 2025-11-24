from flask import Flask, render_template, request, jsonify
import torch
import torch.nn as nn
from PIL import Image
import numpy as np
import io
import pickle
from scipy.stats import entropy

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024

# --------------------
# Model definitions
# --------------------
class BinaryDNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(196614, 384), nn.BatchNorm1d(384), nn.ReLU(), nn.Dropout(0.45),
            nn.Linear(384, 192), nn.BatchNorm1d(192), nn.ReLU(), nn.Dropout(0.35),
            nn.Linear(192, 96), nn.BatchNorm1d(96), nn.ReLU(), nn.Dropout(0.25),
            nn.Linear(96, 2))
    def forward(self, x): return self.net(x)

class MultiDNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(196614, 384), nn.BatchNorm1d(384), nn.ReLU(), nn.Dropout(0.45),
            nn.Linear(384, 192), nn.BatchNorm1d(192), nn.ReLU(), nn.Dropout(0.35),
            nn.Linear(192, 96), nn.BatchNorm1d(96), nn.ReLU(), nn.Dropout(0.25),
            nn.Linear(96, 4))
    def forward(self, x): return self.net(x)

# --------------------
# Load artifacts ONCE
# --------------------
with open('final_scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
with open('final_bin_le.pkl', 'rb') as f:
    binary_le = pickle.load(f)
with open('final_multi_le.pkl', 'rb') as f:
    multi_le = pickle.load(f)

device = torch.device('cpu')
binary_model = BinaryDNN()
binary_model.load_state_dict(torch.load('final_bin_best.pth', map_location=device))
binary_model.eval()
multi_model = MultiDNN()
multi_model.load_state_dict(torch.load('final_multi_best.pth', map_location=device))
multi_model.eval()

# --------------------
# V3 Feature extraction
# --------------------
def extract_lsb_feature(img, k=2):
    mask = (1 << k) - 1
    return (img & mask).flatten()

def extract_randomness_features(lsb_feat):
    chunk_size = 512
    lsb_reshaped = lsb_feat.reshape(-1, chunk_size)
    ent_values = []
    for block in lsb_reshaped[::10]:
        hist = np.bincount(block.astype(np.int32), minlength=4)
        hist = hist / max(hist.sum(), 1)
        ent = entropy(hist + 1e-10)
        ent_values.append(ent)
    mean_entropy = np.mean(ent_values)
    sample = lsb_feat[:5000].astype(np.int32)
    transitions = np.sum(np.abs(np.diff(sample)))
    transition_rate = transitions / max(len(sample) - 1, 1)
    hist_full = np.bincount(lsb_feat.astype(np.int32), minlength=4)
    expected = len(lsb_feat) / 4
    chi2_stat = np.sum((hist_full - expected) ** 2 / expected) if expected > 0 else 0
    return np.array([mean_entropy, transition_rate, chi2_stat], dtype=np.float32)

def extract_url_pattern_features(lsb_feat):
    chunk_size = 4
    num_chunks = len(lsb_feat) // chunk_size
    lsb_trimmed = lsb_feat[:num_chunks * chunk_size].astype(np.int32)
    byte_values = []
    for i in range(0, len(lsb_trimmed), chunk_size):
        val = (lsb_trimmed[i] << 6) | (lsb_trimmed[i+1] << 4) | (lsb_trimmed[i+2] << 2) | lsb_trimmed[i+3]
        byte_values.append(val)
    byte_values = np.array(byte_values)
    url_char_count = np.sum((byte_values >= 10) & (byte_values <= 15))
    html_char_count = np.sum((byte_values >= 0) & (byte_values <= 5))
    unique_ratio = len(np.unique(byte_values)) / max(len(byte_values), 1)
    return np.array([
        url_char_count / max(len(byte_values), 1),
        html_char_count / max(len(byte_values), 1),
        unique_ratio
    ], dtype=np.float32)

def extract_features(image, k=2):
    image = image.resize((256, 256), Image.LANCZOS)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    img_array = np.array(image)
    lsb_feat = extract_lsb_feature(img_array, k=k)
    rand_feat = extract_randomness_features(lsb_feat)
    url_feat = extract_url_pattern_features(lsb_feat)
    combined = np.concatenate([lsb_feat.astype(np.float32), rand_feat, url_feat])
    return combined

# --------------------
# Flask endpoints
# --------------------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        file = request.files['file']
        image = Image.open(io.BytesIO(file.read()))
        features = extract_features(image)
        features_normalized = scaler.transform(features.reshape(1, -1)).astype(np.float32)
        features_tensor = torch.FloatTensor(features_normalized)

        with torch.no_grad():
            binary_output = binary_model(features_tensor)
            binary_probs = torch.softmax(binary_output, dim=1).cpu().numpy()[0]
            binary_pred_idx = int(np.argmax(binary_probs))
            binary_class = binary_le.classes_[binary_pred_idx]
        
        if binary_class == 'stego':
            with torch.no_grad():
                multi_output = multi_model(features_tensor)
                multi_probs = torch.softmax(multi_output, dim=1).cpu().numpy()[0]
                multi_pred_idx = int(np.argmax(multi_probs))
                payload_type = multi_le.classes_[multi_pred_idx]
        else:
            payload_type = 'clean'
            multi_probs = np.zeros(len(multi_le.classes_))
            multi_probs[multi_le.transform(['clean'])[0]] = 1.0

        result = {
            'binary_prediction': {
                'class': binary_class,
                'confidence': float(binary_probs[binary_pred_idx]),
                'probabilities': {c: float(binary_probs[binary_le.transform([c])[0]]) for c in binary_le.classes_}
            },
            'multiclass_prediction': {
                'payload_type': payload_type,
                'probabilities': {c: float(multi_probs[i]) for i, c in enumerate(multi_le.classes_)}
            }
        }
        return jsonify(result)
    except Exception as e:
        import traceback
        print('Error:', str(e))
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({'error': 'File size exceeds 5MB'}), 413

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
