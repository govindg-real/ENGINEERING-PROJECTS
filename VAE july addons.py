import os
import random
import glob
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.linear_model import Ridge
from sklearn.cluster import KMeans
from sklearn.model_selection import cross_val_score

# --- Seeding & Device ---
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)

try:
    import nibabel as nib
    HAS_NIB = True
except ImportError:
    HAS_NIB = False

# --- Config ---
DATA_DIR = "/mnt/storage/processed_mri/sst/test_sandbox/baseline"
EPOCHS = 10  # Kept low for your initial test run
BATCH_SIZE = 64
LR = 1e-3
LATENT_DIM = 512
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- 1. Multi-Participant DataLoader ---
def load_fmri_4d(path):
    img = nib.load(path)
    x = img.get_fdata().astype(np.float32)
    x = np.nan_to_num(x)
    mn, mx = x.min(), x.max()
    return (x - mn) / (mx - mn + 1e-8)

class MultiSubjectSliceDataset(Dataset):
    def __init__(self, subject_paths):
        self.slices = []
        print(f"Indexing metadata for {len(subject_paths)} subjects...")
        for path in subject_paths:
            img = nib.load(path)
            shape = img.shape
            Z, T = shape[2], shape[3]
            for z in range(Z):
                for t in range(T):
                    self.slices.append((path, z, t))
        random.shuffle(self.slices)

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, i):
        path, z, t = self.slices[i]
        vol4d = load_fmri_4d(path)
        sl = vol4d[:, :, z, t].astype(np.float32)
        sl = np.expand_dims(sl, 0)
        return torch.from_numpy(sl)

# Find subjects and create loaders
all_subjects = glob.glob(os.path.join(DATA_DIR, "*.nii*"))
if not all_subjects:
    print("No subjects found! Check your DATA_DIR.")

random.shuffle(all_subjects)
split_idx = int(0.9 * len(all_subjects))
train_subs, val_subs = all_subjects[:split_idx], all_subjects[split_idx:]

train_loader = DataLoader(MultiSubjectSliceDataset(train_subs), batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
val_loader = DataLoader(MultiSubjectSliceDataset(val_subs), batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

# Get image dimensions dynamically
sample_batch = next(iter(train_loader))
_, _, H, W = sample_batch.shape

# --- 2. Model (Conv2DVAE) ---
class Conv2DVAE(nn.Module):
    def __init__(self, latent_dim=32, in_ch=1, hw=(64, 64)):
        super().__init__()
        H, W = hw
        self.enc = nn.Sequential(
            nn.Conv2d(in_ch, 16, 3, 2, 1), nn.ReLU(True),
            nn.Conv2d(16, 32, 3, 2, 1),   nn.ReLU(True),
            nn.Conv2d(32, 64, 3, 2, 1),   nn.ReLU(True),
            nn.Conv2d(64, 128, 3, 2, 1),  nn.ReLU(True),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, in_ch, H, W)
            h = self.enc(dummy)
        self.enc_shape = h.shape[1:]
        self.flat_dim = int(np.prod(self.enc_shape))

        self.fc_mu = nn.Linear(self.flat_dim, latent_dim)
        self.fc_lv = nn.Linear(self.flat_dim, latent_dim)

        self.fc_up = nn.Linear(latent_dim, self.flat_dim)
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),  nn.ReLU(True),
            nn.ConvTranspose2d(32, 16, 4, 2, 1),  nn.ReLU(True),
            nn.ConvTranspose2d(16, 1, 4, 2, 1),
            nn.Sigmoid(),
        )

    def encode(self, x):
        h = self.enc(x).view(x.size(0), -1)
        return self.fc_mu(h), self.fc_lv(h)

    def forward(self, x):
        mu, lv = self.encode(x)
        std = torch.exp(0.5 * lv)
        z = mu + torch.randn_like(std) * std
        h = self.fc_up(z).view(-1, *self.enc_shape)
        xhat = self.dec(h)
        
        # Match dimensions if rounding issues occur
        if xhat.shape != x.shape:
            min_h, min_w = min(xhat.shape[2], x.shape[2]), min(xhat.shape[3], x.shape[3])
            xhat, x = xhat[:, :, :min_h, :min_w], x[:, :, :min_h, :min_w]
        return xhat, mu, lv

model = Conv2DVAE(latent_dim=LATENT_DIM, hw=(H, W)).to(device)
opt = torch.optim.Adam(model.parameters(), lr=LR)

def vae_loss(x, xhat, mu, lv, beta=1.0):
    recon = nn.functional.mse_loss(xhat, x, reduction='sum') / x.size(0)
    kl = -0.5 * torch.sum(1 + lv - mu.pow(2) - lv.exp()) / x.size(0)
    return recon + (beta * kl)

# --- 3. Training Loop ---
for ep in range(1, EPOCHS + 1):
    model.train()
    tr_loss, count = 0.0, 0
    for x in train_loader:
        x = x.to(device)
        opt.zero_grad()
        xhat, mu, lv = model(x)
        loss = vae_loss(x, xhat, mu, lv)
        loss.backward()
        opt.step()
        tr_loss += loss.item() * x.size(0)
        count += x.size(0)
    print(f"Epoch {ep:02d} | Train Loss: {tr_loss/count:.4f}")

# --- 4. Latent Vector Extraction ---
def extract_subject_latent(subject_path):
    model.eval()
    vol4d = load_fmri_4d(subject_path)
    Z, T = vol4d.shape[2], vol4d.shape[3]
    latents = []
    with torch.no_grad():
        for z in range(Z):
            for t in range(T):
                sl = vol4d[:, :, z, t].astype(np.float32)
                if (sl > 1e-6).mean() < 0.01: continue
                sl_tensor = torch.from_numpy(sl).unsqueeze(0).unsqueeze(0).to(device)
                mu, _ = model.encode(sl_tensor)
                latents.append(mu.cpu().numpy())
    return np.mean(np.vstack(latents), axis=0) if latents else np.zeros(LATENT_DIM)

print("\nExtracting group latent space feature matrices...")
X_features = np.array([extract_subject_latent(sub) for sub in all_subjects])


# =====================================================================
# ANALYTICS (Using dummy arrays until you load your real behavioral data)
# =====================================================================

y_ssrt = np.random.randn(len(all_subjects))            # DUMMY SSRT
y_mental_health = np.random.randn(len(all_subjects))   # DUMMY MENTAL HEALTH

# --- Linear Regression (SSRT & Mental Health) ---
print("\n--- Linear Regression Predictions ---")
reg_behavior = Ridge(alpha=1.0)
scores_beh = cross_val_score(reg_behavior, X_features, y_ssrt, cv=5, scoring='r2')
print(f"SSRT Prediction R^2: {scores_beh.mean():.3f}")

reg_health = Ridge(alpha=1.0)
scores_health = cross_val_score(reg_health, X_features, y_mental_health, cv=5, scoring='r2')
print(f"Mental Health Prediction R^2: {scores_health.mean():.3f}")

# --- Clustering & Mental Health Check ---
print("\n--- Clustering Analysis ---")
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(X_features)

for c in range(3):
    cluster_scores = y_mental_health[cluster_labels == c]
    print(f"Cluster Group {c} (n={len(cluster_scores)}): Mean Mental Health Score = {cluster_scores.mean():.3f}")