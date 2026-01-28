import torch
import torchvision.transforms as T
from PIL import Image
from model import SiameseNet
import torch.nn.functional as F

device = "cuda" if torch.cuda.is_available() else "cpu"

# Same transform as training
transform = T.Compose([
    T.Resize((224,224)),
    T.ToTensor(),
    T.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
])

# Load trained model
model = SiameseNet().to(device)
model.load_state_dict(torch.load("models/siamese_vit_cnn.pth", map_location=device))
model.eval()

def verify(img1_path, img2_path, threshold=0.7):
    img1 = transform(Image.open(img1_path).convert("RGB")).unsqueeze(0).to(device)
    img2 = transform(Image.open(img2_path).convert("RGB")).unsqueeze(0).to(device)

    with torch.no_grad():
        e1, e2 = model(img1, img2)
        dist = F.pairwise_distance(e1, e2).item()

    print("Distance:", dist)

    if dist < threshold:
        print("✅ SAME PERSON (GENUINE)")
    else:
        print("❌ FORGERY / DIFFERENT")

# ---------------- TEST EXAMPLES ----------------
# Change these paths to real images from your test folder

verify(
    r"C:\Users\comp\signature_verification_vit_cnn\data\sign_data\test\049\01_049.png",
    r"C:\Users\comp\signature_verification_vit_cnn\data\sign_data\test\049\02_049.png"
)
verify(
    r"C:\Users\comp\signature_verification_vit_cnn\data\sign_data\test\049\01_049.png",
    r"C:\Users\comp\signature_verification_vit_cnn\data\sign_data\test\049_forg\01_0114049.png"
)

verify(
    r"C:\Users\comp\signature_verification_vit_cnn\data\sign_data\test\050\01_050.png",
    r"C:\Users\comp\signature_verification_vit_cnn\data\sign_data\test\050\02_050.png"
)

verify(
    r"C:\Users\comp\signature_verification_vit_cnn\data\sign_data\test\050\01_050.png",
    r"C:\Users\comp\signature_verification_vit_cnn\data\sign_data\test\050_forg\01_0125050.png"
)



verify(
    r"C:\Users\comp\signature_verification_vit_cnn\data\sign_data\test\051\01_051.png",
    r"C:\Users\comp\signature_verification_vit_cnn\data\sign_data\test\051\02_051.png"
)
verify(
    r"C:\Users\comp\signature_verification_vit_cnn\data\sign_data\test\051\01_051.png",
    r"C:\Users\comp\signature_verification_vit_cnn\data\sign_data\test\051_forg\01_0104051.png"
)

verify(
    r"C:\Users\comp\signature_verification_vit_cnn\data\sign_data\test\052\01_052.png",
    r"C:\Users\comp\signature_verification_vit_cnn\data\sign_data\test\052\02_052.png"
)

verify(
    r"C:\Users\comp\signature_verification_vit_cnn\data\sign_data\test\052\01_052.png",
    r"C:\Users\comp\signature_verification_vit_cnn\data\sign_data\test\052_forg\01_0106052.png"
)