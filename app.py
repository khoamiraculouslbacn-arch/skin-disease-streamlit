import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import gdown
import os

st.set_page_config(page_title="Chẩn đoán Bệnh Da Liễu", layout="centered", page_icon="🩺")

st.title("🩺 HỆ THỐNG CHẨN ĐOÁN BỆNH DA LIỄU")
st.markdown("**Mô hình: EfficientNet-B2 + CBAM** | Đồ án tốt nghiệp")

# ====================== LOAD MODEL ======================
@st.cache_resource
def load_model():
    model_path = "efficientnet_b2_cbam_best.pth"
    
    if not os.path.exists(model_path):
        with st.spinner("🔄 Đang tải mô hình từ Google Drive... (lần đầu có thể mất 40-90 giây)"):
            url = "https://drive.google.com/uc?id=1QIT3-RROTG-LTYiAZC6czlP-QxSir3oI"
            gdown.download(url, model_path, quiet=False)
        st.success("✅ Tải mô hình thành công!")
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    model = EfficientNetCBAM(num_classes=23)
    
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    return model

model = load_model()

# ====================== CLASS NAMES ======================
class_names = [
    "Acne and Rosacea Photos", "Actinic Keratosis Basal Cell Carcinoma", "Atopic Dermatitis Photos",
    "Bullous Disease Photos", "Cellulitis Impetigo", "Eczema Photos", "Exanthems and Drug Eruptions",
    "Hair Loss Photos", "Herpes HPV", "Light Diseases", "Lupus", "Melanoma", "Nail Fungus",
    "Poison Ivy", "Psoriasis Lichen Planus", "Rosacea", "Seborrheic Keratoses", "Systemic Disease",
    "Tinea Ringworm", "Urticaria Hives", "Vascular Tumors", "Vasculitis", "Warts Molluscum"
]

# ====================== TRANSFORM ======================
transform = transforms.Compose([
    transforms.Resize(300),
    transforms.CenterCrop(288),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ====================== PREDICT ======================
def predict(image):
    input_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.nn.functional.softmax(output[0], dim=0)
        top_prob, top_idx = torch.topk(probs, 5)
    
    results = []
    for i in range(5):
        results.append(f"{class_names[top_idx[i]]}: {top_prob[i].item()*100:.2f}%")
    return results

# ====================== UI ======================
st.sidebar.header("📤 Tải ảnh lên")
uploaded_file = st.sidebar.file_uploader("Chọn ảnh da cần chẩn đoán", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    col1, col2 = st.columns([1, 1.2])
    
    with col1:
        st.image(image, caption="Ảnh đã upload", use_column_width=True)
    
    with col2:
        with st.spinner("Đang phân tích ảnh..."):
            results = predict(image)
        
        st.subheader("🔍 Kết quả dự đoán (Top 5)")
        for r in results:
            st.metric(label=r.split(":")[0], value=r.split(":")[1])

else:
    st.info("👆 Vui lòng upload ảnh da từ thanh bên trái")

st.caption("Ứng dụng hỗ trợ chẩn đoán bệnh da liễu sử dụng trí tuệ nhân tạo | Đồ án tốt nghiệp")
