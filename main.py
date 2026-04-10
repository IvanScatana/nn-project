import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# ========== ПРЯМАЯ ССЫЛКА НА МОДЕЛЬ ==========
MODEL_URL = "https://huggingface.co/Scatana/nn_streamlit/resolve/main/best_model_0.9012_epoch1.pth"

CLASS_NAMES = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@st.cache_resource
def load_model():
    model = models.convnext_base(weights=None)
    model.classifier[2] = nn.Linear(1024, 6)
    
    state_dict = torch.hub.load_state_dict_from_url(
        MODEL_URL,
        map_location='cpu',
        file_name='best_model.pth'
    )
    model.load_state_dict(state_dict)
    model.eval()
    model.to(DEVICE)
    return model

def preprocess_image(image: Image.Image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0).to(DEVICE)

def main():
    st.title("Классификатор изображений")
    st.write("Загрузите фото – модель определит: здание, лес, ледник, гора, море или улица.")
    
    with st.spinner("Загрузка модели..."):
        model = load_model()
    st.success("Модель готова!")
    
    uploaded_file = st.file_uploader("Выберите изображение", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption="Ваше изображение", use_container_width=True)
        
        if st.button("Классифицировать"):
            with st.spinner("Анализ..."):
                tensor = preprocess_image(image)
                with torch.no_grad():
                    logits = model(tensor)
                    probs = torch.softmax(logits, dim=1)[0]
                    pred_idx = torch.argmax(probs).item()
                st.success(f"**Класс:** {CLASS_NAMES[pred_idx]}")
                st.metric("Уверенность", f"{probs[pred_idx]*100:.2f}%")

if __name__ == "__main__":
    main()