import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import time
import numpy as np

# ========== 1. КОНФИГУРАЦИЯ ==========
MODEL_URL = "https://huggingface.co/Scatana/nn_streamlit/resolve/main/convnext_base.pth"
CLASS_NAMES = ['Здания', 'Лес', 'Ледники', 'Горы', 'Море', 'Улица']
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ========== 2. ФУНКЦИЯ ДЛЯ ПОЛУЧЕНИЯ ЦВЕТА ПО УВЕРЕННОСТИ ==========
def get_confidence_color(confidence):
    """
    Возвращает цвет в формате RGB в зависимости от уверенности (0..1).
    От красного (0) через жёлтый (0.5) к зелёному (1).
    """
    c = max(0.0, min(1.0, confidence))
    if c <= 0.5:
        r = 255
        g = int(255 * (c / 0.5))
        b = 0
    else:
        r = int(255 * (1 - (c - 0.5) / 0.5))
        g = 255
        b = 0
    return f"rgb({r}, {g}, {b})"


# ========== 3. ЗАГРУЗКА МОДЕЛИ ==========
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


# ========== 4. ПРЕДОБРАБОТКА ==========
def preprocess_image(image: Image.Image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    return transform(image).unsqueeze(0).to(DEVICE)


# ========== 5. ПРЕДСКАЗАНИЕ (ТОП-2 + ВРЕМЯ) ==========
def predict_top2_with_time(image_tensor, model):
    start_time = time.time()
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        probs_np = probabilities.cpu().numpy()[0]
        top2_indices = np.argsort(probs_np)[-2:][::-1]
        top2_probs = probs_np[top2_indices]
    elapsed_time = time.time() - start_time
    return top2_indices, top2_probs, elapsed_time


# ========== 6. ФОРМАТИРОВАНИЕ С ЦВЕТОМ ==========
def format_prediction_with_gradient(class_name, confidence):
    color = get_confidence_color(confidence)
    confidence_pct = confidence * 100
    return f'**<span style="color:{color};">{class_name} ({confidence_pct:.1f}%)</span>**'


# ========== 7. ОСНОВНОЙ ИНТЕРФЕЙС ==========
def main():
    st.set_page_config(page_title="Классификатор изображений", layout="wide")
    st.title("🏞️ Классификатор изображений")
    st.markdown("Модель определяет: здание, лес, ледник, гора, море или улица.")
    st.markdown("🎨 **Цвет уверенности:** от 🔴 красного (0%) до 🟢 зелёного (100%).")

    # Загрузка модели
    with st.spinner("Загрузка модели..."):
        try:
            model = load_model()
            st.success("Модель загружена")
        except Exception as e:
            st.error(f"Ошибка загрузки модели: {e}")
            st.stop()

    # Виджет загрузки нескольких изображений
    uploaded_files = st.file_uploader(
        "Выберите одно или несколько изображений (JPG, JPEG, PNG)",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True
    )

    if uploaded_files and len(uploaded_files) > 0:
        st.subheader(f"Загружено {len(uploaded_files)} изображений")

        if st.button("Классифицировать все"):
            results = []
            progress_bar = st.progress(0)
            status_text = st.empty()

            for i, uploaded_file in enumerate(uploaded_files):
                image = Image.open(uploaded_file).convert('RGB')
                img_tensor = preprocess_image(image)
                top2_idx, top2_probs, elapsed = predict_top2_with_time(img_tensor, model)

                top1_class = CLASS_NAMES[top2_idx[0]]
                top1_prob = top2_probs[0]
                top2_class = CLASS_NAMES[top2_idx[1]]
                top2_prob = top2_probs[1]

                results.append({
                    "filename": uploaded_file.name,
                    "top1": (top1_class, top1_prob),
                    "top2": (top2_class, top2_prob),
                    "time": elapsed,
                    "image": image
                })
                progress_bar.progress((i + 1) / len(uploaded_files))
                status_text.text(f"Обработано {i+1} из {len(uploaded_files)}")

            progress_bar.empty()
            status_text.empty()
            st.success("Классификация завершена!")

            # Галерея с результатами
            st.subheader("Галерея с предсказаниями")
            cols = st.columns(3)
            for idx, res in enumerate(results):
                col = cols[idx % 3]
                with col:
                    st.image(res["image"], caption=res["filename"], width=180)
                    top1_html = format_prediction_with_gradient(res["top1"][0], res["top1"][1])
                    st.markdown(f"{top1_html}<br>📌 {res['top2'][0]} ({res['top2'][1]*100:.1f}%)<br>⏱️ {res['time']:.3f} сек", unsafe_allow_html=True)

            st.caption("🎨 Цвет top-1 меняется от красного (низкая уверенность) до зелёного (высокая).")


if __name__ == "__main__":
    main()