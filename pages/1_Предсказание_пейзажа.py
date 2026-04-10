import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import pandas as pd
import time
import requests
from io import BytesIO

# ========== 1. КОНФИГУРАЦИЯ ==========
MODEL_URL = "https://huggingface.co/Scatana/nn_streamlit/resolve/main/convnext_base.pth"
CLASS_NAMES = ['Здания', 'Лес', 'Ледники', 'Горы', 'Море', 'Улица']
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ========== 2. ЗАГРУЗКА МОДЕЛИ ==========
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


# ========== 3. ПРЕДОБРАБОТКА ==========
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


# ========== 4. ПРЕДСКАЗАНИЕ С ЗАМЕРОМ ВРЕМЕНИ ==========
def predict_single_with_time(image_tensor, model):
    start_time = time.time()
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        predicted_idx = torch.argmax(probabilities, dim=1).item()
    elapsed_time = time.time() - start_time
    return predicted_idx, probabilities.cpu().numpy()[0], elapsed_time


# ========== 5. ЗАГРУЗКА ИЗОБРАЖЕНИЯ ПО ССЫЛКЕ ==========
def load_image_from_url(url):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content)).convert('RGB')
        return image
    except Exception as e:
        st.error(f"Не удалось загрузить изображение по ссылке: {e}")
        return None


# ========== 6. ОСНОВНОЙ ИНТЕРФЕЙС ==========
def main():
    st.set_page_config(page_title="Классификатор изображений", layout="wide")
    st.title("🏞️ Классификатор изображений")
    st.markdown("Модель определяет: здание, лес, ледник, гора, море или улица.")

    # Загрузка модели
    with st.spinner("Загрузка модели..."):
        try:
            model = load_model()
            st.success("Модель загружена")
        except Exception as e:
            st.error(f"Ошибка загрузки модели: {e}")
            st.stop()

    # ----- РАЗДЕЛ: ЗАГРУЗКА ФАЙЛОВ -----
    st.subheader("📁 Загрузите изображения с компьютера")
    uploaded_files = st.file_uploader(
        "Выберите одно или несколько изображений (JPG, JPEG, PNG)",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True
    )

    # ----- РАЗДЕЛ: ССЫЛКА НА ИЗОБРАЖЕНИЕ -----
    st.subheader("🔗 Или вставьте ссылку на изображение")
    image_url = st.text_input("Ссылка на изображение (URL)")

    # ----- ОБРАБОТКА ЗАГРУЖЕННЫХ ФАЙЛОВ -----
    if uploaded_files:
        if st.button("Классифицировать загруженные файлы"):
            results = []
            progress_bar = st.progress(0)
            status_text = st.empty()

            for i, file in enumerate(uploaded_files):
                image = Image.open(file).convert('RGB')
                tensor = preprocess_image(image)
                pred_idx, probs, elapsed = predict_single_with_time(tensor, model)
                predicted_class = CLASS_NAMES[pred_idx]
                confidence = probs[pred_idx] * 100

                results.append({
                    "filename": file.name,
                    "predicted_class": predicted_class,
                    "confidence": f"{confidence:.2f}%",
                    "time": f"{elapsed:.3f} сек",
                    "image": image
                })
                progress_bar.progress((i + 1) / len(uploaded_files))
                status_text.text(f"Обработано {i+1} из {len(uploaded_files)}")

            progress_bar.empty()
            status_text.empty()
            st.success("Классификация завершена!")

            # Галерея с результатами (без таблицы)
            st.subheader("Галерея с предсказаниями")
            cols = st.columns(3)
            for idx, res in enumerate(results):
                col = cols[idx % 3]
                with col:
                    st.image(res["image"], caption=res["filename"], use_container_width=True)
                    st.markdown(f"**{res['predicted_class']}**  \nУверенность: {res['confidence']}  \n⏱️ {res['time']}")

    # ----- ОБРАБОТКА ССЫЛКИ -----
    if image_url:
        if st.button("Классифицировать по ссылке"):
            image = load_image_from_url(image_url)
            if image is not None:
                st.image(image, caption="Изображение по ссылке", use_container_width=True)
                tensor = preprocess_image(image)
                pred_idx, probs, elapsed = predict_single_with_time(tensor, model)
                predicted_class = CLASS_NAMES[pred_idx]
                confidence = probs[pred_idx] * 100

                st.markdown(f"### Результат:")
                st.markdown(f"**Класс:** {predicted_class}")
                st.markdown(f"**Уверенность:** {confidence:.2f}%")
                st.markdown(f"**Время предсказания:** {elapsed:.3f} секунд")
            else:
                st.error("Не удалось загрузить изображение по ссылке")

    st.caption("Примечание: Уверенность показывает, насколько модель уверена в своём выборе.")


if __name__ == "__main__":
    main()