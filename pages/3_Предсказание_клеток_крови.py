import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import pandas as pd

MODEL_URL = "https://huggingface.co/Scatana/nn_streamlit/resolve/main/weights_model_convnext_small_v2.pt"

CLASS_NAMES = ['EOSINOPHIL', 'LYMPHOCYTE', 'MONOCYTE', 'NEUTROPHIL']

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@st.cache_resource
def load_model():
    model = models.convnext_small(pretrained=False)
    model.classifier[2] = nn.Linear(in_features=768, out_features=4, bias=True)
    
    state_dict = torch.hub.load_state_dict_from_url(
        MODEL_URL,
        map_location='cpu',
        file_name='weights_model_convnext_small_v2.pt'
    )
    model.load_state_dict(state_dict)
    
    model.eval()
    model.to(DEVICE)
    return model


# ========== 3. ПРЕДОБРАБОТКА ИЗОБРАЖЕНИЯ ==========
def preprocess_image(image: Image.Image):

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    img_tensor = transform(image).unsqueeze(0)
    return img_tensor.to(DEVICE)


def predict_single(image_tensor, model):
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        predicted_idx = torch.argmax(probabilities, dim=1).item()
    return predicted_idx, probabilities.cpu().numpy()[0]


# ========== 5. ОСНОВНОЙ ИНТЕРФЕЙС STREAMLIT ==========
def main():
    st.set_page_config(page_title="Классификатор клеток", layout="wide")
    st.title("💉 Классификатор клеток 🩸")
    st.markdown("Загрузите **одну или несколько** фотографий, и модель определит, что на них изображено: ")

    # Загружаем модель (один раз, кэшируется)
    with st.spinner("Загрузка модели... Пожалуйста, подождите."):
        try:
            model = load_model()
            st.success("Модель успешно загружена!")
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

        # Кнопка для запуска классификации всех изображений
        if st.button("Классифицировать все"):
            results = []
            progress_bar = st.progress(0)
            status_text = st.empty()

            for i, uploaded_file in enumerate(uploaded_files):
                # Открываем изображение
                image = Image.open(uploaded_file).convert('RGB')
                
                # Предобработка и предсказание
                img_tensor = preprocess_image(image)
                pred_idx, probs = predict_single(img_tensor, model)
                predicted_class = CLASS_NAMES[pred_idx]
                confidence = probs[pred_idx] * 100
                
                # Сохраняем результат
                results.append({
                    "filename": uploaded_file.name,
                    "predicted_class": predicted_class,
                    "confidence": f"{confidence:.2f}%",
                    "image": image
                })
                
                # Обновляем прогресс
                progress_bar.progress((i + 1) / len(uploaded_files))
                status_text.text(f"Обработано {i+1} из {len(uploaded_files)}")
            
            progress_bar.empty()
            status_text.empty()
            st.success("Классификация завершена!")

            # Отображение результатов в виде таблицы и галереи
            st.subheader("Результаты классификации")
            
            # Таблица с результатами (без изображений)
            df = pd.DataFrame(results)
            df_display = df.drop(columns=['image'])
            st.dataframe(df_display, use_container_width=True)
            
            # Галерея: покажем каждое изображение с подписью
            st.subheader("Галерея с предсказаниями")
            cols = st.columns(3)  # 3 колонки для отображения
            for idx, res in enumerate(results):
                col = cols[idx % 3]
                with col:
                    st.image(res["image"], caption=f"{res['filename']}", use_container_width=True)
                    st.markdown(f"**{res['predicted_class']}**  \n  Уверенность: {res['confidence']}")
            st.caption("Примечание: Уверенность показывает, насколько модель уверена в своём выборе.")


if __name__ == "__main__":
    main()