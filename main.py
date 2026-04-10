import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import pandas as pd
import os

# ========== 1. КОНФИГУРАЦИЯ ==========
# Модель ConvNeXt (Hugging Face)
MODEL_URL = "https://huggingface.co/Scatana/nn_streamlit/resolve/main/best_model_0.9012_epoch1.pth"

# Классы для ConvNeXt (6 классов)
CLASS_NAMES_CONVNEXT = ['Здания', 'Лес', 'Ледники', 'Горы', 'Море', 'Улица']

# Классы для вашей ResNet101 (100 видов спорта)
CLASS_NAMES_RESNET101 = [
    'air hockey', 'ampute football', 'archery', 'arm wrestling', 'axe throwing',
    'balance beam', 'barell racing', 'baseball', 'basketball', 'baton twirling',
    'bike polo', 'billiards', 'bmx', 'bobsled', 'bowling', 'boxing', 'bull riding',
    'bungee jumping', 'canoe slamon', 'cheerleading', 'chuckwagon racing', 'cricket',
    'croquet', 'curling', 'disc golf', 'fencing', 'field hockey', 'figure skating men',
    'figure skating pairs', 'figure skating women', 'fly fishing', 'football',
    'formula 1 racing', 'frisbee', 'gaga', 'giant slalom', 'golf', 'hammer throw',
    'hang gliding', 'harness racing', 'high jump', 'hockey', 'horse jumping',
    'horse racing', 'horseshoe pitching', 'hurdles', 'hydroplane racing', 'ice climbing',
    'ice yachting', 'jai alai', 'javelin', 'jousting', 'judo', 'lacrosse', 'log rolling',
    'luge', 'motorcycle racing', 'mushing', 'nascar racing', 'olympic wrestling',
    'parallel bar', 'pole climbing', 'pole dancing', 'pole vault', 'polo', 'pommel horse',
    'rings', 'rock climbing', 'roller derby', 'rollerblade racing', 'rowing', 'rugby',
    'sailboat racing', 'shot put', 'shuffleboard', 'sidecar racing', 'ski jumping',
    'sky surfing', 'skydiving', 'snow boarding', 'snowmobile racing', 'speed skating',
    'steer wrestling', 'sumo wrestling', 'surfing', 'swimming', 'table tennis', 'tennis',
    'track bicycle', 'trapeze', 'tug of war', 'ultimate', 'uneven bars', 'volleyball',
    'water cycling', 'water polo', 'weightlifting', 'wheelchair basketball',
    'wheelchair racing', 'wingsuit flying'
]

# Количество классов для ResNet101 (должно быть 100)
NUM_CLASSES_RESNET101 = len(CLASS_NAMES_RESNET101)

# Устройство
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Путь к локальной модели ResNet101
RESNET101_PATH = "resnet101_model.pth"


# ========== 2. ЗАГРУЗКА CONVNEXT (Hugging Face) ==========
@st.cache_resource
def load_convnext_model():
    """Загружает ConvNeXt с весами из Hugging Face."""
    model = models.convnext_base(weights=None)
    model.classifier[2] = nn.Linear(1024, len(CLASS_NAMES_CONVNEXT))
    state_dict = torch.hub.load_state_dict_from_url(
        MODEL_URL,
        map_location='cpu',
        file_name='best_model.pth'
    )
    model.load_state_dict(state_dict)
    model.eval()
    model.to(DEVICE)
    return model


# ========== 3. ЗАГРУЗКА RESNET101 (локальный файл) ==========
@st.cache_resource
def load_resnet101_model():
    """Загружает ResNet101 с весами из локального файла."""
    if not os.path.exists(RESNET101_PATH):
        st.error(f"Файл модели не найден: {RESNET101_PATH}")
        st.stop()

    # Создаём архитектуру ResNet101
    model = models.resnet101(pretrained=False)
    # Заменяем последний полносвязный слой на 100 классов
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, NUM_CLASSES_RESNET101)

    # Загружаем веса
    state_dict = torch.load(RESNET101_PATH, map_location='cpu')
    model.load_state_dict(state_dict)
    model.eval()
    model.to(DEVICE)
    return model


# ========== 4. ПРЕДОБРАБОТКА ИЗОБРАЖЕНИЯ (общая для обеих моделей) ==========
def preprocess_image(image: Image.Image):
    """Преобразует PIL Image в тензор (224x224, нормализация ImageNet)."""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    img_tensor = transform(image).unsqueeze(0)   # добавляем batch dimension
    return img_tensor.to(DEVICE)


# ========== 5. ФУНКЦИЯ ПРЕДСКАЗАНИЯ ==========
def predict_single(image_tensor, model):
    """Возвращает предсказанный индекс и вероятности всех классов."""
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        predicted_idx = torch.argmax(probabilities, dim=1).item()
    return predicted_idx, probabilities.cpu().numpy()[0]


# ========== 6. ОСНОВНОЙ ИНТЕРФЕЙС СТРИМЛИТ ==========
def main():
    st.set_page_config(page_title="Классификатор спорта и сцен", layout="wide")
    st.title("🏅 Классификатор изображений")
    st.markdown("""
    **Две модели:**
    - **ConvNeXt** – определяет тип ландшафта/сцены (6 классов: здания, лес, ледники, горы, море, улица)
    - **ResNet101** – определяет вид спорта (100 классов, датасет Sports-100)
    """)

    # Загружаем модели (кэшируются, загрузка один раз)
    with st.spinner("Загрузка модели ConvNeXt..."):
        convnext_model = load_convnext_model()
        st.success("✅ ConvNeXt загружена")

    with st.spinner("Загрузка модели ResNet101..."):
        resnet101_model = load_resnet101_model()
        st.success("✅ ResNet101 загружена")

    # Создаём две вкладки
    tab1, tab2 = st.tabs(["📷 ConvNeXt (сцены)", "🏃 ResNet101 (спорт)"])

    # ---------- ВКЛАДКА 1: ConvNeXt ----------
    with tab1:
        st.header("Классификация сцен (ConvNeXt)")
        st.markdown("Модель определяет: здание, лес, ледник, гора, море, улица.")

        uploaded_files_tab1 = st.file_uploader(
            "Выберите одно или несколько изображений",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True,
            key="tab1_uploader"
        )

        if uploaded_files_tab1:
            st.subheader(f"Загружено {len(uploaded_files_tab1)} изображений")
            if st.button("Классифицировать (ConvNeXt)", key="tab1_predict"):
                results = []
                progress_bar = st.progress(0)
                status_text = st.empty()

                for i, file in enumerate(uploaded_files_tab1):
                    image = Image.open(file).convert('RGB')
                    tensor = preprocess_image(image)
                    pred_idx, probs = predict_single(tensor, convnext_model)
                    predicted_class = CLASS_NAMES_CONVNEXT[pred_idx]
                    confidence = probs[pred_idx] * 100

                    results.append({
                        "Файл": file.name,
                        "Класс": predicted_class,
                        "Уверенность": f"{confidence:.2f}%",
                        "Изображение": image
                    })

                    progress_bar.progress((i + 1) / len(uploaded_files_tab1))
                    status_text.text(f"Обработано {i+1} из {len(uploaded_files_tab1)}")

                progress_bar.empty()
                status_text.empty()
                st.success("Классификация завершена!")

                # Таблица результатов
                df = pd.DataFrame(results)
                st.dataframe(df.drop(columns=["Изображение"]), use_container_width=True)

                # Галерея
                st.subheader("Галерея с предсказаниями")
                cols = st.columns(3)
                for idx, res in enumerate(results):
                    col = cols[idx % 3]
                    with col:
                        st.image(res["Изображение"], caption=res["Файл"], use_container_width=True)
                        st.markdown(f"**{res['Класс']}**  \nУверенность: {res['Уверенность']}")
                st.caption("Уверенность показывает, насколько модель уверена в своём выборе.")

    # ---------- ВКЛАДКА 2: ResNet101 ----------
    with tab2:
        st.header("Классификация видов спорта (ResNet101)")
        st.markdown(f"Модель обучена на **{NUM_CLASSES_RESNET101}** спортивных активностях (например, футбол, баскетбол, теннис и т.д.).")

        uploaded_files_tab2 = st.file_uploader(
            "Выберите одно или несколько изображений",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True,
            key="tab2_uploader"
        )

        if uploaded_files_tab2:
            st.subheader(f"Загружено {len(uploaded_files_tab2)} изображений")
            if st.button("Классифицировать (ResNet101)", key="tab2_predict"):
                results = []
                progress_bar = st.progress(0)
                status_text = st.empty()

                for i, file in enumerate(uploaded_files_tab2):
                    image = Image.open(file).convert('RGB')
                    tensor = preprocess_image(image)
                    pred_idx, probs = predict_single(tensor, resnet101_model)
                    predicted_class = CLASS_NAMES_RESNET101[pred_idx]
                    confidence = probs[pred_idx] * 100

                    results.append({
                        "Файл": file.name,
                        "Вид спорта": predicted_class,
                        "Уверенность": f"{confidence:.2f}%",
                        "Изображение": image
                    })

                    progress_bar.progress((i + 1) / len(uploaded_files_tab2))
                    status_text.text(f"Обработано {i+1} из {len(uploaded_files_tab2)}")

                progress_bar.empty()
                status_text.empty()
                st.success("Классификация завершена!")

                # Таблица результатов
                df = pd.DataFrame(results)
                st.dataframe(df.drop(columns=["Изображение"]), use_container_width=True)

                # Галерея
                st.subheader("Галерея с предсказаниями")
                cols = st.columns(3)
                for idx, res in enumerate(results):
                    col = cols[idx % 3]
                    with col:
                        st.image(res["Изображение"], caption=res["Файл"], use_container_width=True)
                        st.markdown(f"**{res['Вид спорта']}**  \nУверенность: {res['Уверенность']}")
                st.caption("Уверенность показывает, насколько модель уверена в своём выборе.")


if __name__ == "__main__":
    main()