import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import pandas as pd

# ========== 1. КОНФИГУРАЦИЯ ==========
# Прямые ссылки на файлы моделей в Hugging Face
CONVNEXT_URL = "https://huggingface.co/Scatana/nn_streamlit/resolve/main/best_model_0.9012_epoch1.pth"
RESNET101_URL = "https://huggingface.co/Scatana/nn_streamlit/resolve/main/resnet101_model.pth"

# Классы для ConvNeXt (6 классов)
CLASS_NAMES_CONVNEXT = ['Здания', 'Лес', 'Ледники', 'Горы', 'Море', 'Улица']

# Классы для ResNet101 (100 видов спорта)
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

# Количество классов для ResNet101
NUM_CLASSES_RESNET101 = len(CLASS_NAMES_RESNET101)

# Устройство
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ========== 2. ЗАГРУЗКА МОДЕЛИ CONVNEXT ==========
@st.cache_resource
def load_convnext_model():
    """Загружает ConvNeXt с весами из Hugging Face."""
    model = models.convnext_base(weights=None)
    model.classifier[2] = nn.Linear(1024, len(CLASS_NAMES_CONVNEXT))
    state_dict = torch.hub.load_state_dict_from_url(
        CONVNEXT_URL,
        map_location='cpu',
        file_name='convnext_model.pth'
    )
    model.load_state_dict(state_dict)
    model.eval()
    model.to(DEVICE)
    return model


# ========== 3. ЗАГРУЗКА МОДЕЛИ RESNET101 ==========
@st.cache_resource
def load_resnet101_model():
    """Загружает ResNet101 с весами из Hugging Face."""
    # Создаём архитектуру ResNet101
    model = models.resnet101(pretrained=False)
    # Заменяем последний слой на 100 классов
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, NUM_CLASSES_RESNET101)
    
    # Загружаем state_dict из URL
    state_dict = torch.hub.load_state_dict_from_url(
        RESNET101_URL,
        map_location='cpu',
        file_name='resnet101_model.pth'
    )
    model.load_state_dict(state_dict)
    model.eval()
    model.to(DEVICE)
    return model


# ========== 4. ПРЕДОБРАБОТКА ИЗОБРАЖЕНИЯ ==========
def preprocess_image(image: Image.Image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0).to(DEVICE)


# ========== 5. ФУНКЦИЯ ПРЕДСКАЗАНИЯ ==========
def predict_single(image_tensor, model):
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        predicted_idx = torch.argmax(probabilities, dim=1).item()
    return predicted_idx, probabilities.cpu().numpy()[0]


# ========== 6. ОСНОВНОЙ ИНТЕРФЕЙС ==========
def main():
    st.set_page_config(page_title="Классификатор спорта и сцен", layout="wide")
    st.title("🏅 Классификатор изображений")
    st.markdown("""
    **Две модели:**
    - **ConvNeXt** – определяет тип ландшафта/сцены (6 классов: здания, лес, ледники, горы, море, улица)
    - **ResNet101** – определяет вид спорта (100 классов, датасет Sports-100)
    """)

    # Загружаем модели (кэшируются, скачивание происходит один раз)
    with st.spinner("Загрузка модели ConvNeXt..."):
        convnext_model = load_convnext_model()
        st.success("✅ ConvNeXt загружена")

    with st.spinner("Загрузка модели ResNet101..."):
        resnet101_model = load_resnet101_model()
        st.success("✅ ResNet101 загружена")

    tab1, tab2 = st.tabs(["📷 ConvNeXt (сцены)", "🏃 ResNet101 (спорт)"])

    # ---------- ВКЛАДКА 1: ConvNeXt ----------
    with tab1:
        st.header("Классификация сцен (ConvNeXt)")
        uploaded = st.file_uploader("Выберите изображения", type=["jpg","jpeg","png"],
                                    accept_multiple_files=True, key="tab1")
        if uploaded:
            if st.button("Классифицировать (ConvNeXt)", key="btn1"):
                results = []
                progress = st.progress(0)
                status = st.empty()
                for i, file in enumerate(uploaded):
                    img = Image.open(file).convert('RGB')
                    tensor = preprocess_image(img)
                    idx, probs = predict_single(tensor, convnext_model)
                    results.append({
                        "Файл": file.name,
                        "Класс": CLASS_NAMES_CONVNEXT[idx],
                        "Уверенность": f"{probs[idx]*100:.2f}%",
                        "Изображение": img
                    })
                    progress.progress((i+1)/len(uploaded))
                    status.text(f"Обработано {i+1} из {len(uploaded)}")
                progress.empty()
                status.empty()
                st.success("Готово!")
                df = pd.DataFrame(results)
                st.dataframe(df.drop(columns=["Изображение"]), use_container_width=True)
                cols = st.columns(3)
                for j, r in enumerate(results):
                    with cols[j%3]:
                        st.image(r["Изображение"], caption=r["Файл"], use_container_width=True)
                        st.write(f"**{r['Класс']}**  \n{r['Уверенность']}")

    # ---------- ВКЛАДКА 2: ResNet101 ----------
    with tab2:
        st.header("Классификация видов спорта (ResNet101)")
        uploaded2 = st.file_uploader("Выберите изображения", type=["jpg","jpeg","png"],
                                     accept_multiple_files=True, key="tab2")
        if uploaded2:
            if st.button("Классифицировать (ResNet101)", key="btn2"):
                results = []
                progress = st.progress(0)
                status = st.empty()
                for i, file in enumerate(uploaded2):
                    img = Image.open(file).convert('RGB')
                    tensor = preprocess_image(img)
                    idx, probs = predict_single(tensor, resnet101_model)
                    results.append({
                        "Файл": file.name,
                        "Вид спорта": CLASS_NAMES_RESNET101[idx],
                        "Уверенность": f"{probs[idx]*100:.2f}%",
                        "Изображение": img
                    })
                    progress.progress((i+1)/len(uploaded2))
                    status.text(f"Обработано {i+1} из {len(uploaded2)}")
                progress.empty()
                status.empty()
                st.success("Готово!")
                df = pd.DataFrame(results)
                st.dataframe(df.drop(columns=["Изображение"]), use_container_width=True)
                cols = st.columns(3)
                for j, r in enumerate(results):
                    with cols[j%3]:
                        st.image(r["Изображение"], caption=r["Файл"], use_container_width=True)
                        st.write(f"**{r['Вид спорта']}**  \n{r['Уверенность']}")

if __name__ == "__main__":
    main()