import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet101
from PIL import Image
import warnings
import logging
import os
from collections import Counter
import requests

warnings.filterwarnings("ignore")
logging.getLogger("streamlit").setLevel(logging.ERROR)

# ====== КЛАССЫ ======
IDX_TO_CLASS = {0:'Аэрохоккей',1:'Футбол ампутантов',2:'Стрельба из лука',3:'Армрестлинг',4:'Метание топора',5:'Бревно',6:'Гонки на бочках',7:'Бейсбол',8:'Баскетбол',9:'Твирлинг',10:'Велополо',11:'Бильярдный спорт',12:'BMX',13:'Бобслей',14:'Боулинг',15:'Бокс',16:'Родео',17:'Банджи-джампинг',18:'Гребной слалом',19:'Чир спорт',20:'Гонки на повозках',21:'Крикет',22:'Крокет',23:'Кёрлинг',24:'Диск-гольф',25:'Фехтование',26:'Хоккей на траве',27:'Фигурное катание (мужчины)',28:'Фигурное катание (пары)',29:'Фигурное катание (женщины)',30:'Нахлыст',31:'Американский футбол',32:'Формула-1',33:'Фрисби',34:'Гага',35:'Гигантский слалом',36:'Гольф',37:'Метание молота',38:'Дельтапланеризм',39:'Гарнессинг',40:'Прыжки в высоту',41:'Хоккей',42:'Конкур',43:'Скачки',44:'Метание подковы',45:'Бег с барьерами',46:'Гонки на гидропланах',47:'Ледолазание',48:'Буерный спорт',49:'Хай-алай',50:'Метание копья',51:'Рыцарский турнир',52:'Дзюдо',53:'Лакросс',54:'Лог-роллинг',55:'Санный спорт',56:'Мотогонки',57:'Гонки на собачьих упряжках',58:'NASCAR',59:'Вольная борьба',60:'Параллельные брусья',61:'Лазание по шесту',62:'Шестовая акробатика',63:'Прыжки с шестом',64:'Поло',65:'Конь',66:'Кольца',67:'Скалолазание',68:'Роллер-дерби',69:'Роллер-спидскейтинг',70:'Академическая гребля',71:'Регби',72:'Парусные гонки',73:'Толкание ядра',74:'Шаффлборд',75:'Гонки с коляской',76:'Прыжки с трамплина',77:'Сёрфинг в небе',78:'Парашютный спорт',79:'Сноубординг',80:'Гонки на снегоходах',81:'Конькобежный спорт',82:'Борьба с быком',83:'Сумо',84:'Сёрфинг',85:'Плавание',86:'Настольный теннис',87:'Теннис',88:'Велотрек',89:'Трапеция',90:'Перетягивание каната',91:'Алтимат',92:'Разновысокие брусья',93:'Волейбол',94:'Водный велосипед',95:'Водное поло',96:'Тяжёлая атлетика',97:'Баскетбол на колясках',98:'Гонки на колясках',99:'Полет в вингсьюте'}

# ====== ТРАНСФОРМАЦИЯ (как при обучении) ======
TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# ====== ЗАГРУЗКА МОДЕЛИ ИЗ HUGGING FACE ======
@st.cache_resource
def download_model_from_hf():
    """Скачивает модель с Hugging Face"""
    model_url = "https://huggingface.co/Scatana/nn_streamlit/resolve/main/resnet101_model.pth"
    model_path = "resnet101_model.pth"
    
    if not os.path.exists(model_path):
        with st.spinner("📥 Скачивание модели с Hugging Face..."):
            response = requests.get(model_url, stream=True)
            response.raise_for_status()
            
            with open(model_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
        st.success("✅ Модель загружена!")
    
    return model_path

@st.cache_resource
def load_model():
    """Загружает модель для классификации спорта"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Скачиваем модель с Hugging Face
    model_path = download_model_from_hf()
    
    model = resnet101(weights=None)
    model.fc = nn.Linear(2048, 100)
    
    state_dict = torch.load(model_path, map_location=device)
    
    # Очистка ключей от префиксов
    clean_state_dict = {}
    for k, v in state_dict.items():
        k = k.replace("model.", "").replace("module.", "").replace("backbone.", "").replace("encoder.", "")
        clean_state_dict[k] = v
    
    model.load_state_dict(clean_state_dict, strict=False)
    model.to(device)
    model.eval()
    
    return model, device

# ====== ФУНКЦИИ ПРЕДСКАЗАНИЯ ======
def predict_top5(image: Image.Image, model: torch.nn.Module, device: torch.device) -> list:
    """
    Возвращает топ-5 предсказаний для изображения
    
    Args:
        image: PIL Image
        model: загруженная модель
        device: устройство (cuda/cpu)
    
    Returns:
        list of tuples: [(class_name, probability), ...]
    """
    image_tensor = TRANSFORM(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(image_tensor)
        probs = torch.softmax(outputs, dim=1)
        top5_prob, top5_idx = torch.topk(probs, 5, dim=1)
    
    top5 = [(IDX_TO_CLASS[idx.item()], prob.item()) 
            for idx, prob in zip(top5_idx[0], top5_prob[0])]
    return top5

def predict(image: Image.Image, model: torch.nn.Module, device: torch.device) -> tuple:
    """
    Возвращает лучшее предсказание
    
    Returns:
        tuple: (class_name, confidence)
    """
    top5 = predict_top5(image, model, device)
    return top5[0][0], top5[0][1]

# ====== ФУНКЦИЯ ДЛЯ STREAMLIT ВКЛАДКИ ======
def show_classifier_tab():
    """
    Отображает вкладку классификатора в Streamlit приложении
    Поддерживает загрузку нескольких изображений
    Модель загружается из Hugging Face
    """
    st.markdown("### 🏆 Классификация видов спорта")
    st.markdown("Загрузите одно или несколько изображений для определения вида спорта")
    
    # Загрузка модели из Hugging Face
    with st.spinner("🔄 Загрузка модели из Hugging Face... Это может занять время при первом запуске"):
        try:
            model, device = load_model()
            st.success(f"✅ Модель загружена | Устройство: {device}")
        except Exception as e:
            st.error(f"❌ Ошибка загрузки модели: {e}")
            st.info("Проверьте подключение к интернету и повторите попытку")
            return
    
    # Настройки
    with st.expander("⚙️ Настройки", expanded=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            show_top5 = st.checkbox("🔍 Показывать топ-5 предсказаний", value=True)
        with col2:
            show_info = st.checkbox("ℹ️ Показывать информацию о модели", value=False)
        with col3:
            images_per_row = st.selectbox("Изображений в строке", [2, 3, 4], index=1)
    
    # Информация о модели
    if show_info:
        with st.expander("ℹ️ Информация о модели", expanded=True):
            st.write(f"**Источник:** Hugging Face (Scatana/nn_streamlit)")
            st.write(f"**Нормализация:** mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]")
            st.write(f"**Размер входа:** 224x224")
            st.write(f"**Классов:** 100")
            st.write(f"**Архитектура:** ResNet101")
            st.write(f"**Устройство:** {device}")
    
    # Загрузка нескольких изображений
    uploaded_files = st.file_uploader(
        "📸 Загрузите изображения (до 20 шт)", 
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        key="sport_classifier"
    )
    
    if uploaded_files:
        if len(uploaded_files) > 20:
            st.warning(f"⚠️ Загружено {len(uploaded_files)} файлов. Будет обработано только 20.")
            uploaded_files = uploaded_files[:20]
        
        st.info(f"📁 Загружено файлов: {len(uploaded_files)}")
        
        # Прогресс-бар
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Обработка всех изображений
        results = []
        for idx, uploaded_file in enumerate(uploaded_files):
            status_text.text(f"Обработка {idx + 1} из {len(uploaded_files)}: {uploaded_file.name}")
            
            try:
                image = Image.open(uploaded_file).convert("RGB")
                top5 = predict_top5(image, model, device)
                results.append({
                    'filename': uploaded_file.name,
                    'image': image,
                    'top5': top5,
                    'main_label': top5[0][0],
                    'confidence': top5[0][1],
                    'success': True
                })
            except Exception as e:
                results.append({
                    'filename': uploaded_file.name,
                    'image': None,
                    'top5': [],
                    'main_label': f"Ошибка: {str(e)}",
                    'confidence': 0.0,
                    'success': False
                })
            
            progress_bar.progress((idx + 1) / len(uploaded_files))
        
        status_text.empty()
        progress_bar.empty()
        
        # Отображение результатов
        st.markdown("---")
        st.markdown(f"### 📊 Результаты анализа ({len(results)} изображений)")
        
        # Статистика в начале
        successful = [r for r in results if r['success']]
        if successful:
            labels = [r['main_label'] for r in successful]
            label_counts = Counter(labels)
            
            col_stat1, col_stat2 = st.columns(2)
            with col_stat1:
                st.metric("✅ Успешно обработано", f"{len(successful)}/{len(results)}")
            with col_stat2:
                st.metric("🏆 Уникальных видов спорта", len(label_counts))
            
            with st.expander("📈 Статистика по видам спорта", expanded=False):
                for label, count in label_counts.most_common():
                    st.write(f"**{label}:** {count} изображений")
        
        # Отображение результатов сеткой
        st.markdown("#### 🖼️ Результаты предсказаний")
        
        # Создаём колонки для сетки
        for idx in range(0, len(results), images_per_row):
            cols = st.columns(images_per_row)
            for col_idx in range(images_per_row):
                result_idx = idx + col_idx
                if result_idx < len(results):
                    result = results[result_idx]
                    with cols[col_idx]:
                        with st.container():
                            if result['success']:
                                # Изображение
                                st.image(result['image'], caption=result['filename'][:30], use_container_width=True)
                                
                                # Основной результат с цветом
                                conf = result['confidence']
                                if conf > 0.7:
                                    st.success(f"🏆 **{result['main_label']}**")
                                elif conf > 0.5:
                                    st.warning(f"🏅 **{result['main_label']}**")
                                else:
                                    st.info(f"❓ **{result['main_label']}**")
                                st.caption(f"🎯 Уверенность: {conf:.2%}")
                                
                                # Топ-5
                                if show_top5 and result['top5']:
                                    with st.expander("🔍 Топ-5", expanded=False):
                                        for label, conf in result['top5']:
                                            st.write(f"**{label}**")
                                            st.progress(conf, text=f"{conf:.2%}")
                            else:
                                st.error(f"❌ **{result['filename']}**")
                                st.write(result['main_label'])
                            
                            st.markdown("---")
        
        # Кнопка для очистки
        if st.button("🗑️ Очистить все изображения", use_container_width=True):
            st.rerun()
        
        st.balloons()
    else:
        st.info("👆 Загрузите изображения для начала анализа")

# ====== ОТДЕЛЬНЫЙ ЗАПУСК ======
if __name__ == "__main__":
    st.set_page_config(page_title="Спортивная классификация", page_icon="🏅", layout="wide")
    show_classifier_tab()