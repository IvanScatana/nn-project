import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# --- 1. Загрузка модели (кэшируется для повышения производительности) ---
@st.cache_resource
def load_model():
    # Шаг 1: Воссоздаем архитектуру модели (как при обучении)
    # Загружаем предобученный ConvNeXt (как вы делали)
    model = models.convnext_base(weights=None)  # Важно: weights=None

    # Шаг 2: Меняем последний слой классификатора под вашу задачу (6 классов)
    num_classes = 6
    model.classifier[2] = nn.Linear(1024, num_classes)

    # Шаг 3: Загружаем ваши сохраненные веса
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Убедитесь, что путь к файлу указан верно
    state_dict = torch.load('best_model_0.9012_epoch1.pth', map_location=device)
    model.load_state_dict(state_dict)

    model = model.to(device)
    model.eval()  # Переводим модель в режим оценки (отключаем Dropout/BatchNorm)
    return model, device

# --- 2. Предобработка изображения (трансформации, как при обучении) ---
def preprocess_image(image):
    # Определяем те же трансформации, что и на валидации
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),  # Приводим к нужному размеру
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Применяем трансформации к загруженному изображению
    image_tensor = preprocess(image)
    # Добавляем размерность батча: (C, H, W) -> (1, C, H, W)
    image_batch = image_tensor.unsqueeze(0)
    return image_batch

# --- 3. Функция для предсказания ---
def predict(image, model, device, class_names):
    # Предобрабатываем изображение и отправляем на устройство (CPU/GPU)
    image_tensor = preprocess_image(image).to(device)

    with torch.no_grad():  # Отключаем вычисление градиентов
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        predicted_class_idx = torch.argmax(probabilities, dim=1).item()

    return predicted_class_idx, probabilities

# --- 4. Создание интерфейса Streamlit ---
def main():
    st.title("Классификатор изображений")
    st.write("Загрузите изображение, чтобы определить его класс.")

    # Загружаем модель (кэширование гарантирует загрузку один раз)
    model, device = load_model()

    # Список имен классов в правильном порядке
    class_names = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']

    # Виджет для загрузки файла
    uploaded_file = st.file_uploader("Выберите изображение...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Открываем изображение с помощью PIL
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption='Загруженное изображение', use_column_width=True)

        st.write("Классификация...")
        # Делаем предсказание
        predicted_idx, probs = predict(image, model, device, class_names)

        # Выводим результат
        st.success(f"Предсказанный класс: **{class_names[predicted_idx]}**")
        st.write(f"Уверенность модели: `{probs[0][predicted_idx].item():.2%}`")

if __name__ == "__main__":
    main()