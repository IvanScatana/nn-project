# import streamlit as st
# import torch
# import torch.nn as nn
# from torchvision import models, transforms
# from PIL import Image
# import pandas as pd

# # ========== 1. КОНФИГУРАЦИЯ ==========
# # Прямая ссылка на файл модели в публичном репозитории Hugging Face
# # Замените на вашу ссылку (пример: Scatana/nn_streamlit)
# MODEL_URL = "https://huggingface.co/Scatana/nn_streamlit/resolve/main/resnet101_model.pth"

# # Список классов в правильном порядке (индексы 0..5)
# CLASS_NAMES_RESNET101 = [
#     'air hockey', 'ampute football', 'archery', 'arm wrestling', 'axe throwing',
#     'balance beam', 'barell racing', 'baseball', 'basketball', 'baton twirling',
#     'bike polo', 'billiards', 'bmx', 'bobsled', 'bowling', 'boxing', 'bull riding',
#     'bungee jumping', 'canoe slamon', 'cheerleading', 'chuckwagon racing', 'cricket',
#     'croquet', 'curling', 'disc golf', 'fencing', 'field hockey', 'figure skating men',
#     'figure skating pairs', 'figure skating women', 'fly fishing', 'football',
#     'formula 1 racing', 'frisbee', 'gaga', 'giant slalom', 'golf', 'hammer throw',
#     'hang gliding', 'harness racing', 'high jump', 'hockey', 'horse jumping',
#     'horse racing', 'horseshoe pitching', 'hurdles', 'hydroplane racing', 'ice climbing',
#     'ice yachting', 'jai alai', 'javelin', 'jousting', 'judo', 'lacrosse', 'log rolling',
#     'luge', 'motorcycle racing', 'mushing', 'nascar racing', 'olympic wrestling',
#     'parallel bar', 'pole climbing', 'pole dancing', 'pole vault', 'polo', 'pommel horse',
#     'rings', 'rock climbing', 'roller derby', 'rollerblade racing', 'rowing', 'rugby',
#     'sailboat racing', 'shot put', 'shuffleboard', 'sidecar racing', 'ski jumping',
#     'sky surfing', 'skydiving', 'snow boarding', 'snowmobile racing', 'speed skating',
#     'steer wrestling', 'sumo wrestling', 'surfing', 'swimming', 'table tennis', 'tennis',
#     'track bicycle', 'trapeze', 'tug of war', 'ultimate', 'uneven bars', 'volleyball',
#     'water cycling', 'water polo', 'weightlifting', 'wheelchair basketball',
#     'wheelchair racing', 'wingsuit flying'
# ]

# # Устройство для инференса (CPU или GPU)
# DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# # ========== 2. ЗАГРУЗКА МОДЕЛИ (КЭШИРУЕТСЯ) ==========
# @st.cache_resource
# def load_model():
#     """
#     Загружает архитектуру модели и веса из Hugging Face.
#     Результат кэшируется, чтобы не скачивать модель при каждом взаимодействии.
#     """
#     # 1. Создаём архитектуру так же, как при обучении
#     model = models.convnext_small(pretrained=False)
#     model.classifier[2] = nn.Linear(in_features=768, out_features=4, bias=True)
    
#     # 2. Загружаем state_dict из URL
#     #    map_location временно ставим CPU, потом перенесём на нужное устройство
#     state_dict = torch.hub.load_state_dict_from_url(
#         MODEL_URL,
#         map_location='cpu',
#         file_name='best_model.pth'
#     )
#     model.load_state_dict(state_dict)
    
#     # 3. Переводим модель в режим оценки и на целевое устройство
#     model.eval()
#     model.to(DEVICE)
#     return model


# # ========== 3. ПРЕДОБРАБОТКА ИЗОБРАЖЕНИЯ ==========
# def preprocess_image(image: Image.Image):
#     """
#     Преобразует PIL Image в тензор, готовый для подачи в модель.
#     Используются те же трансформации, что и при валидации.
#     """
#     transform = transforms.Compose([
#         transforms.Resize((224, 224)),
#         transforms.ToTensor(),
#         transforms.Normalize(
#             mean=[0.485, 0.456, 0.406],
#             std=[0.229, 0.224, 0.225]
#         )
#     ])
#     # Добавляем размерность батча (1, 3, 224, 224)
#     img_tensor = transform(image).unsqueeze(0)
#     return img_tensor.to(DEVICE)


# # ========== 4. ФУНКЦИЯ ПРЕДСКАЗАНИЯ ДЛЯ ОДНОГО ИЗОБРАЖЕНИЯ ==========
# def predict_single(image_tensor, model):
#     """
#     Выполняет инференс модели и возвращает предсказанный индекс класса и вероятности.
#     """
#     with torch.no_grad():
#         outputs = model(image_tensor)
#         probabilities = torch.nn.functional.softmax(outputs, dim=1)
#         predicted_idx = torch.argmax(probabilities, dim=1).item()
#     return predicted_idx, probabilities.cpu().numpy()[0]


# # ========== 5. ОСНОВНОЙ ИНТЕРФЕЙС STREAMLIT ==========
# def main():
#     st.set_page_config(page_title="Классификатор изображений (множественная загрузка)", layout="wide")
#     st.title("🏞️ Классификатор изображений")
#     st.markdown("Загрузите **одно или несколько** фотографий, и модель определит, что на них изображено: "
#                 "здание, лес, ледник, гора, море или улица.")

#     # Загружаем модель (один раз, кэшируется)
#     with st.spinner("Загрузка модели... Пожалуйста, подождите."):
#         try:
#             model = load_model()
#             st.success("Модель успешно загружена!")
#         except Exception as e:
#             st.error(f"Ошибка загрузки модели: {e}")
#             st.stop()

#     # Виджет загрузки нескольких изображений
#     uploaded_files = st.file_uploader(
#         "Выберите одно или несколько изображений (JPG, JPEG, PNG)",
#         type=["jpg", "jpeg", "png"],
#         accept_multiple_files=True
#     )

#     if uploaded_files and len(uploaded_files) > 0:
#         st.subheader(f"Загружено {len(uploaded_files)} изображений")

#         # Кнопка для запуска классификации всех изображений
#         if st.button("Классифицировать все"):
#             results = []
#             progress_bar = st.progress(0)
#             status_text = st.empty()

#             for i, uploaded_file in enumerate(uploaded_files):
#                 # Открываем изображение
#                 image = Image.open(uploaded_file).convert('RGB')
                
#                 # Предобработка и предсказание
#                 img_tensor = preprocess_image(image)
#                 pred_idx, probs = predict_single(img_tensor, model)
#                 predicted_class = CLASS_NAMES[pred_idx]
#                 confidence = probs[pred_idx] * 100
                
#                 # Сохраняем результат
#                 results.append({
#                     "filename": uploaded_file.name,
#                     "predicted_class": predicted_class,
#                     "confidence": f"{confidence:.2f}%",
#                     "image": image
#                 })
                
#                 # Обновляем прогресс
#                 progress_bar.progress((i + 1) / len(uploaded_files))
#                 status_text.text(f"Обработано {i+1} из {len(uploaded_files)}")
            
#             progress_bar.empty()
#             status_text.empty()
#             st.success("Классификация завершена!")

#             # Отображение результатов в виде таблицы и галереи
#             st.subheader("Результаты классификации")
            
#             # Таблица с результатами (без изображений)
#             df = pd.DataFrame(results)
#             df_display = df.drop(columns=['image'])
#             st.dataframe(df_display, use_container_width=True)
            
#             # Галерея: покажем каждое изображение с подписью
#             st.subheader("Галерея с предсказаниями")
#             cols = st.columns(3)  # 3 колонки для отображения
#             for idx, res in enumerate(results):
#                 col = cols[idx % 3]
#                 with col:
#                     st.image(res["image"], caption=f"{res['filename']}", use_container_width=True)
#                     st.markdown(f"**{res['predicted_class']}**  \n  Уверенность: {res['confidence']}")
#             st.caption("Примечание: Уверенность показывает, насколько модель уверена в своём выборе.")


# if __name__ == "__main__":
#     main()