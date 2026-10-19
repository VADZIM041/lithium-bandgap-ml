import streamlit as st
import pandas as pd
from xgboost import XGBRegressor

# Настройка страницы
st.set_page_config(page_title="ML-Предсказатель Материалов", page_icon="🔋")
st.title("🔋 Предсказание Band Gap для Литиевых материалов")
st.write("Введите химическую формулу (например, LiFePO4), и нейросеть предскажет ширину запрещенной зоны.")

@st.cache_resource
def train_model():
    # Загружаем УЖЕ ГОТОВЫЙ датасет с признаками matminer
    df = pd.read_csv("lithium_ml_dataset_final.csv")
    
    # Кристаллические системы (One-Hot)
    crystal_dummies = pd.get_dummies(df['crystal_system'], prefix='crystal')
    
    # Исключаем ненужные текстовые колонки и целевую переменную
    cols_to_drop = ['material_id', 'formula', 'band_gap', 'crystal_system', 'spacegroup']
    feature_cols = [c for c in df.columns if c not in cols_to_drop]
    
    X = pd.concat([df[feature_cols], crystal_dummies], axis=1)
    y = df['band_gap']
    
    # Обучение
    model = XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=6, random_state=42)
    model.fit(X, y)
    
    return model, df, X.columns

model, df, feature_columns = train_model()

# Интерфейс
formula_input = st.text_input("Химическая формула:", "LiFePO4")

if st.button("Предсказать!"):
    material_row = df[df['formula'] == formula_input]
    
    if material_row.empty:
        st.error(f"Материала {formula_input} нет в базе данных проекта.")
    else:
        # Берем строку с признаками этого материала из готового датасета
        row_features = material_row.drop(columns=['material_id', 'formula', 'band_gap', 'spacegroup'])
        crystal_sys = row_features['crystal_system'].values[0]
        
        # Убираем текстовое название кристалла
        row_features = row_features.drop(columns=['crystal_system'])
        
        # Добавляем One-Hot для кристалла
        crystal_feature = pd.DataFrame(0, index=[0], columns=[c for c in feature_columns if c.startswith('crystal_')])
        col_name = f"crystal_{crystal_sys}"
        if col_name in crystal_feature.columns:
            crystal_feature[col_name] = 1
            
        # Собираем всё и выравниваем
        X_new = pd.concat([row_features.reset_index(drop=True), crystal_feature], axis=1)
        X_new = X_new.reindex(columns=feature_columns, fill_value=0)
        
        # Предсказание
        prediction = model.predict(X_new)[0]
        real_value = material_row['band_gap'].values[0]
        
        st.success(f"### Предсказанный Band Gap: **{prediction:.3f} эВ**")
        st.info(f"Реальное значение в базе: {real_value:.3f} эВ (Ошибка: {abs(prediction - real_value):.3f} эВ)")
