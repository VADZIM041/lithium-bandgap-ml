import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from pymatgen.core import Composition

# Настройка страницы
st.set_page_config(page_title="ML-Предсказатель Материалов", page_icon="🔋")
st.title("🔋 Предсказание Band Gap для Литиевых материалов")
st.write("Введите химическую формулу (например, LiFePO4), и нейросеть предскажет ширину запрещенной зоны.")

# Загрузка и обучение (кэшируем, чтобы не обучать при каждом клике)
@st.cache_resource
def train_model():
    df = pd.read_csv("lithium_ml_dataset.csv")
    def get_elements(formula):
        try: return Composition(formula).get_el_amt_dict()
        except: return {}
    elements_df = pd.json_normalize(df['formula'].apply(get_elements)).fillna(0)
    crystal_dummies = pd.get_dummies(df['crystal_system'], prefix='crystal')
    X = pd.concat([df[['energy_above_hull', 'formation_energy', 'volume', 'density', 'nsites']], elements_df, crystal_dummies], axis=1)
    y = df['band_gap']
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model, df, elements_df.columns, crystal_dummies.columns

model, df, el_cols, crystal_cols = train_model()

# Поле ввода
formula_input = st.text_input("Химическая формула:", "LiFePO4")

if st.button("Предсказать!"):
    material_row = df[df['formula'] == formula_input]
    
    if material_row.empty:
        st.error(f"Материала {formula_input} нет в базе данных проекта. Нейросеть пока не умеет предсказывать с нуля без базовых физических свойств.")
    else:
        idx = material_row.index[0]
        el_dict = Composition(formula_input).get_el_amt_dict()
        el_features = pd.DataFrame([el_dict]).reindex(columns=el_cols, fill_value=0)
        
        crystal_sys = material_row['crystal_system'].values[0]
        crystal_feature = pd.DataFrame(0, index=[0], columns=crystal_cols)
        col_name = f"crystal_{crystal_sys}"
        if col_name in crystal_feature.columns:
            crystal_feature[col_name] = 1
            
        num_features = material_row[['energy_above_hull', 'formation_energy', 'volume', 'density', 'nsites']].reset_index(drop=True)
        X_new = pd.concat([num_features, el_features, crystal_feature], axis=1)
        
        prediction = model.predict(X_new)[0]
        real_value = material_row['band_gap'].values[0]
        
        st.success(f"### Предсказанный Band Gap: **{prediction:.3f} эВ**")
        st.info(f"Реальное значение в базе: {real_value:.3f} эВ (Ошибка: {abs(prediction - real_value):.3f} эВ)")
