import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle
import io
import joblib
import numpy as np
from sklearn.preprocessing import LabelEncoder
pickle_in = open("Arvand_Gradient_Boosting.pkl", "rb")
model = pickle.load(pickle_in)
encoded_columns = joblib.load('encoded_columns.joblib')

def apply_custom_style():
    st.markdown(
        """
        <style>
            .st-bf {
                border-radius: 15px;
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
                padding: 20px;
                margin-bottom: 20px;
                background-color: #f5f5f5;
            }
            .st-bf header {
                font-size: 1.5em;
                color: #333;
                margin-bottom: 15px;
            }
            .st-bf label {
                color: #555;
            }
            .st-bf .stButton button {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 10px 20px;
                text-align: center;
                text-decoration: none;
                display: inline-block;
                font-size: 16px;
                border-radius: 5px;
                cursor: pointer;
                transition-duration: 0.3s;
            }
            .st-bf .stButton button:hover {
                background-color: #45a049;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


apply_custom_style()

st.title("Кредитный Скоринг")


st.markdown("<div class='st-bf'>", unsafe_allow_html=True)
st.header("Персональная информация")
gender = st.radio("Пол:", ['Мужской', 'Женский'])
married_status = st.radio("Семейное положение:", ['Оиладор', 'Чудошуда', 'Беоила', 'Бевамард (бевазан)'])
nationality = st.radio("Национальность:", ['Точик', 'Узбек', 'Киргиз', 'Рус', 'Тотор', 'Другие', 'Украин'])
educ = st.radio("Образование:", ['Миёна', 'Миёнаи махсус', 'Оли', 'Олии нопурра', 'Миёнаи нопурра', 'Аспирантура'])
family_size = st.number_input("Размер семьи:", min_value=0, max_value=100)
bus_experience = st.number_input("Опыт в бизнесе:", min_value=0, max_value=100)
age = st.number_input("Возраст:", min_value=18, max_value=100, value=25)
st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<div class='st-bf'>", unsafe_allow_html=True)
st.header("Финансовая информация")
loan_amount = st.number_input("Сумма кредита:", min_value=0, max_value=1000000000)
loan_term = st.number_input("Срок кредита (в месяцах):", min_value=0, max_value=1000)
credit_history = st.number_input("Количество кредитных историй:", min_value=0, max_value=10000)
net_profit = st.number_input("Чистая прибыль:", value=0)
st.markdown("</div>", unsafe_allow_html=True)


st.markdown("<div class='st-bf'>", unsafe_allow_html=True)
st.header("Детали кредита")
filial = st.selectbox("Филиал:", ['Хучанд', 'Исфара', 'Истаравшан', 'Душанбе', 'Ч. Расулов', 'Панчакент'])
region = st.selectbox("Регион:", ['Ашт', 'Кистакуз', 'Худжанд-Панчшанбе', 'Худжанд-Центр', 'Ифтихор', 'Оббурдон', 'Бустон','Мархамат', 'Сомгор', 'Шарк', 'Дусти',
       'Пунук', 'Уяс', 'Оппон', 'Конибодом', 'Кулканд', 'Ниёзбек',
       'Исфара', 'Каракчикум', 'Ворух', 'Мехнатобод', 'Ободи',
       'Калининобод', 'Шахристон', 'Зафаробод', 'Чашмасор', 'Истаравшан',
       'Х.Алиев', 'Равшан', 'Истаравшан-филиал', 'Гончи', 'Нофароч',
       'Ничони', 'Навкент', 'Некфайз', 'Гули сурх', 'Мучун', 'Душанбе',
       'Турсунзода', 'Вахдат', 'Хисор', 'Сино', 'Рогун', 'Файзобод',
       'Рудаки', 'Спитамен', 'Дж.Расулов', 'Гулякандоз', 'Куруш',
       'Панчакент', 'Ёри', 'Саразм', 'Гусар'])
level = st.selectbox("Уровень клиента:", ['Бовари', 'VIP', 'Хамкори', 'Шарик'])
credit_purpose = st.selectbox("Цель Кредита:", ['Животноводство и молочная продукция', 'Кредит на технику', 'Торговля', 'Кредит на мебель','Кредит на ремонт дома',
       'Обслуживание', 'Медицинский кредит', 'Кишту кор',
       'Производственный кредит', 'Кредит на мероприятия',
       'Кредит на путешествия', 'Кредит на транспорт',
       'Кредит на покупку жилья', 'Переработка молока',
       'Кредит на образование', 'Ремонт рабочего места',
       'Универсальный кредит', 'Кредит на потребительские нужды',
       'Кредит на сельское хозяйство', 'Кредит на животноводство',
       'Сушка фруктов', 'Коммерческий кредит',
       'Другие потребительские кредиты'])
loan_purpose = st.selectbox("Назначение кредита:", ['Бизнес-кредит', 'Потребительский кредит', 'Жилищный кредит', 'Энергосберегающие проекты'])
pledge = st.selectbox("Залог:", ['без залога', 'поручительство', 'недвижимость', 'движимое имущество'])
st.markdown("</div>", unsafe_allow_html=True)

     
input_data = pd.DataFrame({
    'Gender': [gender],
    'FamilySize': [family_size],
    'BusExper': [bus_experience],
    'Сумма кредита': [loan_amount],
    'Срок кредита': [loan_term],
    'Количество кредитных историй': [credit_history],
    'Чистая прибыль': [net_profit],
    'Age': [age],
    'Married': [married_status],
    'Nationality': [nationality],
    'Educ': [educ],
    'Filial': [filial],
    'Region': [region],
    'Уровень клиента': [level],
    'Цель Кредита': [credit_purpose],
    'Назначение': [loan_purpose],
    'Залог': [pledge]  
})

if st.button("Предсказать",key="predict_button"):
    
    input_data['Gender'] = input_data['Gender'].map({'Мужской': 0, 'Женский': 1})

    input_data['Gender'] = input_data['Gender'].astype(int)
    
    df_cat = input_data.select_dtypes(include=['object'])
    df_num = input_data.select_dtypes(exclude=['object'])
    df_encoded = pd.get_dummies(df_cat , dtype=np.uint8 )

    for col in encoded_columns:
        if col not in df_encoded.columns:
            df_encoded[col] = np.uint8(0)

    df_encoded = df_encoded[encoded_columns]    
    df = pd.concat([df_num, df_encoded], axis = True)
        
    #st.dataframe(df)
    credit_score = model.predict(df)[0]
    
    probability = model.predict_proba(df)[:, 0]
    st.markdown("<div class='st-bf'>", unsafe_allow_html=True)
    st.subheader("Решение по выдаче кредита:")

    if credit_score == 0:
        st.success("✅ Можно выдать кредит.")
        #st.balloons()
    else:
        st.error("❌ Отказать в выдаче кредита.")
        requested_loan_amount = input_data['Сумма кредита']
        suggested_amount = 0.8 * requested_loan_amount.iloc[0]
        st.subheader("Предложенная сумма кредита:")
        st.write(f"{suggested_amount:.2f} сомони.")

    st.subheader("Вероятность возврата кредита:")
    st.write(f"{probability[0]*100:.2f}%")
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("Developed by: Muslim")
