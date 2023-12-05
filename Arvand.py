import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle
import io
import numpy as np
pickle_in = open("Arvand_Gradient_Boosting_Scaled.pkl", "rb")
model = pickle.load(pickle_in)

df = pd.read_csv('arvand.csv')

st.title("Кредитный Скоринг")

gender = st.selectbox("Пол:", ['Мужской', 'Женский'])
married_status = st.selectbox("Семейное положение:", ['Оиладор', 'Чудошуда', 'Беоила', 'Бевамард (бевазан)'])

nationality = st.selectbox("Национальность:", ['Точик', 'Узбек', 'Киргиз', 'Рус', 'Тотор', 'Другие', 'Украин'])
educ = st.selectbox("Образование:", ['Миёна', 'Миёнаи махсус', 'Оли', 'Олии нопурра', 'Миёнаи нопурра', 'Аспирантура'])
family_size = st.number_input("Размер семьи:", min_value=0, max_value=100)
bus_experience = st.number_input("Опыт в бизнесе:" , min_value=0, max_value=100)
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
loan_amount = st.number_input("Сумма кредита:", min_value=0, max_value=1000000000)
loan_term = st.number_input("Срок кредита (в месяцах):", min_value=0, max_value=1000)
pledge = st.selectbox("Залог:", ['без залога', 'поручительство', 'недвижимость', 'движимое имущество'])
credit_history = st.number_input("Количество кредитных историй:", min_value=0, max_value=10000)
net_profit = st.number_input("Чистая прибыль:", value=0)
age = st.number_input("Возраст:", min_value=18, max_value=100, value=25)

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
s = str()
dp = pd.DataFrame()
input_data['Gender'] = input_data['Gender'].map({'Мужской': 0, 'Женский': 1})
for i in input_data:
    s += i 
    s += '_'
    for j in input_data[i]:
        s += str(j)
    dp[s] = 0;
    s = str()
    
for i in dp:
    for j in df:
        if i == j:
            df[j] = np.uint8(1)
        else:
            df[j] = np.uint8(0)
    
df_num = input_data.select_dtypes(exclude=['object'])

for i in df_num:
    for j in df:
        if i == j:
            df[j] = df_num[i]
     
    

credit_score = model.predict(df)[0]
if st.button("Предсказать"):

    probability = model.predict_proba(df)[:, 1]
    st.subheader("Решение по выдаче кредита:")
    if probability >= 55:
        st.success("Можно выдать кредит.")
    else:
        st.error("Отказать в выдаче кредита.")

    st.subheader("Вероятность возврата кредита:")
    st.write(f"{probability[0]*100:.2f}%")
