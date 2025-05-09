import streamlit as st
import pandas as pd
import numpy as np

st.title("Кредитный Скоринг")

st.header("Персональная информация")
gender = st.radio("Пол:", ['Мужской', 'Женский'])
married_status = st.radio("Семейное положение:", ['Оиладор', 'Чудошуда', 'Беоила', 'Бевамард (бевазан)'])
nationality = st.radio("Национальность:", ['Точик', 'Узбек', 'Киргиз', 'Рус', 'Тотор', 'Другие', 'Украин'])
educ = st.radio("Образование:", ['Миёна', 'Миёнаи махсус', 'Оли', 'Олии нопурра', 'Миёнаи нопурра', 'Аспирантура'])
family_size = st.number_input("Размер семьи:", min_value=0, max_value=100)
bus_experience = st.number_input("Опыт в бизнесе:", min_value=0, max_value=100)
age = st.number_input("Возраст:", min_value=18, max_value=100, value=25)

st.header("Финансовая информация")
loan_amount = st.number_input("Сумма кредита:", min_value=0, max_value=1000000000)
loan_term = st.number_input("Срок кредита (в месяцах):", min_value=0, max_value=1000)
credit_history = st.number_input("Количество кредитных историй:", min_value=0, max_value=10000)
net_profit = st.number_input("Чистая прибыль:", value=0)
st.markdown("</div>", unsafe_allow_html=True)

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

if st.button("Предсказать"):
    # Generate random prediction instead of using the model
    credit_score = np.random.randint(0, 2)  # Randomly 0 (approve) or 1 (deny)
    probability = np.random.uniform(0, 1)   # Random probability between 0 and 1

    st.subheader("Решение по выдаче кредита:")
    if credit_score == 0:
        st.success("✅ Можно выдать кредит.")
    else:
        st.error("❌ Отказать в выдаче кредита.")
        low = 0
        high = input_data['Сумма кредита'].max()  # Use original loan amount as max
        mid = 0
        # Simulate binary search with random logic
        while low <= high:
            mid = (low + high) // 2
            # Randomly decide if this amount is "approved"
            if np.random.randint(0, 2) == 0:
                low = mid + 1
            else:
                high = mid - 1
        suggested_amount = high
        st.subheader("Предложенная сумма кредита:")
        st.write(f"{suggested_amount:.2f} сомони.")

    st.subheader("Вероятность возврата кредита:")
    st.write(f"{probability*100:.2f}%")
    st.markdown("Developed by: Muslim")
