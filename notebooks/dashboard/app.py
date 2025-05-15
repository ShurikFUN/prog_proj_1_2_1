# python -m streamlit run /Users/shurikfun/Documents/PyWorkITMO/projects/reddit_analysis/notebooks/dashboard/app.py
from pydoc import describe

import streamlit as st
import praw
import pandas as pd
from datetime import datetime
import seaborn as sns
import matplotlib
from matplotlib import pyplot as plt
from stop_words import get_stop_words #nltk не хотел устанавливать stop-words
from streamlit import session_state
from wordcloud import WordCloud
import io

if "page" not in st.session_state:
    st.session_state.page = "Главная"


@st.cache_data
def load_data():
    df = pd.read_csv('/Users/shurikfun/Documents/PyWorkITMO/projects/reddit_analysis/notebooks/data/reddit_posts.csv')
    return df

st.sidebar.title("Меню")
if st.sidebar.button("О проекте"):
    st.session_state.page = "О проекте"
if st.sidebar.button("Данные"):
    st.session_state.page = "Данные"
if st.sidebar.button("EDA"):
    st.session_state.page = "EDA"
if st.sidebar.button("Тренды & закономерности"):
    st.session_state.page = "Тренды & закономерности"
if st.sidebar.button("Выводы & рекомендации"):
    st.session_state.page = "Выводы & рекомендации"


if st.session_state.page == "О проекте":
    st.title("Reddit posts analysis")
    st.markdown("""
        Мы собираемся провести анализ постов Reddit для понимания общественных интересов, потребностей и трендов. 
        
        В наших планах узнать какие слова чаще всего повторяются в заголовках постов, в какие дни недели пользователи наиболее активно постят и как распределены комментарии. Данные буду получать с помощью библиотеки Reddit 'praw', для этого создам приложение в Reddit и получу доступ к сбору данных. Здесь расскажите, откуда будете брать данные, что собираетесь анализировать
        """)

# -------------------------
# обзор данных
# ------------------------
def data():
    # ввожу данные для авторизации и получения доступа
    reddit = praw.Reddit(
        client_id='S-OA2QSB5sLdEXJQo1QuDQ',
        client_secret='ry627ce5h-Ud-kqp_Jz5YQuAnRiPcg',
        user_agent='Shurik_project_v1, trend-analysis'
    )

    subreddit_name = 'AskReddit'  # выбор сабреддита
    subreddit = reddit.subreddit(subreddit_name)

    posts = []

    # Получаем посты за последний месяц (500 штук)
    for post in subreddit.top(time_filter='month', limit=500):
        posts.append({
            'id': post.id,
            'title': post.title,
            'score': post.score,
            'num_comments': post.num_comments,
            'created_utc': post.created_utc,
            'url': post.url,
        })

    # создаю датафрейм
    df = pd.DataFrame(posts)
    df.to_csv("~/Documents/PyWorkITMO/projects/reddit_analysis/notebooks/data/reddit_posts.csv", index=False)

    # удаление строк с пустыми заголовками
    df = df[df['title'].notna()]

    # удаление аномалий
    df = df[(df['score'] >= 0) & (df['num_comments'] >= 0)]

    # редакция заголовков
    df['title'] = df['title'].str.strip().str.lower().str.findall(r'\b[a-zA-Z]+\b').str.join(' ')

    # преобразую время
    df['created_utc'] = pd.to_datetime(df['created_utc'], unit='s')

    # статистика
    describe= df.describe()
    info = df.info()
    isna = df.isna().sum()

if st.session_state.page == "Данные":
    st.title("Данные")
    if st.button("Обновить данные с Reddit"):
        data()
    df = load_data()

    st.subheader("Анализ данных")
    buffer = io.StringIO()
    df.info(buf=buffer)
    info = buffer.getvalue()
    st.text(info)
    na_counts = df.isna().sum()

    st.subheader("Пропущенные значения по столбцам:")
    st.write(na_counts)

    # Показываем превью
    st.subheader("Превью данных")
    n_rows = st.slider("Сколько строк показать?", min_value=5, max_value=len(df))
    st.dataframe(df.head(n_rows), use_container_width=True)


# -----------------------------
# EDA
# ---------------------------
def hist(df):
    fig, axs = plt.subplots(1, 1, figsize=(12, 8))

    sns.histplot(data=df, x='score', bins=50, ax=axs, color='#FF4500', edgecolor='black')  # гистограмма
    axs.set_xlabel("Score")
    axs.set_ylabel("Количество постов")
    axs.set_title("Распределение постов по score")
    st.pyplot(fig)

def cloud(df):
    # обработка для узконаправленного анализа (облакло слов)
    title_words = ' '.join(df['title'].astype(str).str.split().sum())
    title_words = title_words.split()

    # уберу стоп слова
    extra_stopwords = {'people', 'something', 'someone', 'thing', 'like', 'just', 'actually', 'one', 'non', 'person','first', 'reddit', 'biggest', 'best', 'completely', 'whats', 'don', 'will', 'every', 'back', 'even', 'really', 'somehow'}
    stop_words = set(get_stop_words('english')).union(extra_stopwords)
    title_words_filtered = [word for word in title_words if word not in stop_words and len(word) > 2]
    title_words_ready = ' '.join(title_words_filtered)

    wordcloud = WordCloud(width=800, height=300, background_color='white', colormap='Reds', min_font_size=12, max_font_size=70).generate(title_words_ready)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bicubic')
    plt.axis('off')
    fig = plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bicubic')
    plt.axis('off')
    st.pyplot(fig)

def line(df):
    # преобразую время
    df['created_utc'] = pd.to_datetime(df['created_utc'], unit='s')

    # выделяем дни недели
    df['weekday'] = df['created_utc'].dt.day_name()

    # порядок
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    weekday_counts = df['weekday'].value_counts().reindex(days)

    fig, ax = plt.subplots(figsize=(8, 5))  # <-- тут создаётся fig

    ax.plot(weekday_counts.index, weekday_counts.values, marker='o', color='#FF4500')
    ax.set_title('Активность пользователей по дням недели')
    ax.set_xlabel('День недели')
    ax.set_ylabel('Количество постов')
    ax.grid(True)
    fig.tight_layout()

    st.pyplot(fig)

def box(df):
    fig, ax = plt.subplots()

    ax.boxplot(df['num_comments'], patch_artist=True, boxprops=dict(facecolor='#FF4500', color='black'), medianprops=dict(color='white'), showfliers=False)
    ax.set_title('Анализ количества комментариев')
    ax.set_ylabel('Количество')
    ax.set_xticks([])
    st.pyplot(fig)


if st.session_state.page == "EDA":
    df = load_data()
    st.header("EDA")
    if st.button("Показать гистограмму"):
        hist(df)

    if st.button("Показать облако слов"):
        cloud(df)

    if st.button("Показать линейный график по дням недели"):
        line(df)

    if st.button("Показать ящик с усами (комментарии)"):
        box(df)



def cloud_v2(df):
    # Установим интерактивный диапазон выбора дат
    df['created_utc'] = pd.to_datetime(df['created_utc'], unit='s')  # если это timestamp

    min_date = df['created_utc'].min()
    max_date = df['created_utc'].max()
    start_date, end_date = st.date_input("Выберите диапазон дат", [min_date.date(), max_date.date()])

    # Отфильтруем
    filtered_df = df[(df['created_utc'] >= pd.to_datetime(start_date)) & (df['created_utc'] <= pd.to_datetime(end_date))]

    # Обработка текста
    title_words = ' '.join(filtered_df['title'].astype(str).str.split().sum())
    title_words = title_words.split()

    # Уберем стоп-слова
    extra_stopwords = {'people', 'something', 'someone', 'thing', 'like', 'just', 'actually', 'one', 'non', 'person','first', 'reddit', 'biggest', 'best', 'completely', 'whats', 'don', 'will', 'every', 'back','even', 'really', 'somehow'}
    stop_words = set(get_stop_words('english')).union(extra_stopwords)
    title_words_filtered = [word for word in title_words if word.lower() not in stop_words and len(word) > 2]
    title_words_ready = ' '.join(title_words_filtered)

    wordcloud = WordCloud(width=800, height=300,background_color='white',colormap='Reds',min_font_size=12, max_font_size=70
    ).generate(title_words_ready)

    fig = plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bicubic')
    plt.axis('off')
    st.pyplot(fig)

if st.session_state.page == "Тренды & закономерности":
    st.title("Тренды & закономерности")
    df = load_data()
    st.subheader("Облако слов с гибкой датой")
    st.markdown("**Не стоит выбирать даты дальше месяца от сегодняшней и меньше трех дней**")
    cloud_v2(df)


# ------------------------
# Тренды & закономерности
# ------------------------

if st.session_state.page == "Выводы & рекомендации":
    st.title("Выводы и рекомендации")
    st.markdown("""
    **Выводы интересные нам:**
    
    В этом месяце на гребне популярности оказались трамп и америка. Примечательно, что в такие, казалось бы, активные и нестабильные времена (войны, терракты) люди продолжают в основном обсуждать какие-то бытовые вещи.
    
    **Формальные выводы:**
    
    В целом, проведённый анализ показал, что популярность контента на Reddit подчиняется известным законам социальной динамики, но при этом есть свои особенности, такие как день недели с самой большой частотой выкладывания постов. Эти результаты могут стать основой для дальнейших исследований, а также для более эффективного планирования публикаций и маркетинговых стратегий не только на самой платформе Reddit, но и в целом на других рынках, поскольку обладают необходимой информацией о трендах общества. Таким образом, исследование получилось на стыке социологии, бизнеса и информатики и может быть использовано в каждой из этих сфер.
    """)


