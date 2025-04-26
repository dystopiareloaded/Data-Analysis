# netflix_streamlit_app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import requests

# --- Streamlit Config ---
st.set_page_config(page_title="Netflix EDA Dashboard", layout="wide")
st.title('🎬 Netflix Data Cleaning & EDA Dashboard')

# --- Data Loading ---
def download_data():
    url = "https://raw.githubusercontent.com/dystopiareloaded/Data-Analysis/main/Day1/netflix1.csv"
    r = requests.get(url)
    if r.status_code == 200:
        with open('netflix.csv', 'wb') as f:
            f.write(r.content)
        st.success("Downloaded dataset from GitHub!")
    else:
        st.error("Failed to download dataset.")

@st.cache_data
def load_data():
    if not os.path.exists('netflix.csv'):
        download_data()
    return pd.read_csv('netflix.csv')

# Sidebar: upload or use default
st.sidebar.header("📂 Data Source")
uploaded = st.sidebar.file_uploader("Upload your netflix1.csv", type=['csv'])
if uploaded:
    df = pd.read_csv(uploaded)
    st.sidebar.success("✅ Using uploaded file!")
else:
    df = load_data()
    st.sidebar.info("ℹ️ Using default GitHub file")

# --- Data Preview ---
st.header("📄 Dataset Preview")
st.dataframe(df.head(), use_container_width=True)

# --- Data Cleaning ---
st.header("🧹 Data Cleaning Steps")
before = df.shape[0]
df = df.drop_duplicates()
after = df.shape[0]
st.write(f"**Dropped {before-after} duplicate rows.**")

# Rename 'listed_in' column
if 'listed_in' in df.columns:
    df = df.rename(columns={'listed_in': 'genres'})

# Fill missing values
for col in ['director', 'cast', 'country', 'rating']:
    if col in df.columns:
        df[col] = df[col].fillna('Unknown')

# Date & Time Engineering
df['date_added'] = pd.to_datetime(df['date_added'], errors='coerce')
df['release_year'] = pd.to_numeric(df['release_year'], errors='coerce')
df['year_added'] = df['date_added'].dt.year
df['month_added'] = df['date_added'].dt.month

# Duration Splitting
def split_duration(x):
    if pd.isna(x['duration']):
        return pd.Series([np.nan, np.nan])
    val, unit = x['duration'].split()[0], x['duration'].split()[1]
    if x['type'] == 'Movie' and 'min' in unit:
        return pd.Series([int(val), np.nan])
    if x['type'] == 'TV Show' and 'season' in unit:
        return pd.Series([np.nan, int(val)])
    return pd.Series([np.nan, np.nan])

if 'duration' in df.columns:
    df[['movie_duration_mins', 'tvshow_duration_seasons']] = df.apply(split_duration, axis=1)
    df = df.drop(columns=['duration'])

st.success("✅ Data cleaned & features engineered successfully!")

# --- Sidebar Filters ---
st.sidebar.header("🎛️ Filters")
ctype = st.sidebar.multiselect('Select Type', options=df['type'].unique(), default=df['type'].unique())
yrange = (int(df['release_year'].min()), int(df['release_year'].max()))
year_slider = st.sidebar.slider('Release Year Range', *yrange, yrange)

df = df[df['type'].isin(ctype) & df['release_year'].between(year_slider[0], year_slider[1])]
st.sidebar.write(f"🔎 Filtered Rows: **{df.shape[0]}**")

# --- EDA Visualizations ---

# 1. General Overview
st.header("📊 1. General Overview")

# 1.1 Movies vs TV Shows — Pie Plot
st.subheader("1.1 Distribution: Movies vs TV Shows")
type_counts = df['type'].value_counts()
fig1, ax1 = plt.subplots(figsize=(5, 5))
ax1.pie(type_counts, labels=type_counts.index, autopct='%1.1f%%', startangle=90, colors=['#66b3ff', '#ff9999'])
ax1.set_title('Movies vs TV Shows')
st.pyplot(fig1)
st.write("**Conclusion:** Shows the proportion of Netflix content type distribution.")

# 1.2 Earliest & Latest Year of Content Release
st.subheader("1.2 Earliest vs Latest Year of Content Release")
earliest, latest = df['release_year'].min(), df['release_year'].max()
fig2, ax2 = plt.subplots(figsize=(5, 3))
ax2.bar(['Earliest', 'Latest'], [earliest, latest], color=['#90ee90', '#ffcccb'])
ax2.set_ylabel('Year')
ax2.set_title('Earliest vs Latest Year')
st.pyplot(fig2)
st.write("**Conclusion:** Shows the range of years Netflix content has been released.")

# 2. Country Insights
st.header("🌍 2. Country Insights")

# 2.1 Top 10 Countries — Bar Plot
st.subheader("2.1 Top 10 Countries by Number of Titles")
top_countries = df['country'].value_counts().head(10)
fig3, ax3 = plt.subplots(figsize=(6, 4))
sns.barplot(x=top_countries.values, y=top_countries.index, palette='Set2', ax=ax3)
ax3.set_title('Top 10 Countries')
st.pyplot(fig3)
st.write("**Conclusion:** Displays Netflix’s global footprint.")

# 2.2 Top 5 Countries - Movie vs TV Show
st.subheader("2.2 Top 5 Countries: Movies vs TV Shows")
top5 = df['country'].value_counts().head(5).index
df5 = df[df['country'].isin(top5)]
fig4, ax4 = plt.subplots(figsize=(6, 4))
sns.countplot(data=df5, y='country', hue='type', palette='pastel', ax=ax4)
ax4.set_title('Top 5 Countries Comparison')
st.pyplot(fig4)
st.write("**Conclusion:** Shows TV Shows vs Movies across top countries.")

# 3. Temporal Patterns
st.header("🕰️ 3. Temporal Patterns")

# 3.1 Netflix Additions Over Time
st.subheader("3.1 Titles Added per Year")
adds = df['year_added'].value_counts().sort_index()
fig5, ax5 = plt.subplots(figsize=(6, 4))
sns.lineplot(x=adds.index, y=adds.values, marker='o', color='purple', ax=ax5)
ax5.set_title('Titles Added Per Year')
st.pyplot(fig5)
st.write("**Conclusion:** Reveals Netflix content addition trends.")

# 3.2 Movies vs TV Shows Over Time
st.subheader("3.2 Movies vs TV Shows Over Time")
trend = df.groupby(['year_added', 'type']).size().unstack().fillna(0)
fig6, ax6 = plt.subplots(figsize=(6, 4))
trend.plot(marker='o', ax=ax6)
ax6.set_title('Trend Over Years')
ax6.grid(True)
st.pyplot(fig6)
st.write("**Conclusion:** Tracks focus shift between Movies and TV shows.")

# 4. Genre/Category Analysis
st.header("📚 4. Genre Analysis")

# 4.1 Top 10 Genres
st.subheader("4.1 Top 10 Genres")
genres = df['genres'].str.split(', ', expand=True).stack().value_counts().head(10)
fig7, ax7 = plt.subplots(figsize=(6, 4))
sns.barplot(x=genres.values, y=genres.index, palette='coolwarm', ax=ax7)
ax7.set_title('Top 10 Genres')
st.pyplot(fig7)
st.write("**Conclusion:** Lists most popular genres.")

# 4.2 Top Genres for Movies and TV Shows
st.subheader("4.2 Top Genres: Movies vs TV Shows")

# Movies
mgen = df[df['type'] == 'Movie'].assign(genre=df['genres'].str.split(', ')).explode('genre')['genre'].value_counts().head(5)
fig8, ax8 = plt.subplots(figsize=(4, 4))
ax8.pie(mgen, labels=mgen.index, autopct='%1.1f%%', startangle=90, colors=sns.color_palette('pastel'))
ax8.set_title('Top Movie Genres')
st.pyplot(fig8)

# TV Shows
tgen = df[df['type'] == 'TV Show'].assign(genre=df['genres'].str.split(', ')).explode('genre')['genre'].value_counts().head(5)
fig9, ax9 = plt.subplots(figsize=(4, 4))
ax9.pie(tgen, labels=tgen.index, autopct='%1.1f%%', startangle=90, colors=sns.color_palette('Set3'))
ax9.set_title('Top TV Show Genres')
st.pyplot(fig9)
st.write("**Conclusion:** Comparison of popular genres.")

# 5. Ratings
st.header("⭐ 5. Ratings")

# 5.1 Content Ratings Distribution
st.subheader("5.1 Ratings Distribution")
fig10, ax10 = plt.subplots(figsize=(6, 4))
sns.countplot(data=df, x='rating', order=df['rating'].value_counts().index, palette='Set3', ax=ax10)
ax10.set_xticklabels(ax10.get_xticklabels(), rotation=45)
ax10.set_title('Ratings Distribution')
st.pyplot(fig10)
st.write("**Conclusion:** Overview of content ratings.")

# 5.2 Top 6 Ratings Pie
st.subheader("5.2 Top 6 Ratings Pie Chart")
top6 = df['rating'].value_counts().head(6)
fig11, ax11 = plt.subplots(figsize=(4, 4))
ax11.pie(top6, labels=top6.index, autopct='%1.1f%%', startangle=90, colors=sns.color_palette('Set2'))
ax11.set_title('Top 6 Ratings')
st.pyplot(fig11)

# 5.3 Ratings Trend Over Time
st.subheader("5.3 Ratings Over Years")
ry = df.groupby(['year_added', 'rating']).size().unstack().fillna(0)
fig12, ax12 = plt.subplots(figsize=(6, 4))
ry.plot(ax=ax12)
ax12.set_title('Ratings Over Time')
st.pyplot(fig12)

# 6. Duration
st.header("⏳ 6. Duration Distribution")

# 6.1 Movie Duration
st.subheader("6.1 Movie Duration Distribution")
fig13, ax13 = plt.subplots(figsize=(6, 4))
sns.histplot(df['movie_duration_mins'], bins=20, kde=True, color='orange', ax=ax13)
ax13.set_title('Movie Durations')
st.pyplot(fig13)



# 7. Director Insights
st.header("🎬 7. Director Insights")

# 7.1 Top 10 Directors
st.subheader("7.1 Top 10 Directors")
dirs = df[~df['director'].isin(['', 'Not Given'])].dropna(subset=['director'])['director'].value_counts().head(10)
fig15, ax15 = plt.subplots(figsize=(6, 4))
sns.barplot(x=dirs.values, y=dirs.index, palette='mako', ax=ax15)
ax15.set_title('Top Directors')
st.pyplot(fig15)

# 7.2 Top TV Show Directors
st.subheader("7.2 Top TV Show Directors")
tv_dirs = df[df['type'] == 'TV Show'][~df['director'].isin(['', 'Not Given'])].dropna(subset=['director'])['director'].value_counts().head(10)
fig16, ax16 = plt.subplots(figsize=(6, 4))
sns.barplot(x=tv_dirs.values, y=tv_dirs.index, palette='viridis', ax=ax16)
ax16.set_title('Top TV Directors')
st.pyplot(fig16)

# --- Footer ---
st.markdown('---')
st.caption('Built with ❤️ by Kaustav Roy Chowdhury')

