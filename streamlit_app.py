import google_play_scraper
import streamlit as st
import datetime
from google_play_scraper import Sort, reviews_all
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory, ArrayDictionary
from Sastrawi.StopWordRemover.StopWordRemover import StopWordRemover
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import hashlib
from wordcloud import WordCloud
from collections import Counter

st.title("🎈 test analisis")
st.write("---------------------------")
st.header("Data Gathering")

# Memilih Apps
SelectMbanking = st.selectbox("Pilih Mbanking yang ingin di analisis :" , ("BYOND", "BSI Mobile"))
if SelectMbanking == "BYOND":
    app_id = "co.id.bankbsi.superapp"
elif SelectMbanking == "BSI Mobile":
    app_id ="com.bsm.activity2"

# Memilih Review 
SelectReview = st.selectbox("Pilih Review Berdasarkan :" , ("Most Relevant", "Newest"))
if SelectReview == "Most Relevant":
    sort_review = Sort.MOST_RELEVANT
elif SelectReview == "Newest":
    sort_review = Sort.NEWEST

# Filter tanggal
col1, col2 = st.columns(2)
with col1:
    datedari = st.date_input("Mulai Dari : ", format= "DD/MM/YYYY")
with col2:
    datesampai = st.date_input("Sampai Dengan : ", format= "DD/MM/YYYY")

if "df" not in st.session_state:
    st.session_state.df = None

# Tombol untuk Mengambil Review
if st.button("Ambil Review"):
    with st.spinner("Mengambil data..."):
        result = reviews_all(
            app_id,
            sleep_milliseconds=0, 
            lang='id', 
            country='id', 
            sort=sort_review
        )

        df = pd.DataFrame(np.array(result), columns=['review'])
        df = df.join(pd.DataFrame(df.pop('review').tolist()))

        df['userName'] = df['userName'].astype('string')
        df['at'] = pd.to_datetime(df['at'])
        df['content'] = df['content'].astype('string')

        filtered_reviews = df[(df['at'] >= pd.Timestamp(datedari)) & (df['at'] <= pd.Timestamp(datesampai))]
        df_hasil = filtered_reviews[['userName', 'score', 'at', 'content']]

        # Label Sentimen
        def label_sentiment(score):
            if score <= 2:
                return 'negatif'
            elif score == 3:
                return 'netral'
            else:
                return 'positif'

        df_hasil['sentimen'] = df_hasil['score'].apply(label_sentiment)

        st.session_state.df = df_hasil
        st.subheader("Review yang Difilter")
        st.dataframe(st.session_state.df)
        st.success(f"Berhasil mengambil {len(st.session_state.df)} review.")

st.write("---------------------------")

if st.session_state.df is not None:
    st.header("Preprocessing Data")
    st.subheader("Cleaning Data")
    if st.button("Lakukan Data Cleaning"):
        with st.spinner("Sedang membersihkan data..."):
            st.subheader("Tipe Data Kolom")
            st.write(st.session_state.df.dtypes)

            duplicate_count = st.session_state.df.duplicated().sum()
            if duplicate_count > 0:
                st.warning(f"Terdapat {duplicate_count} duplikasi. Menghapus duplikasi...")
                st.session_state.df = st.session_state.df.drop_duplicates()
                st.success("Duplikasi berhasil dihapus!")
            else:
                st.info("Tidak ada duplikasi dalam dataset.")

            nan_count = st.session_state.df.isna().sum().sum()
            if nan_count > 0:
                st.warning(f"Terdapat {nan_count} nilai NaN. Mengganti dengan 0...")
                st.session_state.df = st.session_state.df.fillna(0)
                st.success("NaN berhasil diganti dengan 0!")
            else:
                st.info("Tidak ada nilai NaN dalam dataset.")

            st.subheader("Data Setelah Cleaning")
            st.dataframe(st.session_state.df)
            st.success(f"{len(st.session_state.df)} Data.")

@st.cache_resource
def load_text_processors():
    stemmer = StemmerFactory().create_stemmer()
    stop_words = StopWordRemoverFactory().get_stop_words()
    stopword_dict = ArrayDictionary(stop_words)
    stopword_remover = StopWordRemover(stopword_dict)
    return stemmer, stopword_remover

def get_vectorized_data(df_content, df_label):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df_content)
    y = df_label
    return train_test_split(X, y, test_size=0.2, random_state=42), vectorizer

def preprocess_text(text, norm_dict, stemmer, stopword_remover):
    text = text.lower()
    for word, replacement in norm_dict.items():
        text = text.replace(word, replacement)
    text = stopword_remover.remove(text)
    text = " ".join([stemmer.stem(word) for word in text.split()])
    return text

if st.session_state.df is not None:
    st.subheader("Transformasi Data")

    if st.button("Lakukan Transformasi Data"):
        with st.spinner("Melakukan preprocessing..."):

            def filter_tokens_by_length(dataframe, column, min_words, max_words):
                word_count = dataframe[column].astype(str).apply(lambda x: len(x.split()))
                mask = (word_count >= min_words) & (word_count <= max_words)
                return dataframe[mask]

            min_words = 3
            max_words = 50
            st.session_state.df = filter_tokens_by_length(st.session_state.df, 'content', min_words, max_words)
            st.success("Filtering berdasarkan panjang kata selesai!")

            norm = {
                "bgt": "banget", "brp": "berapa", "blm": "belum", "lbh": "lebih", "tp": "tapi", "ngga": "tidak", 
                "nggak": "tidak", "gak": "tidak", "dpt": "dapat", "lg": "lagi", "krn": "karena", "jgn": "jangan",
                "nabung": "menabung", "tabungan": "tabungan", "bunga": "bunga", "cicilan": "cicilan", "utang": "Hutang",
                "pinjem": "Pinjam", "byr": "bayar", "login": "login", "eror": "error", "otp": "OTP", "notif": "notifikasi",
                "pitur": "fitur", "kmn": "mana", "bisa2": "bisa", "uwang": "uang", "kyk": "seperti", "km": "kamu",
                "byk": "banyak", "mrk": "mereka", "profisional": "profesional", "mulu": "terus", "payah": "jelek",
                "identivikasi": "identifikasi", "apk": "aplikasi"
            }

            stemmer, stopword_remover = load_text_processors()
            progress = st.progress(0)

            processed_texts = []
            for i, text in enumerate(st.session_state.df['content']):
                processed = preprocess_text(str(text), norm, stemmer, stopword_remover)
                processed_texts.append(processed)
                progress.progress((i + 1) / len(st.session_state.df))

            st.session_state.df['content'] = processed_texts
            st.success("Semua proses preprocessing selesai!")

            st.subheader("Data Setelah Preprocessing")
            st.dataframe(st.session_state.df)

    if st.button("Jalankan Model dan Visualisasi"):
        st.subheader("Model Fit & Visualisasi Sentimen")

        df = st.session_state.df
        if 'content' in df.columns and 'sentimen' in df.columns:
            st.subheader("Data yang Siap Dilatih")
            st.dataframe(df[['content', 'sentimen']].head())

            (X_train, X_test, y_train, y_test), vectorizer = get_vectorized_data(df['content'], df['sentimen'])

            classifiers = {
                "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
                "Logistic Regression": LogisticRegression(max_iter=200),
                "Naive Bayes": MultinomialNB()
            }

            result = []
            y_preds = {}

            for clf_name, clf in classifiers.items():
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)
                y_preds[clf_name] = y_pred

                accuracy = accuracy_score(y_test, y_pred)
                recall = recall_score(y_test, y_pred, average='weighted')
                precision = precision_score(y_test, y_pred, average='weighted')
                f1 = f1_score(y_test, y_pred, average='weighted')

                result.append({
                    "Classifier": clf_name,
                    "Accuracy": accuracy,
                    "Recall": recall,
                    "Precision": precision,
                    "F1 Score": f1
                })

            result_df = pd.DataFrame(result)
            st.dataframe(result_df)

            for clf_name in classifiers.keys():
                st.subheader(f"Confusion Matrix - {clf_name}")
                cm = confusion_matrix(y_test, y_preds[clf_name], labels=["negatif", "netral", "positif"])
                fig, ax = plt.subplots()
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["negatif", "netral", "positif"],
                            yticklabels=["negatif", "netral", "positif"], ax=ax)
                plt.ylabel('Actual')
                plt.xlabel('Predicted')
                st.pyplot(fig)

            # WordCloud Semua
            st.subheader("Word Cloud dari Semua Review")
            all_text = " ".join(df['content'].tolist())
            wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_text)
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis('off')
            st.pyplot(fig)

            # WordCloud Positif & Negatif
            st.subheader("Word Cloud Berdasarkan Sentimen")
            for label, color in zip(["positif", "negatif"], ["Greens", "Reds"]):
                text = " ".join(df[df['sentimen'] == label]['content'])
                wc = WordCloud(width=800, height=400, background_color='white', colormap=color).generate(text)
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.imshow(wc, interpolation='bilinear')
                ax.axis('off')
                st.pyplot(fig)

            # Top 10 Kata
            st.subheader("Top 10 Kata Paling Sering Muncul")
            positif_words = " ".join(df[df['sentimen'] == 'positif']['content']).split()
            negatif_words = " ".join(df[df['sentimen'] == 'negatif']['content']).split()

            pos_counter = Counter(positif_words).most_common(10)
            neg_counter = Counter(negatif_words).most_common(10)

            st.markdown("#### Top 10 Kata Positif")
            words, freqs = zip(*pos_counter)
            fig1, ax1 = plt.subplots()
            ax1.barh(words[::-1], freqs[::-1], color='green')
            ax1.set_title("Top 10 Kata Positif")
            st.pyplot(fig1)

            st.markdown("#### Top 10 Kata Negatif")
            words, freqs = zip(*neg_counter)
            fig2, ax2 = plt.subplots()
            ax2.barh(words[::-1], freqs[::-1], color='red')
            ax2.set_title("Top 10 Kata Negatif")
            st.pyplot(fig2)

            # Distribusi Sentimen
            st.subheader("Distribusi Sentimen Review")
            sentiment_counts = df['sentimen'].value_counts()
            fig3, ax3 = plt.subplots()
            ax3.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', colors=["red", "orange", "green"])
            ax3.set_title("Persentase Sentimen")
            st.pyplot(fig3)

            fig4, ax4 = plt.subplots()
            ax4.bar(sentiment_counts.index, sentiment_counts.values, color=["red", "green", "orange"])
            ax4.set_ylabel("Jumlah Review")
            ax4.set_title("Distribusi Sentimen")
            st.pyplot(fig4)

        else:
            st.warning("Data belum memiliki kolom 'content' dan 'sentimen'. Silakan preprocessing terlebih dahulu.")
else:
    st.warning("Data belum tersedia. Silakan lakukan data preprocessing terlebih dahulu.")
