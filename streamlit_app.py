import streamlit as st
import datetime
from google_play_scraper import Sort, reviews_all
st.title("🎈 test analisis")
st.write(
    "---------------------------"
)
st.header(
    "Data Gathering"
)
#Memilih Apps
SelectMbanking = st.selectbox(
    "Pilih Mbanking yang ingin di analisis :" , ("BYOND", "BSI Mobile")
)
if SelectMbanking == "BYOND":
    app_id = "co.id.bankbsi.superapp"
   
elif SelectMbanking == "BSI Mobile":
    app_id ="com.bsm.activity2"

#Memilih Review 
SelectReview = st.selectbox(
    "Pilih Review Berdasarkan :" , ("Most Relevant", "Newest")
)
if SelectReview == "Most Relevant":
    sort_review = Sort.MOST_RELEVANT
   
elif SelectReview == "Newest":
    sort_review = Sort.NEWEST

#filter tanggal
col1,col2 = st.columns(2)
with col1:
    datedari = st.date_input("Mulai Dari : ",format= "DD/MM/YYYY" )
with col2:
    datesampai = st.date_input("Sampai Dengan : ",format= "DD/MM/YYYY" )
#proses data gathering
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

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
            sort = sort_review
        )

        # Konversi ke DataFrame
        df = pd.DataFrame(np.array(result), columns=['review'])
        df = df.join(pd.DataFrame(df.pop('review').tolist()))

        # Konversi kolom 
        df['userName'] = df['userName'].astype('string')
        df['at'] = pd.to_datetime(df['at'])
        df['content'] = df['content'].astype('string')

        # Filter berdasarkan tanggal
        filtered_reviews = df[(df['at'] >= pd.Timestamp(datedari)) & (df['at'] <= pd.Timestamp(datesampai))]
        df_hasil=filtered_reviews[['userName','score','at','content']]
        
        # Simpan DataFrame ke session_state
        st.session_state.df = df_hasil

        # Menampilkan hasil
        st.subheader("Review yang Difilter")
        st.dataframe(st.session_state.df)

        # Menampilkan jumlah review yang diambil
        st.success(f"Berhasil mengambil {len(st.session_state.df)} review.")
st.write(
    "---------------------------"
)
#preprocessing data
if st.session_state.df is not None:
    st.header("Preprocessing Data")
    st.subheader("Cleaning Data")
    if st.button("Lakukan Data Cleaning"):
        with st.spinner("Sedang membersihkan data..."):
            # Cek Tipe Data
            st.subheader("Tipe Data Kolom")
            st.write(st.session_state.df.dtypes)

            # Cek dan Hapus Duplikasi
            duplicate_count = st.session_state.df.duplicated().sum()
            if duplicate_count > 0:
                st.warning(f"Terdapat {duplicate_count} duplikasi. Menghapus duplikasi...")
                st.session_state.df = st.session_state.df.drop_duplicates()
                st.success("Duplikasi berhasil dihapus!")
            else:
                st.info("Tidak ada duplikasi dalam dataset.")

            # Cek dan Ganti NaN dengan 0
            nan_count = st.session_state.df.isna().sum().sum()  # Hitung total NaN di semua kolom
            if nan_count > 0:
                st.warning(f"Terdapat {nan_count} nilai NaN. Mengganti dengan 0...")
                st.session_state.df = st.session_state.df.fillna(0)
                st.success("NaN berhasil diganti dengan 0!")
            else:
                st.info("Tidak ada nilai NaN dalam dataset.")

            # Menampilkan DataFrame yang sudah dibersihkan
            st.subheader("Data Setelah Cleaning")
            st.dataframe(st.session_state.df)
            st.success(f"{len(st.session_state.df)} Data.")

# **Transformasi Data**
import Sastrawi
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory, StopWordRemover, ArrayDictionary
more_stop_words = []
if st.session_state.df is not None:
    st.subheader("Transformasi Data")

    if st.button("Lakukan Transformasi Data"):
        with st.spinner("Melakukan preprocessing..."):

            # 1. Filtering jumlah kata dalam content
            def filter_tokens_by_length(dataframe, column, min_words, max_words):
                word_count = dataframe[column].astype(str).apply(lambda x: len(x.split()))
                mask = (word_count >= min_words) & (word_count <= max_words)
                return dataframe[mask]

            min_words = 3
            max_words = 50
            st.session_state.df = filter_tokens_by_length(st.session_state.df, 'content', min_words, max_words)
            st.success("Filtering berdasarkan panjang kata selesai!")

            # 2. Normalisasi
            norm = {
                "bgt": "banget", "brp": "berapa", "blm": "belum", "lbh": "lebih", "tp": "tapi", "ngga": "tidak", 
                "nggak": "tidak", "gak": "tidak", "dpt": "dapat", "lg": "lagi", "krn": "karena", "jgn": "jangan",
                "nabung": "menabung", "tabungan": "tabungan", "bunga": "bunga", "cicilan": "cicilan", "utang": "Hutang",
                "pinjem": "Pinjam", "byr": "bayar", "login": "login", "eror": "error", "otp": "OTP", "notif": "notifikasi",
                "pitur": "fitur", "kmn": "mana", "bisa2": "bisa", "uwang": "uang", "kyk": "seperti", "km": "kamu",
                "byk": "banyak", "mrk": "mereka", "profisional": "profesional", "mulu": "terus", "payah": "jelek",
                "identivikasi": "identifikasi", "apk": "aplikasi"
            }

            def normalisasi(text):
                for word, replacement in norm.items():
                    text = text.replace(word, replacement)
                return text

            st.session_state.df['content'] = st.session_state.df['content'].apply(normalisasi)
            st.success("Normalisasi selesai!")

            # 3. Stopword Removal
            stop_words = StopWordRemoverFactory().get_stop_words()
            stopword_remover = StopWordRemover(ArrayDictionary(stop_words))

            def remove_stopwords(text):
                return stopword_remover.remove(text)

            st.session_state.df['content'] = st.session_state.df['content'].apply(remove_stopwords)
            st.success("Stopword removal selesai!")

            # 4. Tokenization
            st.session_state.df['content'] = st.session_state.df['content'].apply(lambda x: x.split())
            st.success("Tokenisasi selesai!")

            # 5. Stemming
            stemmer = StemmerFactory().create_stemmer()

            def stemming(text_list):
                return " ".join([stemmer.stem(word) for word in text_list])

            st.session_state.df['content'] = st.session_state.df['content'].apply(stemming)
            st.success("Stemming selesai!")

            st.subheader("Data Setelah Preprocessing")
            st.dataframe(st.session_state.df)
