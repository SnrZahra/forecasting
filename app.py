import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error
import plotly.graph_objects as go
import warnings
from PIL import Image
import io

# Mengabaikan warning dari model, tidak akan mempengaruhi hasil
warnings.filterwarnings("ignore")

# --- KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="Dashboard Forecasting BAZNAS",
    page_icon="images.jpeg",
    layout="wide"
)

# --- FUNGSI UNTUK CSS KUSTOM ---
def load_custom_css():
    """Menyuntikkan CSS untuk mengubah tampilan sesuai tema."""
    css = """
    <style>
        /* Latar belakang utama menjadi abu-abu */
        [data-testid="stAppViewContainer"] {
            background-color: #f0f2f6;
        }

        /* [PERUBAHAN] Warna teks utama menjadi hitam */
        .main .block-container {
            color: #000000;
        }
        h1, h2, h3, h4, h5, h6 {
            color: #000000 !important;
        }

        /* Pengaturan Sidebar */
        [data-testid="stSidebar"] {
            background-color: #00502D;
        }
        [data-testid="stSidebar"] p, [data-testid="stSidebar"] label, [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3, .stFileUploader label {
            color: #FFFFFF !important;
        }
        
        /* Tombol Utama */
        .stButton button {
            background-color: #000000;
            color: #FFFFFF;
            border-radius: 8px;
            border: 2px solid #000000;
            transition: 0.3s ease;
        }
        .stButton button:hover {
            background-color: #FFA500;
            color: #000000;
            border: 2px solid #FFA500;
        }
        
        /* Container evaluasi */
        .eval-container {
            background-color: rgba(255, 255, 255, 0.5);
            border-radius: 8px;
            padding: 15px;
            margin-top: 20px;
            height: 100%;
        }
        .eval-container table {
            width: 100%;
            color: black;
        }
        .eval-container td:first-child { width: 70%; }
        .eval-container td:last-child { font-weight: bold; text-align: right; }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# Panggil CSS
load_custom_css()

# --- FUNGSI-FUNGSI BANTU ---
@st.cache_data
def load_data(uploaded_file):
    """
    Memuat dan memproses data dari file yang di-upload.
    Akan mengurutkan data secara otomatis.
    """
    if uploaded_file is None:
        return None
    try:
        data = pd.read_excel(uploaded_file)
        
        if 'Bulan Tahun' not in data.columns:
            st.error("Kolom 'Bulan Tahun' tidak ditemukan. Mohon periksa file Anda.")
            return None

        data['Bulan Tahun'] = pd.to_datetime(data['Bulan Tahun'], errors='coerce')
        data.dropna(subset=['Bulan Tahun'], inplace=True)
        
        data = data.set_index('Bulan Tahun').sort_index()
        
        return data[['Modal Usaha', 'Rombong']].resample('MS').sum()
    except Exception as e:
        st.error(f"Gagal memuat data: {e}. Pastikan format file dan nama kolom sudah benar.")
        return None

def get_best_model(series_data, target_column):
    """Memilih dan membangun model terbaik berdasarkan jenis target."""
    if target_column == 'Rombong':
        model_name = "ARIMA (1,0,1)"
        model = ARIMA(series_data, order=(1, 0, 1))
    elif target_column == 'Modal Usaha':
        model_name = "SARIMA (1,0,1)(0,1,1,12)"
        model = SARIMAX(series_data, order=(1, 0, 1), seasonal_order=(0, 1, 1, 12), enforce_stationarity=False, enforce_invertibility=False)
    return model, model_name

def get_evaluation_metrics(data, target_column):
    """Menghitung metrik dengan train-test split yang sesuai."""
    if len(data) < 24:
        return 0, 0, 0
        
    split_point = int(len(data) * 0.9) if target_column == 'Modal Usaha' else int(len(data) * 0.8)
    train_data, test_data = data.iloc[:split_point], data.iloc[split_point:]
    y_true = test_data[target_column]
    
    model_for_eval, _ = get_best_model(train_data[target_column], target_column)
    fit_model = model_for_eval.fit()
    
    y_pred = fit_model.get_forecast(steps=len(test_data)).predicted_mean
    y_pred.index = y_true.index
    
    mae, rmse = mean_absolute_error(y_true, y_pred), np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true.replace(0, np.nan))) * 100
    
    return mae, rmse, mape

def train_and_predict_future(data, target_column, n_periods):
    """Melatih model pada SEMUA data untuk prediksi masa depan."""
    model_for_future, model_name = get_best_model(data[target_column], target_column)
    fit_model = model_for_future.fit()

    forecast = fit_model.get_forecast(steps=n_periods)
    pred_index = pd.date_range(start=data.index[-1] + pd.DateOffset(months=1), periods=n_periods, freq='MS')
    pred_df = pd.DataFrame(forecast.predicted_mean.values, index=pred_index, columns=['Prediksi'])
    conf_int = forecast.conf_int()
    pred_df['Lower CI'], pred_df['Upper CI'] = conf_int.iloc[:, 0].values, conf_int.iloc[:, 1].values
    return pred_df, model_name

# --- LAYOUT APLIKASI ---

# SIDEBAR
with st.sidebar:
    try:
        logo = Image.open('images.jpeg')
        st.image(logo)
    except FileNotFoundError:
        st.warning("File logo 'images.jpeg' tidak ditemukan.")

    st.title("Panel Kontrol")
    
    uploaded_file = st.file_uploader("Upload Dataset Anda", type=['xlsx', 'xls'])
    st.markdown("---")

    data_bulanan = load_data(uploaded_file) 

    if data_bulanan is not None:
        target_display_name = st.selectbox("Pilih Target Prediksi:", ("Bantuan Rombong", "Modal Usaha"))
        target_column_name = "Rombong" if target_display_name == "Bantuan Rombong" else "Modal Usaha"
        
        st.markdown("---")
        st.write("**Model Terbaik yang Digunakan:**")
        st.info("ARIMA (1,0,1)" if target_column_name == "Rombong" else "SARIMA (1,0,1)(0,1,1,12)")
        st.markdown("---")
        
        min_date, max_date = data_bulanan.index.min().date(), data_bulanan.index.max().date()

        st.write("**Filter Rentang Waktu:**")
        start_date = st.date_input("Tanggal Mulai", min_value=min_date, max_value=max_date, value=min_date)
        end_date = st.date_input("Tanggal Akhir", min_value=min_date, max_value=max_date, value=max_date)

        st.markdown("---")
        periods_input = st.number_input("Periode Prediksi (Bulan):", 1, 48, 12, 1)
        predict_button = st.button(label="BUAT PREDIKSI", use_container_width=True)

# HALAMAN UTAMA
if data_bulanan is not None:
    st.markdown(f'<h1>📈 Dashboard Forecasting: {target_display_name}</h1>', unsafe_allow_html=True)
    st.markdown("---")

    filtered_data = data_bulanan.loc[start_date:end_date]
    if filtered_data.empty:
        st.error("Rentang data yang Anda pilih tidak valid atau tidak mengandung data.")
    else:
        st.markdown(f'<h3>Grafik Data Aktual (Rentang: {start_date.strftime("%b %Y")} - {end_date.strftime("%b %Y")})</h3>', unsafe_allow_html=True)
        fig_actual = go.Figure()
        fig_actual.add_trace(go.Scatter(x=filtered_data.index, y=filtered_data[target_column_name], mode='lines+markers', name='Data Historis (Dipilih)', line=dict(color='#00502D', width=3)))
        
        # Mengatur warna font pada grafik secara eksplisit
        fig_actual.update_layout(
            plot_bgcolor='white', paper_bgcolor='white',
            font=dict(color='black'), # Aturan umum untuk semua teks
            xaxis=dict(title='Tanggal', title_font=dict(color='black'), tickfont=dict(color='black'), gridcolor='#D7D7D7'),
            yaxis=dict(title='Jumlah (Rupiah)', title_font=dict(color='black'), tickfont=dict(color='black'), gridcolor='#D7D7D7', tickprefix='Rp ', tickformat=',.0f'),
            legend=dict(font=dict(color='black'), bgcolor='rgba(255,255,255,0.6)')
        )
        st.plotly_chart(fig_actual, use_container_width=True)

        if 'predict_button' in locals() and predict_button:
            with st.spinner(f"Membuat prediksi dan evaluasi untuk {target_display_name}..."):
                mae, rmse, mape = get_evaluation_metrics(filtered_data, target_column_name)
                prediction_df, model_name = train_and_predict_future(filtered_data, target_column_name, periods_input)
            
            st.markdown("---")
            st.markdown(f'<h3>Hasil Prediksi Menggunakan {model_name}</h3>', unsafe_allow_html=True)
            
            fig_pred = go.Figure()
            fig_pred.add_trace(go.Scatter(x=filtered_data.index, y=filtered_data[target_column_name], name='Data Historis (Dipilih)', line=dict(color='#00502D')))
            fig_pred.add_trace(go.Scatter(x=prediction_df.index, y=prediction_df['Prediksi'], name='Hasil Prediksi', line=dict(color='#FF8C00', dash='dash')))
            fig_pred.add_trace(go.Scatter(x=prediction_df.index, y=prediction_df['Lower CI'], fill=None, mode='lines', line_color='rgba(255,140,0,0.3)', showlegend=False))
            fig_pred.add_trace(go.Scatter(x=prediction_df.index, y=prediction_df['Upper CI'], fill='tonexty', mode='lines', line_color='rgba(255,140,0,0.3)', name='Interval Kepercayaan'))
            
            # [PERBAIKAN UTAMA] Mengatur posisi judul dan legenda
            fig_pred.update_layout(
                plot_bgcolor='white', paper_bgcolor='white',
                font=dict(color='black'), # Aturan umum untuk semua teks
                title=dict(
                    text=f"Prediksi {target_display_name} untuk {periods_input} Bulan ke Depan", 
                    font=dict(color='black'),
                    x=0.5, # Posisikan judul di tengah
                    xanchor='center'
                ),
                xaxis=dict(title='Tanggal', title_font=dict(color='black'), tickfont=dict(color='black'), gridcolor='#D7D7D7'),
                yaxis=dict(title='Jumlah (Rupiah)', title_font=dict(color='black'), tickfont=dict(color='black'), gridcolor='#D7D7D7', tickprefix='Rp ', tickformat=',.0f'),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02, # Posisikan legenda di atas chart
                    xanchor="right",
                    x=1, # Posisikan legenda di kanan
                    font=dict(color='black'),
                    bgcolor='rgba(255,255,255,0.6)'
                )
            )
            st.plotly_chart(fig_pred, use_container_width=True)

            col1, col2 = st.columns(2)
            with col1:
                st.markdown('<h4>🔢 Tabel Nilai Prediksi</h4>', unsafe_allow_html=True)
                display_df = prediction_df[['Prediksi']].copy()
                display_df.index = display_df.index.strftime('%B %Y')
                display_df.index.name = "Bulan Prediksi"
                display_df['Prediksi'] = display_df['Prediksi'].apply(lambda x: f"Rp {x:,.0f}".replace(',', '.'))
                st.dataframe(display_df, use_container_width=True, height=400)

            with col2:
                st.markdown('<h4>📝 Evaluasi Model (berdasarkan data yang difilter)</h4>', unsafe_allow_html=True)
                mae_val, rmse_val, mape_val = f"{mae:,.0f}".replace(',', '.'), f"{rmse:,.0f}".replace(',', '.'), f"{mape:.2f}%"
                
                eval_html = f"""
                <div class="eval-container">
                    <table>
                        <tr><td>Nilai Rata-rata kesalahan absolut dalam prediksi (MAE)</td><td>: {mae_val}</td></tr>
                        <tr><td>Nilai Rata-rata dari selisih antara nilai prediksi dan nilai aktual (RMSE)</td><td>: {rmse_val}</td></tr>
                        <tr><td>Presentase Nilai Rata-rata dari selisih antara nilai prediksi dan nilai aktual (MAPE)</td><td>: {mape_val}</td></tr>
                    </table>
                </div>
                """
                st.markdown(eval_html, unsafe_allow_html=True)
else:
    # Pesan default jika belum ada file yang diupload
    st.markdown(
        """
        <div style="background-color: #FFFFFF; border-radius: 10px; padding: 40px; text-align: center;">
            <h2 style="color: black;">👋 Selamat Datang!</h2>
            <p style="color: black; font-size: 1.2em;">
                Silakan upload file Excel pada sidebar untuk memulai analisis.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )