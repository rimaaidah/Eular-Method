import streamlit
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# Konfigurasi halaman
# ---------------------------------------------------------
streamlit.set_page_config(page_title="Euler COVID Jabar", layout="wide")


# ---------------------------------------------------------
# 1. Fungsi bantu: load & agregasi data
# ---------------------------------------------------------
@streamlit.cache_data
def load_data(csv_path: str) -> pd.DataFrame:
    """
    Membaca file CSV dan mengagregasi data menjadi
    total kasus aktif (konfirmasi_aktif) per tanggal untuk seluruh Jawa Barat.
    """
    df = pd.read_csv(csv_path)
    df["tanggal"] = pd.to_datetime(df["tanggal"])

    # Group by tanggal dan jumlahkan kasus aktif seluruh kab/kota
    df_group = (
        df.groupby("tanggal", as_index=False)["konfirmasi_aktif"]
          .sum()
          .sort_values("tanggal")
          .reset_index(drop=True)
    )

    # Buang nilai NaN jika ada
    df_group = df_group[df_group["konfirmasi_aktif"].notna()].copy()
    df_group = df_group.reset_index(drop=True)

    return df_group


def f_exp(t: float, I: float, params: dict) -> float:
    """
    Model eksponensial: dI/dt = r * I

    Parameters
    ----------
    t : float
        Waktu (hari). Tidak dipakai eksplisit di model ini, tapi tetap disertakan.
    I : float
        Nilai kasus aktif pada waktu t.
    params : dict
        Kamus parameter yang minimal berisi 'r'.

    Returns
    -------
    float
        Nilai turunan dI/dt.
    """
    r = params["r"]
    return r * I


def euler_solve(f, t0, y0, h, n_steps, params):
    """
    Menyelesaikan ODE 1 dimensi dengan Metode Euler.

    f       : fungsi ODE f(t, y, params)
    t0      : waktu awal
    y0      : nilai awal
    h       : step size (langkah waktu)
    n_steps : jumlah langkah Euler
    params  : parameter model (dict)
    """
    t_values = np.zeros(n_steps + 1)
    y_values = np.zeros(n_steps + 1)

    t_values[0] = t0
    y_values[0] = y0

    for n in range(n_steps):
        # Rumus Euler: y_{n+1} = y_n + h * f(t_n, y_n)
        y_values[n + 1] = y_values[n] + h * f(t_values[n], y_values[n], params)
        t_values[n + 1] = t_values[n] + h

    return t_values, y_values


# ---------------------------------------------------------
# 2. MAIN APP
# ---------------------------------------------------------

streamlit.title("Metode Euler untuk Kasus Aktif COVID-19 Jawa Barat")

streamlit.write(
    """
Aplikasi ini merupakan implementasi **Metode Euler** untuk memodelkan
**kasus aktif COVID-19 di Jawa Barat** menggunakan data dari Open Data Jabar
(*perkembangan harian kasus terkonfirmasi positif COVID-19 berdasarkan kabupaten/kota*).

Data diolah menjadi **total kasus aktif per hari untuk seluruh Jawa Barat**, lalu
dimodelkan dengan persamaan diferensial sederhana:

\\[
\\frac{dI}{dt} = r I
\\]

dengan:
- \\( I(t) \\): total kasus aktif pada hari ke-\\( t \\),
- \\( r \\): laju pertumbuhan kasus aktif per hari.
"""
)

# 2.1 Load data
csv_path = "covid_jabar_perkembangan_harian.csv"
df_group = load_data(csv_path)

streamlit.subheader("Cuplikan Data COVID-19 (Total Kasus Aktif per Tanggal)")
streamlit.dataframe(df_group.head())

# ---------------------------------------------------------
# 3. Sidebar: parameter input (sesuai soal TA)
# ---------------------------------------------------------

streamlit.sidebar.header("Pengaturan Simulasi (Metode Euler)")

# Ambil deret nilai kasus aktif
I_data = df_group["konfirmasi_aktif"].values.astype(float)
tanggal_all = df_group["tanggal"].values

# Estimasi r awal secara sederhana dari rata-rata log-growth
mask_pos = I_data > 0
r_default = 0.1  # fallback

if mask_pos.sum() >= 2:
    I_pos = I_data[mask_pos]
    growth_logs = np.log(I_pos[1:] / I_pos[:-1])
    r_default = float(growth_logs.mean())

r = streamlit.sidebar.number_input(
    "Laju pertumbuhan r (per hari)",
    value=round(r_default, 4),
    step=0.01,
    format="%.4f",
    help="Semakin besar r, semakin cepat pertumbuhan kasus aktif (model eksponensial)."
)

h = streamlit.sidebar.slider(
    "Step size h (hari per langkah Euler)",
    min_value=0.1,
    max_value=2.0,
    value=1.0,
    step=0.1,
    help="Semakin kecil h, simulasi lebih halus tetapi langkah yang dibutuhkan lebih banyak."
)

max_days = min(200, len(df_group))
duration_days = streamlit.sidebar.slider(
    "Jumlah hari yang disimulasikan",
    min_value=10,
    max_value=max_days,
    value=min(60, max_days),
    step=1,
    help="Durasi total simulasi yang akan dibandingkan dengan data."
)

# ---------------------------------------------------------
# 4. Menjalankan simulasi Euler
# ---------------------------------------------------------

# Siapkan data untuk simulasi
I_data_used = I_data.copy()
tanggal_used = tanggal_all

# Hitung jumlah langkah Euler dari durasi dan h
n_steps = int(duration_days / h)
if n_steps < 1:
    streamlit.error("Kombinasi durasi dan h membuat jumlah langkah terlalu sedikit.")
    streamlit.stop()

# Jangan melebihi panjang data
n_steps = min(n_steps, len(I_data_used) - 1)

# Kondisi awal model diambil dari data hari pertama
I0 = float(I_data_used[0])
params = {"r": r}
t0 = 0.0

t_sim, I_sim = euler_solve(f_exp, t0, I0, h, n_steps, params)

# Sesuaikan panjang data & tanggal dengan simulasi
I_data_plot = I_data_used[: n_steps + 1]
tanggal_plot = tanggal_used[: n_steps + 1]

# Hitung error antara data dan model
diff = I_sim - I_data_plot
mse = float(np.mean(diff**2))
mae = float(np.mean(np.abs(diff)))

# ---------------------------------------------------------
# 5. Tampilkan hasil (grafik + metrik error)
# ---------------------------------------------------------

streamlit.subheader("Hasil Simulasi vs Data Asli")

# Plot dengan matplotlib
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(tanggal_plot, I_data_plot, label="Data asli (kasus aktif)")
ax.plot(tanggal_plot, I_sim, "--", label="Simulasi Euler (dI/dt = rI)")
ax.set_xlabel("Tanggal")
ax.set_ylabel("Kasus aktif (total Jawa Barat)")
ax.set_title("Perbandingan Data vs Simulasi Euler")
ax.legend()
plt.xticks(rotation=45)
plt.tight_layout()

streamlit.pyplot(fig)

# Tampilkan parameter & error
streamlit.markdown(
    f"""
### Ringkasan Parameter & Error

**Parameter simulasi saat ini:**
- Laju pertumbuhan \\( r \\) = `{r:.4f}` per hari  
- Step size \\( h \\) = `{h:.2f}` hari per langkah  
- Jumlah langkah Euler = `{n_steps}`  

**Error antara model dan data (periode simulasi):**
- Mean Squared Error (MSE) = `{mse:.2f}`  
- Mean Absolute Error (MAE) = `{mae:.2f}`
"""
)

streamlit.info(
    "Coba ubah nilai r dan h. "
    "Perhatikan bagaimana bentuk kurva simulasi dan nilai error (MSE/MAE) berubah. "
)
