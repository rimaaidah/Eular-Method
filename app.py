import streamlit
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# Konfigurasi halaman
# ---------------------------------------------------------
streamlit.set_page_config(page_title="SIR Euler COVID Jabar", layout="wide")

# Populasi total (kira-kira) Jawa Barat
N = 50_000_000  # bisa kamu tulis juga di laporan


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


# ---------------------------------------------------------
# 2. Model SIR & Metode Euler 2D
# ---------------------------------------------------------
def f_sir(t, y, params):
    """
    Model SIR sederhana dalam bentuk proporsi:

        ds/dt = -beta * s * i
        di/dt =  beta * s * i - gamma * i

    y = [s, i], params = {"beta": ..., "gamma": ...}
    """
    s, i = y
    beta = params["beta"]
    gamma = params["gamma"]

    dsdt = -beta * s * i
    didt = beta * s * i - gamma * i
    return np.array([dsdt, didt])


def euler_solve_sir(f, t0, y0, h, n_steps, params):
    """
    Metode Euler untuk sistem SIR 2 variabel (s dan i).

    f       : fungsi ODE f(t, y, params)
    t0      : waktu awal
    y0      : array awal [s0, i0]
    h       : step size (hari per langkah)
    n_steps : jumlah langkah Euler
    params  : dict parameter (beta, gamma)
    """
    t_values = np.zeros(n_steps + 1)
    y_values = np.zeros((n_steps + 1, len(y0)))

    t_values[0] = t0
    y_values[0, :] = y0

    for n in range(n_steps):
        y_values[n+1, :] = y_values[n, :] + h * f(t_values[n], y_values[n, :], params)
        t_values[n+1] = t_values[n] + h

    return t_values, y_values


# ---------------------------------------------------------
# 3. MAIN APP
# ---------------------------------------------------------

streamlit.title("Model SIR + Metode Euler untuk Kasus Aktif COVID-19 Jawa Barat")

streamlit.write(
    r"""
Aplikasi ini menggunakan data **kasus aktif COVID-19** dari Open Data Jawa Barat,
kemudian memodelkan dinamika penularannya dengan **model SIR sederhana**:

\[
\begin{aligned}
\frac{dS}{dt} &= -\beta S I, \\
\frac{dI}{dt} &= \beta S I - \gamma I.
\end{aligned}
\]

dengan:
- \(S(t)\): jumlah penduduk **rentan** (susceptible),
- \(I(t)\): jumlah penduduk **terinfeksi / kasus aktif**,
- \(\beta\): laju penularan,
- \(\gamma\): laju kesembuhan / keluar dari status infeksi.

Pada implementasi ini, persamaan diselesaikan secara numerik menggunakan **Metode Euler**.
"""
)

# 3.1 Load data
csv_path = "covid_jabar_perkembangan_harian.csv"
df_group = load_data(csv_path)

streamlit.subheader("Cuplikan Data COVID-19 (Total Kasus Aktif per Tanggal)")
streamlit.dataframe(df_group.head())

# ---------------------------------------------------------
# 4. Sidebar: parameter input
# ---------------------------------------------------------

streamlit.sidebar.header("Pengaturan Simulasi Model SIR")

# Data I(t) asli (jumlah orang)
I_data = df_group["konfirmasi_aktif"].values.astype(float)
tanggal_all = df_group["tanggal"].values

# Inisialisasi i0 dan s0 dalam proporsi
I0_count = I_data[0]
i0 = I0_count / N
s0 = 1.0 - i0  # asumsi awal: hampir semua masih rentan, R0 ≈ 0

# Estimasi r awal dari data (untuk membantu set beta default)
mask_pos = I_data > 0
r_est = 0.1
if mask_pos.sum() >= 2:
    I_pos = I_data[mask_pos]
    growth_logs = np.log(I_pos[1:] / I_pos[:-1])
    r_est = float(growth_logs.mean())

# Asumsi awal gamma (misal rata-rata lama infeksi 10 hari -> gamma ≈ 0.1)
gamma_default = 0.1

# Dari pendekatan linear: r ≈ beta * S0 - gamma  => beta ≈ (r + gamma) / S0
beta_default = (r_est + gamma_default) / s0 if s0 > 0 else 0.2

beta = streamlit.sidebar.number_input(
    "β (laju penularan)",
    value=round(beta_default, 4),
    step=0.01,
    format="%.4f",
    help="Semakin besar β, penularan antar individu rentan dan terinfeksi semakin cepat."
)

gamma = streamlit.sidebar.number_input(
    "γ (laju kesembuhan)",
    value=round(gamma_default, 4),
    step=0.01,
    format="%.4f",
    help="Sekitar γ ≈ 1 / (lama rata-rata hari seseorang terinfeksi)."
)

h = streamlit.sidebar.slider(
    "Step size h (hari per langkah Euler)",
    min_value=0.1,
    max_value=2.0,
    value=1.0,
    step=0.1,
    help="Semakin kecil h, simulasi lebih halus tetapi butuh lebih banyak langkah."
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
# 5. Menjalankan simulasi SIR dengan Euler
# ---------------------------------------------------------

# Hitung jumlah langkah Euler
n_steps = int(duration_days / h)
if n_steps < 1:
    streamlit.error("Kombinasi durasi dan h membuat jumlah langkah terlalu sedikit.")
    streamlit.stop()

# Jangan melebihi panjang data
n_steps = min(n_steps, len(I_data) - 1)

# Kondisi awal (dalam proporsi)
y0 = np.array([s0, i0])
params = {"beta": beta, "gamma": gamma}

t_sim, y_sim = euler_solve_sir(f_sir, t0=0.0, y0=y0, h=h, n_steps=n_steps, params=params)

s_sim = y_sim[:, 0]           # proporsi S(t)
i_sim = y_sim[:, 1]           # proporsi I(t)
I_sim_count = i_sim * N       # kembali ke "jumlah orang"

# Sesuaikan data asli dengan panjang simulasi
I_data_plot = I_data[: n_steps + 1]
tanggal_plot = tanggal_all[: n_steps + 1]

# Hitung error
diff = I_sim_count - I_data_plot
mse = float(np.mean(diff**2))
mae = float(np.mean(np.abs(diff)))

# ---------------------------------------------------------
# 6. Tampilkan hasil
# ---------------------------------------------------------

streamlit.subheader("Hasil Simulasi SIR vs Data Kasus Aktif")

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(tanggal_plot, I_data_plot, label="Data asli (kasus aktif)")
ax.plot(tanggal_plot, I_sim_count, "--", label="Simulasi SIR (Metode Euler)")
ax.set_xlabel("Tanggal")
ax.set_ylabel("Kasus aktif (jumlah orang)")
ax.set_title("Perbandingan Data vs Simulasi Model SIR (Euler)")
ax.legend()
plt.xticks(rotation=45)
plt.tight_layout()

streamlit.pyplot(fig)

streamlit.markdown(
    f"""
### Ringkasan Parameter & Error

**Parameter simulasi saat ini:**
- β (laju penularan) = `{beta:.4f}`  
- γ (laju kesembuhan) = `{gamma:.4f}`  
- Step size \\( h \\) = `{h:.2f}` hari per langkah  
- Jumlah langkah Euler = `{n_steps}`  
- Populasi total diasumsikan \\( N = {N:,} \\) orang  

**Error antara model dan data (periode simulasi):**
- Mean Squared Error (MSE) = `{mse:.2f}`  
- Mean Absolute Error (MAE) = `{mae:.2f}`
"""
)

streamlit.info(
    "Coba ubah nilai β dan γ di sidebar, lalu amati bagaimana bentuk kurva I(t) dan nilai error (MSE/MAE) berubah. "
    "Pengaturan parameter yang membuat kurva simulasi paling mendekati data dapat dianggap sebagai model yang paling cocok untuk periode tersebut."
)
