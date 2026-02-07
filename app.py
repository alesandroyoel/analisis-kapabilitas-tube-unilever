import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import re

# --- KONFIGURASI TARGET (Bisa diubah) ---
TARGET_CPK = 1.33 

# Konfigurasi Halaman
st.set_page_config(page_title="Analisis Kualitas & Peringkat Supplier", layout="wide")

st.title("🏭 Analisis Kualitas & Peringkat Supplier")

# --- FUNGSI PARSING HELPER ---
def parse_filename_date(filename):
    match = re.search(r'(\d{8})', filename)
    if match:
        date_str = match.group(1)
        return f"{date_str[4:]}-{date_str[2:4]}-{date_str[:2]}" 
    return "Unknown_Date"

def parse_sheet_info(sheet_name):
    parts = sheet_name.split('_')
    if len(parts) >= 2:
        mc = parts[0]
        supplier = "_".join(parts[1:]) 
        return mc, supplier
    return None, None

# --- BAGIAN 1: UPLOAD DATA ---
# ... (kode import dan setup halaman tetap sama) ...

# --- MODIFIKASI FUNGSI PARSING (Agar lebih ketat) ---
def parse_sheet_info(sheet_name):
    parts = sheet_name.split('_')
    # Minimal harus ada 2 bagian (MC dan Supplier)
    if len(parts) >= 2:
        mc = parts[0]
        supplier = "_".join(parts[1:]) 
        return mc, supplier
    return None, None  # Kembalikan None jika format salah

# --- BAGIAN 1: UPLOAD DATA ---
st.sidebar.header("1. Upload Data")
uploaded_files = st.sidebar.file_uploader(
    "Upload File Excel (Non_skip_lot_*.xlsx)", 
    type=['xlsx', 'xls'], 
    accept_multiple_files=True
)

master_df = pd.DataFrame()

if uploaded_files:
    all_data_list = []
    skipped_sheets = [] # Untuk mencatat sheet yang dilewati
    
    with st.spinner('Membaca data...'):
        for file in uploaded_files:
            try:
                file_date = parse_filename_date(file.name)
                xls = pd.read_excel(file, sheet_name=None, header=2)
                
                for sheet_name, df in xls.items():
                    # Cek Format Nama Sheet
                    mc, supplier = parse_sheet_info(sheet_name)
                    
                    if mc is None: 
                        # JIKA FORMAT SALAH: Lewati sheet ini
                        skipped_sheets.append(f"{file.name} - {sheet_name}")
                        continue 
                    
                    # JIKA FORMAT BENAR: Lanjut proses
                    df = df.dropna(how='all', axis=1)
                    
                    df['Source_Date'] = file_date
                    df['Source_File'] = file.name
                    df['Material_Code'] = mc
                    df['Supplier'] = supplier
                    
                    if 'No' in df.columns:
                        df['Urutan'] = df['No']
                    else:
                        df['Urutan'] = range(1, len(df) + 1)
                        
                    all_data_list.append(df)
            except Exception as e:
                st.error(f"Error file {file.name}: {e}")

    # Tampilkan Peringatan jika ada sheet yang di-skip (Opsional, agar user tau)
    if skipped_sheets:
        with st.expander("⚠️ Beberapa sheet dilewati (Format nama tidak sesuai)"):
            st.write("Sheet berikut tidak memiliki format 'MC_Supplier' sehingga tidak dianalisis:")
            st.write(skipped_sheets)

    if all_data_list:
        master_df = pd.concat(all_data_list, ignore_index=True)

        # --- BAGIAN 2: FILTER SIDEBAR ---
        st.sidebar.markdown("---")
        st.sidebar.header("2. Filter Data")
        
        # Pilih Material Code
        mcs = master_df['Material_Code'].unique()
        sel_mc = st.sidebar.selectbox("Material Code", mcs)
        
        # Filter awal berdasarkan MC (untuk keperluan Leaderboard nanti)
        df_mc_all = master_df[master_df['Material_Code'] == sel_mc]
        
        # Pilih Supplier (Untuk Analisis Detail di Atas)
        suppliers = df_mc_all['Supplier'].unique()
        sel_sup = st.sidebar.selectbox("Pilih Supplier (Analisis Detail)", suppliers)
        
        # Filter Final untuk Visualisasi Utama
        df_final = df_mc_all[df_mc_all['Supplier'] == sel_sup]
        
        dates = sorted(df_final['Source_Date'].unique())
        sel_dates = st.sidebar.multiselect("Pilih Tanggal", dates, default=dates)
        if sel_dates:
            df_final = df_final[df_final['Source_Date'].isin(sel_dates)]

        # --- BAGIAN 3: PARAMETER ---
        st.write(f"### Analisis Detail: {sel_mc} - {sel_sup}")
        
        numeric_cols = df_final.select_dtypes(include=[np.number]).columns.tolist()
        exclude = ['Source_File', 'Material_Code', 'Supplier', 'No', 'Urutan']
        numeric_cols = [c for c in numeric_cols if c not in exclude]

        if numeric_cols:
            col_param1, col_param2 = st.columns([1, 3])
            
            with col_param1:
                st.info("👇 Atur Spesifikasi")
                measure_col = st.selectbox("Parameter Ukur", numeric_cols)
                
                # Auto-suggest
                data = df_final[measure_col].dropna()
                s_min, s_max = data.min(), data.max()
                
                lsl = st.number_input("LSL", value=s_min*0.9, format="%.4f")
                usl = st.number_input("USL", value=s_max*1.1, format="%.4f")
                target = st.number_input("Target", value=(lsl+usl)/2, format="%.4f")
            
            with col_param2:
                # Perhitungan Statistik
                mu = np.mean(data)
                sigma = np.std(data, ddof=1)
                d_min = np.min(data)
                d_max = np.max(data)
                
                # Hitung Cp Cpk
                Cp = (usl - lsl) / (6 * sigma)
                Cpu = (usl - mu) / (3 * sigma)
                Cpl = (mu - lsl) / (3 * sigma)
                Cpk = min(Cpu, Cpl)
                
                # --- VISUAL METRICS (Highlight Cpk) ---
                st.markdown("##### 🏆 Indikator Kapabilitas")
                kpi1, kpi2, kpi3 = st.columns([1, 1, 2]) 
                
                kpi1.metric("Cp", f"{Cp:.2f}")
                
                # Logika Warna Cpk
                delta_val = Cpk - TARGET_CPK
                warnanya = "normal" if Cpk >= TARGET_CPK else "inverse"
                kpi2.metric("Cpk", f"{Cpk:.2f}", delta=f"{delta_val:.2f} vs Target {TARGET_CPK}", delta_color=warnanya)

                st.markdown("---") 

                # Statistik Deskriptif
                st.markdown("##### 📊 Statistik Data")
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Mean", f"{mu:.2f}")
                m2.metric("Std Dev", f"{sigma:.2f}")
                m3.metric("Min", f"{d_min:.2f}")
                m4.metric("Max", f"{d_max:.2f}")
                
                st.markdown("---")

                # --- BAGIAN 4: TABS VISUALISASI ---
                tab1, tab2 = st.tabs(["📊 Histogram", "📈 Run Chart (Data vs Spek)"])
                
                # TAB 1: HISTOGRAM
                with tab1:
                    fig1, ax1 = plt.subplots(figsize=(10, 4))
                    ax1.hist(data, bins=25, density=True, alpha=0.6, color='#3498db', label='Data')
                    x = np.linspace(min(lsl, data.min()), max(usl, data.max()), 100)
                    p = stats.norm.pdf(x, mu, sigma)
                    ax1.plot(x, p, 'k', linewidth=2, label='Normal Fit')
                    ax1.axvline(lsl, color='r', linestyle='--', label='LSL')
                    ax1.axvline(usl, color='r', linestyle='--', label='USL')
                    ax1.axvline(mu, color='g', linestyle='-', label='Mean')
                    ax1.set_title(f"Histogram: {measure_col}")
                    ax1.legend()
                    st.pyplot(fig1)

                # TAB 2: RUN CHART (TANPA UCL/LCL)
                with tab2:
                    st.markdown("**Run Chart: Data Aktual vs Batas Spesifikasi**")
                    fig2, ax2 = plt.subplots(figsize=(10, 4))
                    x_values = range(1, len(data)+1)
                    
                    # Plot Data
                    ax2.plot(x_values, data, marker='o', linestyle='-', color='#2c3e50', markersize=4, label='Data Aktual')
                    
                    # Garis Spek (USL/LSL)
                    ax2.axhline(usl, color='r', linestyle='--', linewidth=2, label=f'USL ({usl})')
                    ax2.axhline(lsl, color='r', linestyle='--', linewidth=2, label=f'LSL ({lsl})')
                    
                    # Garis Tengah (Mean/Target)
                    if target:
                         ax2.axhline(target, color='green', linestyle='-', alpha=0.5, label='Target')
                    else:
                         ax2.axhline(mu, color='green', linestyle='-', alpha=0.5, label='Mean')

                    # Highlight Titik Reject (Keluar Spek)
                    rejects = data[(data > usl) | (data < lsl)]
                    if not rejects.empty:
                        ax2.scatter(rejects.index + 1, rejects, color='red', s=100, zorder=5, label='Reject (Out of Spec)')

                    ax2.set_xlabel("Nomor Sampel")
                    ax2.set_ylabel(measure_col)
                    ax2.set_title(f"Trend Data: {measure_col}")
                    ax2.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
                    ax2.grid(True, linestyle=':', alpha=0.5)
                    st.pyplot(fig2)

            # --- BAGIAN 5: LEADERBOARD SUPPLIER ---
            st.write("---")
            st.header(f"🥇 Peringkat Supplier untuk Material: {sel_mc}")
            st.markdown(f"Tabel ini membandingkan semua supplier yang menyuplai **{sel_mc}** berdasarkan parameter **{measure_col}** dan batas spek yang Anda masukkan di atas.")

            # Hitung Cpk untuk SEMUA supplier di MC ini
            leaderboard_data = []
            
            # Ambil semua supplier yg punya MC ini
            all_suppliers_in_mc = df_mc_all['Supplier'].unique()
            
            for sup in all_suppliers_in_mc:
                # Ambil data supplier tsb
                d_sup = df_mc_all[df_mc_all['Supplier'] == sup][measure_col].dropna()
                
                if len(d_sup) > 1:
                    m_sup = np.mean(d_sup)
                    s_sup = np.std(d_sup, ddof=1)
                    
                    # Hitung Cpk pakai USL/LSL yang sama dengan inputan user tadi
                    Cpu_s = (usl - m_sup) / (3 * s_sup)
                    Cpl_s = (m_sup - lsl) / (3 * s_sup)
                    Cpk_s = min(Cpu_s, Cpl_s)
                    
                    leaderboard_data.append({
                        "Supplier": sup,
                        "Cpk": Cpk_s,
                        "Mean": m_sup,
                        "Min": np.min(d_sup),
                        "Max": np.max(d_sup),
                        "Sampel": len(d_sup)
                    })
            
            if leaderboard_data:
                df_leaderboard = pd.DataFrame(leaderboard_data)
                # Urutkan Ranking
                df_leaderboard = df_leaderboard.sort_values(by="Cpk", ascending=False).reset_index(drop=True)
                
                # Tampilkan Tabel
                st.dataframe(
                    df_leaderboard.style.format({
                        "Cpk": "{:.2f}",
                        "Mean": "{:.2f}",
                        "Min": "{:.2f}",
                        "Max": "{:.2f}"
                    }).background_gradient(subset=['Cpk'], cmap='RdYlGn', vmin=0.5, vmax=2.0),
                    use_container_width=True
                )
            else:
                st.warning("Data tidak cukup untuk membuat peringkat.")

        else:
            st.warning("Tidak ada kolom numerik untuk dianalisis.")