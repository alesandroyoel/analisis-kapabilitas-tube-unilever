import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import re

# --- KONFIGURASI TARGET & BATASAN ---
TARGET_CPK = 1
MAX_SUPPLIERS = 5

# Konfigurasi Halaman
st.set_page_config(page_title="Analisis Kapabilitas & Peringkat Supplier", layout="wide")

st.title("🏭 Analisis Kapabilitas & Peringkat Supplier")

# --- FUNGSI PARSING ---
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
st.sidebar.header("📂 Data Input")
uploaded_files = st.sidebar.file_uploader(
    "Upload File Excel", 
    type=['xlsx', 'xls'], 
    accept_multiple_files=True
)

if uploaded_files:
    all_data_list = []
    skipped_sheets = []
    
    with st.spinner('Membaca data...'):
        for file in uploaded_files:
            try:
                file_date = parse_filename_date(file.name)
                xls = pd.read_excel(file, sheet_name=None, header=2)
                
                for sheet_name, df in xls.items():
                    mc, supplier = parse_sheet_info(sheet_name)
                    if mc is None: 
                        skipped_sheets.append(f"{file.name} - {sheet_name}")
                        continue 
                    
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

    if skipped_sheets:
        with st.expander("⚠️ Beberapa sheet dilewati (Format nama tidak sesuai)"):
            st.write(skipped_sheets)

    if all_data_list:
        master_df = pd.concat(all_data_list, ignore_index=True)

        # --- BAGIAN 2: FILTER BERJENJANG (CASCADING) ---
        st.sidebar.markdown("---")
        st.sidebar.header("⚙️ Filter Analisis")
        
        # FILTER 1: MATERIAL CODE
        mcs = master_df['Material_Code'].unique()
        sel_mc = st.sidebar.selectbox(
            "Material Code", 
            options=mcs, 
            index=None, 
            placeholder="Pilih Material..."
        )
        
        if not sel_mc:
            st.info("👈 Pilih **Material Code** di sidebar.")
        else:
            df_mc_all = master_df[master_df['Material_Code'] == sel_mc]
            
            # FILTER 2: SUPPLIER 
            suppliers = df_mc_all['Supplier'].unique()
            sel_sups = st.sidebar.multiselect(
                f"Supplier (Maks {MAX_SUPPLIERS})", 
                options=suppliers, 
                default=[], 
                max_selections=MAX_SUPPLIERS,
                help=f"Supplier untuk material {sel_mc}."
            )
            
            if not sel_sups:
                st.info("👈 Pilih minimal satu **Supplier** di sidebar.")
            else:
                df_sup_all = df_mc_all[df_mc_all['Supplier'].isin(sel_sups)]
                
                # FILTER 3: TANGGAL 
                dates = sorted(df_sup_all['Source_Date'].unique())
                sel_dates = st.sidebar.multiselect(
                    "Tanggal", 
                    options=dates, 
                    default=[],
                    help=f"Material {sel_mc} dari supplier {', '.join(sel_sups)} diambil pada tanggal-tanggal berikut."
                )
                
                if not sel_dates:
                    st.info("👈 Pilih **Tanggal** pengamatan di sidebar.")
                else:
                    # --- KETIGA FILTER SUDAH TERISI -> TAMPILKAN DASHBOARD ---
                    df_final = df_sup_all[df_sup_all['Source_Date'].isin(sel_dates)]

                    st.write(f"### Analisis Detail & Komparasi: {sel_mc}")
                    
                    numeric_cols = df_final.select_dtypes(include=[np.number]).columns.tolist()
                    exclude = ['Source_File', 'Material_Code', 'Supplier', 'No', 'Urutan']
                    numeric_cols = [c for c in numeric_cols if c not in exclude]

                    if numeric_cols:
                        col_param1, col_param2 = st.columns([1, 3])
                        
                        with col_param1:
                            st.info("👇 Atur Spesifikasi")
                            measure_col = st.selectbox("Parameter Ukur", numeric_cols)
                            
                            data_all = df_final[measure_col].dropna()
                            if not data_all.empty:
                                s_min, s_max = data_all.min(), data_all.max()
                                lsl = st.number_input("LSL", value=float(s_min*0.9), format="%.4f")
                                usl = st.number_input("USL", value=float(s_max*1.1), format="%.4f")
                                target = st.number_input("Target", value=float((lsl+usl)/2), format="%.4f")
                            else:
                                st.error("Data pada tanggal ini kosong.")
                                st.stop()
                        
                        with col_param2:
                            st.markdown("##### 🏆 Perbandingan Kapabilitas Supplier")
                            for sup in sel_sups:
                                data_sup = df_final[df_final['Supplier'] == sup][measure_col].dropna()
                                
                                if len(data_sup) > 1:
                                    mu = np.mean(data_sup)
                                    sigma = np.std(data_sup, ddof=1)
                                    
                                    d_min = np.min(data_sup)
                                    d_max = np.max(data_sup)
                                    
                                    Cp = (usl - lsl) / (6 * sigma)
                                    Cpu = (usl - mu) / (3 * sigma)
                                    Cpl = (mu - lsl) / (3 * sigma)
                                    Cpk = min(Cpu, Cpl)
                                    
                                    st.markdown(f"**🔹 {sup}** (Sampel n={len(data_sup)})")
                                    
                                    # --- MODIFIKASI 1: METRIK DIBUAT 2 BARIS ---
                                    col_m1, col_m2, col_m3 = st.columns(3)
                                    col_m4, col_m5, col_m6 = st.columns(3)
                                    
                                    col_m1.metric("Cp", f"{Cp:.2f}")
                                    
                                    delta_val = Cpk - TARGET_CPK
                                    warnanya = "normal" if Cpk >= TARGET_CPK else "inverse"
                                    col_m2.metric("Cpk", f"{Cpk:.2f}", delta=f"{delta_val:.2f} vs Target", delta_color=warnanya)
                                    
                                    col_m3.metric("Mean", f"{mu:.2f}")
                                    
                                    col_m4.metric("Std Dev", f"{sigma:.2f}")
                                    col_m5.metric("Min", f"{d_min:.2f}")
                                    col_m6.metric("Max", f"{d_max:.2f}")
                                    
                                    st.markdown("---")
                                else:
                                    st.warning(f"Data {sup} tidak cukup (n={len(data_sup)})")

                        # --- BAGIAN 4: TABS VISUALISASI ---
                        # --- MODIFIKASI 2: MENAMBAH TABS "Tren Cpk Harian" ---
                        tab1, tab2, tab3 = st.tabs(["📊 Histogram Komparasi", "📈 Run Chart Komparasi", "📅 Tren Cpk Harian"])
                        colors = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6', '#f1c40f', '#e67e22', '#1abc9c']
                        
                        with tab1:
                            fig1, ax1 = plt.subplots(figsize=(10, 4))
                            
                            for i, sup in enumerate(sel_sups):
                                data_sup = df_final[df_final['Supplier'] == sup][measure_col].dropna()
                                if len(data_sup) > 1:
                                    c = colors[i % len(colors)]
                                    
                                    ax1.hist(data_sup, bins=20, density=True, alpha=0.5, label=sup, color=c)
                                    
                                    mu = np.mean(data_sup)
                                    sigma = np.std(data_sup, ddof=1)
                                    x = np.linspace(min(lsl, data_all.min()), max(usl, data_all.max()), 100)
                                    p = stats.norm.pdf(x, mu, sigma)
                                    
                                    ax1.plot(x, p, linewidth=2, color=c)

                            ax1.axvline(lsl, color='r', linestyle='--', linewidth=2, label='LSL')
                            ax1.axvline(usl, color='r', linestyle='--', linewidth=2, label='USL')
                            ax1.set_title(f"Histogram & Kurva Distribusi: {measure_col}")
                            
                            handles, labels = ax1.get_legend_handles_labels()
                            by_label = dict(zip(labels, handles))
                            ax1.legend(by_label.values(), by_label.keys(), loc='upper left', bbox_to_anchor=(1.02, 1), fontsize='small')
                            st.pyplot(fig1)

                        with tab2:
                            fig2, ax2 = plt.subplots(figsize=(10, 4))
                            
                            for i, sup in enumerate(sel_sups):
                                data_sup = df_final[df_final['Supplier'] == sup][measure_col].dropna()
                                if len(data_sup) > 0:
                                    c = colors[i % len(colors)]
                                    x_vals = range(1, len(data_sup) + 1)
                                    y_vals = data_sup.values
                                    
                                    ax2.plot(x_vals, y_vals, marker='o', linestyle='-', color=c, markersize=4, alpha=0.8, label=sup)
                                    
                                    mask_reject = (y_vals > usl) | (y_vals < lsl)
                                    if mask_reject.any():
                                        ax2.scatter(np.array(x_vals)[mask_reject], y_vals[mask_reject], color='red', s=80, zorder=5)

                            ax2.axhline(usl, color='r', linestyle='--', linewidth=2, label=f'USL ({usl})')
                            ax2.axhline(lsl, color='r', linestyle='--', linewidth=2, label=f'LSL ({lsl})')
                            if target:
                                ax2.axhline(target, color='green', linestyle='-', alpha=0.5, label='Target')

                            ax2.set_xlabel("Sampel")
                            ax2.set_ylabel(measure_col)
                            ax2.set_title(f"Trend Data: {measure_col}")
                            
                            handles, labels = ax2.get_legend_handles_labels()
                            by_label = dict(zip(labels, handles))
                            ax2.legend(by_label.values(), by_label.keys(), loc='upper left', bbox_to_anchor=(1.02, 1), fontsize='small')
                            ax2.grid(True, linestyle=':', alpha=0.5)
                            st.pyplot(fig2)

                        # --- MODIFIKASI 3: LOGIKA TREN HARIAN ---
                        with tab3:
                            st.markdown("#### 📅 Breakdown Kapabilitas Berdasarkan Tanggal")
                            st.caption("Detail performa statistik per tanggal")
                            
                            daily_records = []
                            available_dates = sorted(df_final['Source_Date'].unique())
                            
                            for sup in sel_sups:
                                for d in available_dates:
                                    d_data = df_final[(df_final['Supplier'] == sup) & (df_final['Source_Date'] == d)][measure_col].dropna()
                                    
                                    if len(d_data) > 1:
                                        mu_d = np.mean(d_data)
                                        sig_d = np.std(d_data, ddof=1)
                                        Cpu_d = (usl - mu_d) / (3 * sig_d)
                                        Cpl_d = (mu_d - lsl) / (3 * sig_d)
                                        Cpk_d = min(Cpu_d, Cpl_d)
                                        
                                        daily_records.append({
                                            "Tanggal": d,
                                            "Supplier": sup,
                                            "Cpk": Cpk_d,
                                            "Mean": mu_d,
                                            "Std Dev": sig_d,
                                            "Min": np.min(d_data),
                                            "Max": np.max(d_data),
                                            "Sampel (n)": len(d_data)
                                        })
                            
                            if daily_records:
                                df_daily = pd.DataFrame(daily_records)
                                
                                st.dataframe(
                                    df_daily.style.format({
                                        "Cpk": "{:.2f}",
                                        "Mean": "{:.2f}",
                                        "Std Dev": "{:.2f}",
                                        "Min": "{:.2f}",
                                        "Max": "{:.2f}"
                                    }).background_gradient(subset=['Cpk'], cmap='RdYlGn', vmin=0.5, vmax=2.0),
                                    use_container_width=True
                                )
                                
                                if len(available_dates) > 1:
                                    st.markdown("#### 📈 Pergerakan Nilai Cpk")
                                    fig3, ax3 = plt.subplots(figsize=(10, 4))
                                    
                                    for i, sup in enumerate(sel_sups):
                                        sup_daily = df_daily[df_daily['Supplier'] == sup].sort_values('Tanggal')
                                        if not sup_daily.empty:
                                            c = colors[i % len(colors)]
                                            ax3.plot(sup_daily['Tanggal'], sup_daily['Cpk'], marker='o', linestyle='-', color=c, label=sup, linewidth=2, markersize=8)
                                    
                                    ax3.axhline(TARGET_CPK, color='r', linestyle='--', label=f'Target Cpk ({TARGET_CPK})')
                                    ax3.set_xlabel("Tanggal")
                                    ax3.set_ylabel("Nilai Cpk")
                                    ax3.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize='small')
                                    ax3.grid(True, linestyle=':', alpha=0.6)
                                    plt.xticks(rotation=45) 
                                    st.pyplot(fig3)
                                else:
                                    st.info("💡 Pilih minimal 2 tanggal di sidebar untuk melihat grafik pergerakan performa.")
                            else:
                                st.warning("Data harian tidak cukup untuk dianalisis.")

                        # --- BAGIAN 5: LEADERBOARD SUPPLIER ---
                        if len(sel_sups) > 1:
                            st.write("---")
                            st.header(f"🥇 Peringkat Komparasi Supplier: {sel_mc}")
                            st.caption("Tabel perbandingan atau peringkat supplier yang diurutkan dari nilai Cpk.")
                            
                            leaderboard_data = []
                            
                            for sup in sel_sups:
                                d_sup = df_final[df_final['Supplier'] == sup][measure_col].dropna()
                                if len(d_sup) > 1:
                                    m_sup = np.mean(d_sup)
                                    s_sup = np.std(d_sup, ddof=1)
                                    
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
                                df_leaderboard = df_leaderboard.sort_values(by="Cpk", ascending=False).reset_index(drop=True)
                                
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

else:
    st.info("Silakan upload file Excel di sidebar.")
