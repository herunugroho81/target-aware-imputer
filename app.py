import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO

class TargetAwareImputer:
    def __init__(self, df, target_column):
        self.df = df.copy()
        self.target_column = target_column
        self.imputation_values = {}  # Menyimpan nilai imputasi per kelas per kolom

    def detect_missing(self):
        total_missing = self.df.isnull().sum()
        percent_missing = (total_missing / len(self.df)) * 100
        return pd.DataFrame({
            'Missing Count': total_missing,
            'Missing %': percent_missing
        }).loc[total_missing > 0]

    def auto_impute_by_target(self):
        for col in self.df.columns:
            if col == self.target_column:
                continue  # Tidak perlu imputasi untuk kolom target

            if self.df[col].isnull().any():
                if pd.api.types.is_numeric_dtype(self.df[col]):
                    # Hitung median per kelas target
                    impute_values = self.df.groupby(self.target_column)[col].transform('median')
                    self.df[col] = self.df[col].fillna(impute_values)
                    self.imputation_values[col] = dict(
                        self.df.groupby(self.target_column)[col].median()
                    )

                elif pd.api.types.is_categorical_dtype(self.df[col]) or self.df[col].dtype == 'object':
                    # Hitung mode per kelas target
                    mode_dict = {}
                    for cls in self.df[self.target_column].unique():
                        subset = self.df[self.df[self.target_column] == cls]
                        mode_value = subset[col].mode()[0] if not subset[col].mode().empty else "Unknown"
                        mode_dict[cls] = mode_value
                        self.df.loc[(self.df[col].isnull()) & (self.df[self.target_column] == cls), col] = mode_value
                    self.imputation_values[col] = mode_dict

                else:
                    # Default fallback
                    self.df[col] = self.df[col].fillna("Unknown")

        return self.df


# 🔧 Fungsi bantu untuk unduh file CSV
def to_csv(df):
    output = BytesIO()
    df.to_csv(output, index=False)
    return output.getvalue()


# 🖥️ UI Streamlit
st.set_page_config(page_title="Target-Aware Imputer", layout="wide")
st.title("🛠️ Aplikasi Imputasi Data Berdasarkan Kolom Target")
st.markdown("Unggah file CSV, pilih kolom target, dan aplikasi akan mengisi nilai kosong berdasarkan kelas.")

uploaded_file = st.file_uploader("Pilih file CSV", type=["csv"])

if uploaded_file is not None:
    try:
        df_raw = pd.read_csv(uploaded_file)
        st.success("✅ File berhasil dimuat!")

        st.subheader("🧾 Pratinjau Data Awal")
        st.dataframe(df_raw.head())

        target_col = st.selectbox("Pilih kolom target (kelas)", options=df_raw.columns.tolist(), index=0)

        if st.button("🚀 Jalankan Imputasi"):
            with st.spinner("Sedang memproses..."):
                imputer = TargetAwareImputer(df_raw, target_column=target_col)
                missing_report = imputer.detect_missing()

                st.subheader("🔍 Laporan Missing Value")
                if not missing_report.empty:
                    st.dataframe(missing_report)
                else:
                    st.info("Tidak ada missing value di dataset ini.")

                df_clean = imputer.auto_impute_by_target()

                st.subheader("📊 Hasil Setelah Imputasi")
                st.dataframe(df_clean.head())

                st.subheader("🔧 Nilai Imputasi yang Digunakan")
                for col, val in imputer.imputation_values.items():
                    st.write(f"**{col}**: {val}")

                csv_data = to_csv(df_clean)
                st.download_button(
                    label="💾 Unduh Data Setelah Imputasi",
                    data=csv_data,
                    file_name="data_hasil_imputasi.csv",
                    mime="text/csv"
                )

    except Exception as e:
        st.error(f"❌ Terjadi kesalahan saat memproses file:\n{e}")
else:
    st.info("📂 Silakan unggah file CSV untuk memulai.")
