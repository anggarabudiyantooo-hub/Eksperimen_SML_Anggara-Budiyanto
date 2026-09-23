# automate_Anggara-Budiyanto.py
# preprocessing otomatis dataset telco churn.
# langkahnya sama dengan notebook Eksperimen_Anggara-Budiyanto.ipynb, dalam bentuk fungsi.

import os
import json
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# lokasi file ini, biar path tidak kacau kalau dijalankan dari folder lain (misal di CI)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def preprocessing_telco(input_csv, output_csv=None):
    """baca data mentah telco churn, bersihkan, lalu kembalikan data siap latih"""
    df = pd.read_csv(input_csv)

    # TotalCharges kebaca sebagai teks karena ada yang isinya kosong
    # (pelanggan dengan tenure 0), jadi diubah ke angka lalu diisi median
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

    # customerID cuma identitas, tidak dipakai model
    df = df.drop("customerID", axis=1)

    # kolom Yes/No diubah jadi 1/0
    kolom_yesno = ["Partner", "Dependents", "PhoneService", "PaperlessBilling", "Churn"]
    for kolom in kolom_yesno:
        df[kolom] = df[kolom].map({"Yes": 1, "No": 0})

    # kolom teks sisanya di-one-hot encoding
    kolom_teks = df.select_dtypes(include="object").columns.tolist()
    df = pd.get_dummies(df, columns=kolom_teks, drop_first=True)

    # hasil one-hot kadang jadi True/False, diubah ke 0/1 biar seragam sama kolom lain
    kolom_bool = df.select_dtypes(include="bool").columns.tolist()
    if len(kolom_bool) > 0:
        df[kolom_bool] = df[kolom_bool].astype(int)

    # kolom angka di-scale ke 0-1
    kolom_angka = ["tenure", "MonthlyCharges", "TotalCharges"]
    scaler = MinMaxScaler()
    df[kolom_angka] = scaler.fit_transform(df[kolom_angka])

    # simpan hasilnya kalau ada output_csv
    if output_csv is not None:
        df.to_csv(output_csv, index=False)

        # simpan juga nama kolom fitur, dipakai lagi saat inference/monitoring
        kolom_fitur = [c for c in df.columns if c != "Churn"]
        with open(os.path.join(os.path.dirname(output_csv), "kolom_fitur.json"), "w") as f:
            json.dump(kolom_fitur, f)

    return df


if __name__ == "__main__":
    data_mentah = os.path.join(BASE_DIR, "..", "telco_churn_raw.csv")
    data_hasil = os.path.join(BASE_DIR, "telco_churn_preprocessing.csv")

    df_bersih = preprocessing_telco(data_mentah, data_hasil)

    print("preprocessing selesai!")
    print("data mentah   :", data_mentah)
    print("hasil disimpan:", data_hasil)
    print("ukuran data   :", df_bersih.shape)
    print(df_bersih.head())
