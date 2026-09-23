# Eksperimen_SML_Anggara-Budiyanto

Repo eksperimen dan preprocessing otomatis dataset Telco Customer Churn. Submission kelas Membangun Sistem Machine Learning (Dicoding). Sumber dataset: https://www.kaggle.com/datasets/blastchar/telco-customer-churn dan file CSV dari repositori resmi IBM di https://github.com/IBM/telco-customer-churn-on-icp4d.

## Isi repo

- `telco_churn_raw.csv` - dataset mentah (IBM Sample Data / Kaggle)
- `preprocessing/Eksperimen_Anggara-Budiyanto.ipynb` - notebook eksperimen: data loading, EDA, preprocessing
- `preprocessing/automate_Anggara-Budiyanto.py` - preprocessing otomatis, langkahnya sama dengan notebook
- `preprocessing/telco_churn_preprocessing.csv` - dataset hasil preprocessing, siap training
- `.github/workflows/preprocessing.yml` - menjalankan preprocessing tiap ada push, hasilnya di-commit balik

## Menjalankan preprocessing manual

```
python preprocessing/automate_Anggara-Budiyanto.py
```
