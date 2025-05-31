import pandas as pd
import os
import mlflow

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

def preprocessing_data(input_file,output_dir):
    # Load dataset
    df = pd.read_csv(input_file)
    
    # Cek jumlah missing values pada dataset
    df.isnull().sum()
    # Tidak Ada Mising Values
    
    # Mengatasi Data Duplikat
    # Mengecek jumlah baris duplikat
    jumlah_duplikat = df.duplicated().sum()
    print(f"Jumlah data duplikat: {jumlah_duplikat}")
    # Menyimpan jumlah data awal sebelum penghapusan
    jumlah_data_awal = df.shape[0]
    # Menghapus baris duplikat
    df = df.drop_duplicates()
    # Menyimpan jumlah data setelah duplikat dihapus
    jumlah_data_setelah = df.shape[0]
    # Menghitung jumlah data yang dihapus
    jumlah_dihapus = jumlah_data_awal - jumlah_data_setelah
    # Menampilkan hasil
    print(f"Jumlah data awal             : {jumlah_data_awal}")
    print(f"Jumlah data setelah dihapus  : {jumlah_data_setelah}")
    print(f"Jumlah data yang dihapus     : {jumlah_dihapus}")

    # Normalisasi dengan MinMaxScaler
    # Fitur numerik
    num_features = ['Umur (bulan)', 'Tinggi Badan (cm)']
    # Ubah ke float agar kompatibel dengan hasil MinMaxScaler
    df[num_features] = df[num_features].astype(float)
    # Inisialisasi dan transformasi
    scaler = MinMaxScaler()
    df.loc[:, num_features] = scaler.fit_transform(df[num_features])
    
    # Mendeteksi Outlier
    # Menghitung dan menampilkan jumlah outlier per fitur numerik
    num_features = df.select_dtypes(include=['number']).columns
    outlier_counts = {}
    for feature in num_features:
        Q1 = df[feature].quantile(0.25)
        Q3 = df[feature].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = df[(df[feature] < lower_bound) | (df[feature] > upper_bound)]
        outlier_counts[feature] = len(outliers)
    for feature, count in outlier_counts.items():
        print(f"{feature}: {count} outlier(s)")
    # Tidak Ada Outliers
    
    # Encoding Data Kategorikal
    # One-Hot Encoding untuk 'Jenis Kelamin'
    df_encoded = pd.get_dummies(df, columns=['Jenis Kelamin'], drop_first=True)
    df_encoded['Jenis Kelamin_perempuan'] = df_encoded['Jenis Kelamin_perempuan'].astype(int)
    # Mapping label untuk 'Status Gizi'
    mapping_status_gizi = {
        'normal': 0,
        'stunted': 1,
        'severely stunted': 2,
        'tinggi': 3
    }
    # Label Encoding Manual dengan Mapping
    df_encoded['Status Gizi'] = df_encoded['Status Gizi'].map(mapping_status_gizi)
    
    #Data Split
    # Pisahkan fitur (X) dan target (y)
    X = df.drop(columns=["Status Gizi"])  # Semua kolom kecuali target
    y = df["Status Gizi"]                 # Hanya kolom target
    # Bagi data latih dan uji
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Save data hasil preprocessing
    folder = "preprocessing/data_balita_preprocessing"
    if not os.path.exists(folder):
        os.makedirs(folder)
    X_train.to_csv(f"{folder}/X_train.csv", index=False)
    X_test.to_csv(f"{folder}/X_test.csv", index=False)
    y_train.to_csv(f"{folder}/y_train.csv", index=False)
    y_test.to_csv(f"{folder}/y_test.csv", index=False)
    print("Data preprocessing berhasil disimpan")
    
    return {
        "rows_clean": df.shape[0],
        "files": [
            os.path.join(output_dir, "X_train.csv"),
            os.path.join(output_dir, "X_test.csv"),
            os.path.join(output_dir, "y_train.csv"),
            os.path.join(output_dir, "y_test.csv"),
        ]
    }
    
if __name__ == "__main__":
    input_file = os.path.join(os.environ.get("GITHUB_WORKSPACE", "."), "data_balita_raw.csv")
    output_dir = os.path.join(os.getenv("GITHUB_WORKSPACE", "./"), "preprocessing/data_balita_preprocessing")

    print(f"Output directory: {output_dir}")
    print(f"Input file: {input_file}")

    mlruns_path = os.path.join(output_dir, "mlruns")
    os.makedirs(mlruns_path, exist_ok=True)

    mlflow.set_tracking_uri(f"file:{mlruns_path}")
    mlflow.set_experiment("Preprocessing_Experiment")

    with mlflow.start_run(run_name="Preprocessing_Run"):
        result = preprocessing_data(input_file, output_dir)

        mlflow.log_param("input_file", input_file)
        mlflow.log_param("output_dir", output_dir)
        mlflow.log_metric("rows_clean", result["rows_clean"])

        for f in result["files"]:
            mlflow.log_artifact(f)