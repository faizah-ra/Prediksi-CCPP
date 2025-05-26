# Laporan Proyek Machine Learning - Nama Anda

## Domain Proyek
Pembangkit listrik adalah tulang punggung pertumbuhan ekonomi modern sekaligus penyumbang besar emisi karbon: sektor ini bertanggung jawab atas hampir 40 % total CO₂ global [IEA, 2024](https://www.iea.org/data-and-statistics?country=WORLD&fuel=Energy%20supply&indicator=Electricity%20generation). Oleh karena itu, meningkatkan efisiensi pembangkit listrik menjadi langkah penting dalam upaya mitigasi perubahan iklim dan pencapaian target net-zero emisi [IEA, 2024](https://www.iea.org/data-and-statistics?country=WORLD&fuel=Energy%20supply&indicator=Electricity%20generation).

Combined Cycle Power Plants (CCPP) yang menggabungkan turbin gas dan uap mampu mencapai efisiensi termal rata-rata 60,5 %, jauh melampaui efisiensi pembangkit siklus tunggal (~38 %) sekaligus memangkas konsumsi bahan bakar & emisi CO₂ hingga sekitar 35 %–40 % per MWh [EPRI, 2023](https://epridatabase.org/heat-rate). Dengan demikian, CCPP merupakan solusi transisi energi utama untuk menggantikan pembangkit fosil konvensional.

Namun, performa CCPP sangat dipengaruhi oleh variabel lingkungan suhu udara, kelembaban relatif, tekanan atmosfer, dan kecepatan angin yang berfluktuasi sepanjang hari dan musim. Data [NOAA,2024](https://www.ncei.noaa.gov/products/land-based-station/integrated-surface-database) menunjukkan pada suhu 35 °C, output CCPP dapat turun hingga 8 % dibanding kondisi optimal. Selain itu, [ASHRAE,2022](https://www.ashrae.org/technical-resources/bookstore/eba-performance-archive) melaporkan efisiensi menurun 2 %–4 % saat kelembaban relatif > 80 %. Variasi tersebut dapat menyebabkan penurunan daya yang signifikan dan mengancam kestabilan pasokan listrik.

Ketidakpastian output daya menjadi tantangan besar dalam manajemen sistem kelistrikan, terutama dengan penetrasi tinggi sumber terbarukan yang juga variatif. Menurut [ENTSO-E,2024](https://transparency.entsoe.eu), fluktuasi harian gabungan melebihi 5 GW tanpa prediksi akurat berpotensi menimbulkan ketidakseimbangan beban dan risiko pemadaman. Selain itu, operasi di luar kondisi optimal mempercepat keausan mesin, menaikkan biaya pemeliharaan, dan mempersingkat umur peralatan.

Untuk mengatasi tantangan tersebut, **model prediksi output daya berbasis data lingkungan real-time** menjadi sangat penting. Dengan memanfaatkan data historis dan **streaming** dari sensor SCADA, estimasi daya dapat menjadi lebih presisi dan adaptif, mendukung perencanaan operasional dinamis serta pengambilan keputusan operator [IRENA, 2023](https://www.irena.org/publications/2023). Sejumlah studi menunjukkan bahwa **Machine Learning (ML)** mampu meningkatkan akurasi prediksi output CCPP secara signifikan. [Tüfekci, 2014](http://dx.doi.org/10.1016/j.ijepes.2014.02.027) menggunakan Support Vector Regression, K-Nearest Neighbors, Random Forest, dan Gradient Boosting pada dataset UCI CCPP (2006–2011), mencapai nilai *R²* hingga 0,93 pada pengujian [Tüfekci, 2014](http://dx.doi.org/10.1016/j.ijepes.2014.02.027). Selanjutnya, Lobo et al. (2019) menerapkan *streaming regressors* pada platform Big Data untuk prediksi daya real-time CCPP, menurunkan waktu proses hingga 50% tanpa mengorbankan akurasi [Lobo et al., 2019](https://arxiv.org/abs/1907.11653).

Proyek ini bertujuan merancang dan membangun model prediksi output energi CCPP menggunakan beragam algoritma Machine Learning. Dengan memanfaatkan data variabel lingkungan seperti suhu udara, kelembaban, tekanan atmosfer, dan kecepatan angin model diharapkan mampu menghasilkan estimasi daya keluaran CCPP yang akurat berdasarkan kondisi nyata. Implementasi model ini akan mendukung operator pembangkit dalam mengambil keputusan operasional secara cepat dan tepat, sehingga efisiensi dan keandalan sistem dapat terjaga secara optimal.


## Business Understanding

### Problem Statements

Menjelaskan pernyataan masalah latar belakang:
- Fluktuasi output daya pada Combined Cycle Power Plants (CCPP) yang dipengaruhi oleh variabel lingkungan seperti suhu ambien, tekanan udara, kelembapan relatif, dan vakum kondensor menyebabkan kesulitan bagi operator dalam menjaga kestabilan pasokan listrik secara konsisten.
- Ketidakakuratan model prediksi daya yang saat ini digunakan berdampak pada perencanaan produksi energi dan efisiensi operasional yang kurang optimal, sehingga berpotensi meningkatkan biaya operasional serta emisi gas rumah kaca.
- Belum tersedia model prediksi daya CCPP yang mengintegrasikan teknik machine learning terkini dengan optimasi hyperparameter secara sistematis untuk meningkatkan akurasi dan keandalan prediksi, khususnya pada kondisi operasi yang bervariasi dan kompleks.
- Model baseline berbasis pendekatan termodinamika konvensional kurang mampu menangkap hubungan non-linear antara variabel lingkungan dan output daya, sehingga akurasi prediksi yang dihasilkan masih rendah, yang berdampak pada efektivitas optimasi beban dan perencanaan pemeliharaan pembangkit.

### Goals
- Mengembangkan model prediksi output daya CCPP yang mampu memproses variabel lingkungan secara efektif dan menghasilkan prediksi dengan akurasi tinggi.
- Meningkatkan keandalan perencanaan produksi energi dan efisiensi operasional melalui prediksi daya yang lebih presisi.
- Mengintegrasikan teknik machine learning terbaru dan metode optimasi hyperparameter untuk menghasilkan model yang robust dan adaptif terhadap variasi kondisi operasional.
- Menghasilkan model yang mampu menangkap hubungan non-linear antar variabel lingkungan dan output daya sehingga mendukung optimasi beban dan perencanaan pemeliharaan pembangkit.

### Solution Statements
- Mengimplementasikan dan membandingkan model Linear Regression sebagai baseline dengan model Gradient Boosting untuk membangun prediksi output daya pada Combined Cycle Power Plants (CCPP).
- Melakukan optimasi hyperparameter secara sistematis menggunakan teknik Random Search untuk meningkatkan performa model secara signifikan.
- Melakukan evaluasi model dengan menggunakan metrik kuantitatif seperti Mean Absolute Error (MAE), Root Mean Square Error (RMSE), dan koefisien determinasi (R²) untuk mengukur akurasi serta ketepatan prediksi yang dihasilkan.
- Menerapkan analisis korelasi menggunakan heat map sebagai metode visualisasi untuk memastikan pemilihan fitur yang relevan dan mempermudah interpretasi hubungan antara variabel input lingkungan dengan output daya.
  
## Data Understanding
Data yang digunakan dalam proyek ini merupakan dataset dari Combined Cycle Power Plant Data Set yang tersedia di UCI Machine Learning Repository. Dataset ini berisi pengukuran variabel lingkungan dan output daya listrik dari pembangkit listrik Combined Cycle Power Plant (CCPP) yang diambil dalam interval waktu tertentu. Dataset ini terdiri dari 9.568 entri data dengan 5 atribut numerik yang saling berkaitan.


### Variabel-variabel pada dataset Combined Cycle Power Plant adalah sebagai berikut:
- ```AT (Ambient Temperature)``` : Suhu udara ambien di sekitar pembangkit listrik, diukur dalam derajat Celsius.
- ```V (Exhaust Vacuum)``` : Tekanan vakum pada kondensor pembangkit listrik, diukur dalam cm Hg.
- ```AP (Ambient Pressure)``` : Tekanan udara ambien, diukur dalam milibar (mbar).
- ```RH (Relative Humidity)``` : Kelembapan relatif udara ambien, dalam persen (%).
- ```PE (Electrical Power Output)``` : Output daya listrik yang dihasilkan oleh pembangkit listrik, dalam megawatt (MW). Variabel ini merupakan target yang akan diprediksi.

### Visualisasi Distribusi Data
![image](https://github.com/user-attachments/assets/8098f6b2-4d89-4d24-b3bf-782280b4196e)

Distribusi variabel numerik dalam dataset ini memberikan gambaran awal yang esensial untuk memahami karakteristik data sebelum dilakukan pemodelan lebih lanjut. Variabel AT (Ambient Temperature) menunjukkan distribusi bimodal, dengan dua puncak distribusi yang jelas. Hal ini mengindikasikan bahwa suhu lingkungan cenderung berada pada dua rentang dominan, kemungkinan dipengaruhi oleh variasi musim atau kondisi geografis tempat data dikumpulkan. Distribusi ini penting untuk diperhatikan karena dapat berdampak pada stabilitas model jika tidak ditangani dengan teknik transformasi atau segmentasi data.

Pada variabel V (Exhaust Vacuum), distribusi tampak multimodal, memperlihatkan adanya beberapa puncak yang mengindikasikan keberadaan kelompok nilai yang terpisah. Hal ini bisa mencerminkan adanya pola operasional tertentu atau segmentasi beban sistem yang terjadi selama periode pengambilan data. Keberadaan distribusi seperti ini perlu dicermati karena bisa menandakan kebutuhan klasterisasi atau stratifikasi pada saat pelatihan model.

Variabel AP (Ambient Pressure) memiliki bentuk distribusi yang mendekati normal (Gaussian) dengan simetri yang baik di sekitar nilai tengah. Distribusi ini menguntungkan secara statistik karena mendukung berbagai asumsi pada algoritma prediktif seperti regresi linier, dan meminimalkan kebutuhan transformasi lebih lanjut.

Sementara itu, RH (Relative Humidity) memperlihatkan distribusi condong ke kiri (skewed right), dengan sebagian besar nilai berada pada kelembapan tinggi. Ini mencerminkan bahwa lingkungan pengamatan umumnya memiliki kelembapan relatif tinggi, suatu faktor yang mungkin memengaruhi efisiensi energi secara tidak langsung.

Variabel target yaitu PE (Energy Output) juga menunjukkan pola bimodal, menandakan bahwa daya listrik yang dihasilkan oleh sistem pembangkit berada dalam dua kelompok utama. Ini bisa disebabkan oleh perbedaan kondisi operasional, jadwal beban, atau mode kerja sistem (seperti beban puncak vs beban normal). Distribusi ini perlu dipertimbangkan dalam pemilihan algoritma regresi dan teknik validasi karena bisa memengaruhi akurasi dan generalisasi model.

Secara keseluruhan, keberagaman pola distribusi antar fitur ini menunjukkan bahwa dataset memiliki kompleksitas yang cukup tinggi. Hal ini memberikan peluang untuk eksplorasi teknik transformasi data yang tepat guna meningkatkan performa model prediktif yang akan dibangun.

### Visualisasi Beberapa Boxplot dalam Grid Subplot
![image](https://github.com/user-attachments/assets/4a3cb500-88b0-4b79-aea5-975f213f2495)



Distribusi variabel numerik melalui boxplot memberikan wawasan penting tentang persebaran nilai dan potensi keberadaan outlier pada setiap fitur.

Variabel AT (Ambient Temperature) menunjukkan distribusi yang cukup seimbang, dengan nilai tengah (median) berada di sekitar 20°C. Rentang interkuartil (IQR) cukup lebar, mengindikasikan variasi suhu yang signifikan dalam dataset. Meskipun terdapat nilai minimum mendekati 2°C, tidak terlihat adanya outlier ekstrem, menunjukkan persebaran data yang wajar.

Fitur V (Exhaust Vacuum) juga menunjukkan distribusi yang luas, dengan median mendekati 52 dan IQR yang menunjukkan variasi yang cukup besar dalam tekanan vakum. Distribusi ini bersifat simetris tanpa outlier yang signifikan, mengindikasikan kestabilan dalam variasi fitur ini.

Pada fitur AP (Ambient Pressure), terdapat beberapa outlier yang cukup mencolok di bagian bawah dan atas distribusi, meskipun median berada pada sekitar 1013 mbar, mendekati tekanan atmosfer standar. Kehadiran outlier ini perlu diperhatikan karena dapat memengaruhi hasil model regresi jika tidak ditangani dengan benar.

Sementara itu, RH (Relative Humidity) memiliki median yang tinggi, menunjukkan bahwa sebagian besar data berada pada tingkat kelembapan yang tinggi. Outlier ditemukan pada kelembapan rendah (sekitar 30%), yang bisa disebabkan oleh kondisi lingkungan tertentu yang tidak umum. Penyebaran data yang cukup tinggi juga menunjukkan variasi kondisi kelembapan yang signifikan.

Variabel target PE (Energy Output) memiliki persebaran yang simetris dengan median sekitar 450 MW. Distribusi ini relatif stabil tanpa keberadaan outlier ekstrem, menandakan bahwa data target cukup bersih dan cocok untuk digunakan dalam pemodelan regresi. IQR yang cukup besar menunjukkan bahwa daya listrik yang dihasilkan memiliki variasi yang mencerminkan dinamika sistem pembangkit energi.

### Visualisasi Relasi Fitur 
![image](https://github.com/user-attachments/assets/5539935a-19a0-42d3-bbd0-6a0523970267)

Visualisasi tersebut memperlihatkan hubungan antara beberapa variabel seperti suhu lingkungan (AT), tekanan vakum buangan (V), tekanan lingkungan (AP), kelembaban relatif (RH), dan output energi (PE). Dari pola penyebaran data, terlihat bahwa suhu lingkungan (AT) dan tekanan vakum (V) memiliki hubungan negatif yang cukup kuat terhadap output energi (PE). Artinya, semakin tinggi suhu atau vakum buangan, output energi cenderung menurun.

Sementara itu, kelembaban relatif (RH) menunjukkan sedikit hubungan positif dengan output energi, meskipun tidak terlalu kuat. Tekanan lingkungan (AP) tampaknya tidak memiliki hubungan yang signifikan karena penyebaran datanya terlihat acak. Beberapa variabel seperti AT dan PE memiliki pola distribusi data yang bimodal, menunjukkan adanya dua kondisi dominan yang mungkin mewakili situasi operasi yang berbeda.

Secara keseluruhan, suhu dan vakum buangan tampak menjadi faktor yang paling memengaruhi output energi. Hubungan antar fitur ini penting untuk dipahami karena dapat membantu dalam membangun model prediksi yang lebih akurat berdasarkan variabel-variabel yang relevan.

### Visualisasi Correlation Matrix
![image](https://github.com/user-attachments/assets/b7fba119-c105-4258-91bb-fa06240f84c6)

Hubungan antar fitur dalam dataset ini menunjukkan adanya pola keterkaitan yang cukup jelas antara masing-masing variabel. Output energi (PE), yang menjadi target utama dalam analisis, tampak sangat dipengaruhi oleh beberapa fitur lain. Suhu lingkungan (AT) memiliki korelasi negatif yang sangat kuat terhadap PE, dengan nilai korelasi -0.95. Ini menandakan bahwa semakin tinggi suhu, maka output energi yang dihasilkan akan cenderung menurun secara signifikan. Hal ini bisa disebabkan oleh efisiensi sistem yang menurun saat suhu udara meningkat.

Tekanan vakum (V) juga menunjukkan korelasi negatif yang tinggi dengan PE, yakni sebesar -0.87. Ini berarti bahwa peningkatan tekanan vakum cenderung diikuti oleh penurunan output energi. Kemungkinan ini berkaitan dengan pengaruh tekanan terhadap proses pembakaran atau konversi energi. Sementara itu, tekanan udara (AP) dan kelembaban relatif (RH) menunjukkan hubungan positif terhadap PE dengan nilai korelasi masing-masing 0.52 dan 0.39. Artinya, kenaikan tekanan udara dan kelembaban relatif cenderung diikuti oleh peningkatan output energi, meskipun hubungan ini tidak sekuat dua fitur sebelumnya.

Dari sisi hubungan antar fitur selain PE, suhu lingkungan (AT) berkorelasi positif cukup tinggi dengan tekanan vakum (V), yaitu sebesar 0.84. Ini menunjukkan bahwa ketika suhu meningkat, tekanan vakum juga cenderung naik. Namun, suhu memiliki hubungan negatif terhadap tekanan udara (AP) dan kelembaban (RH), yang menunjukkan bahwa suhu tinggi biasanya disertai dengan tekanan udara yang lebih rendah dan kelembaban yang lebih rendah.

Tekanan udara (AP) dan kelembaban (RH) sendiri hanya memiliki hubungan yang sangat lemah, hampir tidak berkorelasi (0.10), menunjukkan bahwa keduanya merupakan variabel yang relatif independen satu sama lain dalam konteks ini. Demikian pula, tekanan udara dan tekanan vakum memiliki korelasi negatif sedang, sebesar -0.41, menunjukkan adanya tren berlawanan antara keduanya.

Secara keseluruhan, informasi ini memberikan wawasan penting tentang faktor-faktor yang memengaruhi output energi. Suhu dan tekanan vakum tampaknya menjadi indikator paling signifikan karena pengaruhnya yang kuat terhadap penurunan performa sistem, sedangkan tekanan udara dan kelembaban relatif memberikan kontribusi positif yang dapat mendukung peningkatan efisiensi dalam kondisi tertentu. Analisis ini dapat menjadi dasar dalam pengambilan keputusan atau pengembangan model prediktif yang lebih akurat.

## Data Preparation
-  Penghapusan Duplikasi Data
```python
# Remove duplicate rows from the DataFrame 'data' to ensure data quality and avoid bias
data = data.drop_duplicates()

# Check again for any remaining duplicate rows after removal
number_of_duplicates = data.duplicated().sum()

# Print the number of duplicates found after dropping duplicates, expected to be zero
print(f"Number of duplicates after removal: {number_of_duplicates}")
```
Pada tahap ini, dilakukan penghapusan baris-baris duplikat dalam dataset menggunakan fungsi drop_duplicates(). Duplikasi data dapat menyebabkan bias dalam model karena model akan "menghitung ulang" informasi yang sama berulang kali. Setelah penghapusan, dicek kembali jumlah baris duplikat yang tersisa, yang diharapkan menjadi nol. Langkah ini penting untuk memastikan kualitas data yang digunakan untuk pelatihan model tetap terjaga dan tidak memberikan bobot berlebih pada data yang sama.

- Penanganan Outlier dengan Metode IQR dan Deteksi Outlier
```python
def iqr_outliers(data):
    outliers = pd.DataFrame()
    for col in data.select_dtypes(include=[np.number]).columns:
        Q1 = data[col].quantile(0.25)
        Q3 = data[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers[col] = ((data[col] < lower_bound) | (data[col] > upper_bound)).astype(int)
    return outliers

def cap_outliers(data, col):
    data = data.copy()  
    Q1 = data[col].quantile(0.25)
    Q3 = data[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    data[col] = np.where(data[col] < lower_bound, lower_bound, data[col])
    data[col] = np.where(data[col] > upper_bound, upper_bound, data[col])
    return data

data = cap_outliers(data, 'AP')
data = cap_outliers(data, 'RH')

print("Outliers after capping (IQR):")
print(iqr_outliers(data).sum())
```
Metode IQR (Interquartile Range) memanfaatkan rentang antara kuartil pertama (Q1) dan kuartil ketiga (Q3) untuk mengidentifikasi nilai ekstrem. IQR sendiri dihitung sebagai selisih Q3–Q1. Batas bawah (lower bound) dan batas atas (upper bound) kemudian ditetapkan pada Q1–1,5·IQR dan Q3+1,5·IQR. Nilai yang jatuh di bawah atau di atas kedua batas ini dianggap sebagai outlier.

Untuk mendeteksi outlier, setiap data diuji apakah berada di luar rentang tersebut. Hasilnya biasanya berupa penandaan biner (1 untuk outlier, 0 untuk normal), sehingga kita dapat menghitung jumlah atau persentase outlier pada setiap fitur.

Sementara itu, penanganan outlier (capping) berarti mengganti nilai-nilai ekstrem ini dengan nilai batas terdekat. Jika sebuah pengukuran lebih rendah dari batas bawah, ia “dipotong” menjadi sama dengan batas bawah; begitu pula, nilai di atas batas atas digantikan batas atas. Dengan demikian, distribusi data menjadi lebih terkendali—ekor distribusi dipersingkat—tanpa menghilangkan baris data secara keseluruhan.

Pendekatan ini menjaga integritas dataset (tidak ada baris hilang) sekaligus melindungi analisis atau model machine learning dari efek merugikan nilai ekstrem. Secara natural, prosesnya terdiri dari dua langkah sederhana: pertama, identifikasi outlier menggunakan perhitungan IQR, lalu tangani outlier tersebut dengan mengganti nilainya agar tetap berada di dalam rentang yang wajar.

- Seleksi Fitur dan Target untuk Model
```python
# Select the feature columns from the dataset to form the input matrix X
# The features include 'AT' (Ambient Temperature), 'V' (Exhaust Vacuum), 'AP' (Ambient Pressure), and 'RH' (Relative Humidity)
X = data[['AT', 'V', 'AP', 'RH']]

# Select the target column 'PE' (Power Output) which is the variable to be predicted
y = data['PE']

# This separation into features (X) and target (y) is essential for supervised learning tasks
# where the model learns to predict y based on the input features X
```
Pada tahap ini, dataset dipisahkan menjadi dua bagian utama: fitur (input) dan target (output). Fitur yang dipilih adalah kolom 'AT' (Temperatur), 'V' (Vacuum), 'AP' (Tekanan), dan 'RH' (Kelembapan), yang berfungsi sebagai variabel prediktor. Sedangkan kolom 'PE' (Power Output) dipilih sebagai target yang akan diprediksi oleh model. Proses pemisahan ini merupakan tahap penting dalam supervised learning agar model dapat belajar memetakan hubungan dari fitur ke target.

- Pembagian Dataset menjadi Data Latih dan Data Uji
```python
# Split the dataset into training and testing sets
# 'test_size=0.2' means 20% of the data is allocated for testing, and 80% for training
# 'random_state=42' ensures reproducibility by fixing the random seed
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Print the shapes of the resulting datasets to verify the split
print("X_train shape:", X_train.shape)  # Number of training samples and features
print("X_test shape:", X_test.shape)    # Number of testing samples and features
print("y_train shape:", y_train.shape)  # Number of training labels
print("y_test shape:", y_test.shape)    # Number of testing labels
```
Tahap terakhir ini membagi data menjadi data latih dan data uji menggunakan fungsi train_test_split. Proporsi yang digunakan adalah 80% data untuk pelatihan dan 20% untuk pengujian. Parameter random_state=42 digunakan agar pembagian data bersifat deterministik dan dapat direproduksi pada eksekusi ulang. Pembagian ini penting agar model dapat dilatih dengan data latih dan kemudian diuji performanya dengan data uji yang belum pernah dilihat sebelumnya, guna mengukur kemampuan generalisasi model secara objektif.


## Modeling
Pada tahap ini dibangun model machine learning untuk memprediksi output daya pada Combined Cycle Power Plants (CCPP) berdasarkan data yang telah melalui tahap preprocessing. Karena variabel target bersifat kontinu, pendekatan yang digunakan adalah regresi. Tiga model regresi yang akan dieksplorasi adalah:
- ```Random Forest: ``` Model Random Forest merupakan ensemble dari banyak pohon keputusan (decision trees) yang dilatih pada subset data dan subset fitur secara acak. Hasil prediksi diperoleh dari rata-rata (regresi) output setiap pohon. Keunggulannya meliputi kemampuan menangkap pola non-linear, robust terhadap outlier, dan minim risiko overfitting berkat agregasi pohon. Selain itu, Random Forest relatif mudah dituning (hanya beberapa parameter utama seperti jumlah pohon dan kedalaman maksimum). Namun, model ini bisa menjadi lambat pada dataset besar dan interpretasinya lebih sulit dibanding model linear.
- ```Gradient Boosting:``` Model Gradient Boosting adalah ensemble model yang membangun banyak pohon keputusan secara bertahap, di mana setiap pohon baru bertugas memperbaiki kesalahan model sebelumnya dengan mengoptimalkan fungsi loss menggunakan gradient descent. Model ini mampu menangkap pola non-linear dengan baik dan menghasilkan performa tinggi jika parameter dituning secara tepat. Namun, pelatihan model ini lebih memakan waktu, rawan overfitting tanpa tuning yang baik, dan proses tuning parameternya cukup kompleks.
- ```XGBoost:``` XGBoost (eXtreme Gradient Boosting) adalah varian Gradient Boosting yang dioptimalkan untuk kecepatan dan performa dengan tambahan regularisasi L1/L2, pemrosesan terdistribusi, serta handling missing values secara otomatis. XGBoost sering kali menghasilkan akurasi tinggi di berbagai kompetisi data science, mampu menangani dataset besar dengan efisien, dan menyediakan kontrol lebih detail atas penalti model. Kekurangannya termasuk kompleksitas setup parameter yang lebih banyak dan kebutuhan sumber daya komputasi lebih tinggi dibanding Random Forest.
Tahapan pembuatan model yang dilakukan adalah sebagai berikut:

1. ```Inisialisasi dan Pelatihan Model:```
   
Pada tahap ini, dilakukan inisialisasi dan pelatihan dua model regresi yaitu Linear Regression dan Gradient Boosting. Masing-masing model dikonfigurasi dengan parameter tertentu yang telah disesuaikan untuk meningkatkan performa model terhadap data yang digunakan.
   - Random Forest
     ```
     # Initialize the model
      rf_model = RandomForestRegressor(n_estimators=200, max_depth=6, random_state=42)
      # Training model
      rf_model.fit(X_train, y_train)
      ```
    
     Penjelasan:
      - Baris pertama menginisialisasi model Random Forest dengan 200 pohon keputusan (n_estimators=200), kedalaman maksimal pohon 6 (max_depth=6), dan menetapkan random_state=42 untuk menghasilkan hasil yang konsisten saat pelatihan diulang.
      - Baris kedua menjalankan proses pelatihan model (fit) menggunakan data fitur X_train dan target y_train, sehingga model belajar mengenali pola hubungan antara input dan output.
     
  - Gradient Boosting Regressor 
      ```
      # Initialize the model
      gb_model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, max_depth=6, random_state=42)
      # Training model
      gb_model.fit(X_train, y_train)
    
      ```
      Penjelasan:
      - GradientBoostingRegressor adalah model ensemble berbasis decision tree yang menggabungkan beberapa pohon secara bertahap untuk memperbaiki kesalahan prediksi.
      - Parameter n_estimators, learning_rate, dan max_depth telah disesuaikan (tuning) untuk mendapatkan performa terbaik.
   
    - XGBoost
      ```
      # Initialize the model
      xgb_model = XGBRegressor(n_estimators=200, learning_rate=0.1, max_depth=6, random_state=42, objective='reg:squarederror')
      # Training model
      xgb_model.fit(X_train, y_train)
      ```
      Penjelasan:
      - Baris pertama menginisialisasi model XGBoost untuk regresi dengan 200 pohon (n_estimators=200), laju pembelajaran (learning_rate=0.1), kedalaman maksimal pohon 6 (max_depth=6), dan random_state=42 agar hasil pelatihan konsisten. Parameter objective='reg:squarederror' digunakan untuk menetapkan fungsi kerugian kuadrat yang umum untuk regresi.
      - Baris kedua melatih model menggunakan data fitur X_train dan target y_train, sehingga model belajar memetakan pola antara input dan output untuk membuat prediksi yang akurat.
        
2. ```Evaluasi Model:```

Evaluasi model dilakukan dengan menggunakan metrik MAE, RMSE, dan R² Score pada data testing serta data training untuk melihat performa dan potensi overfitting.

| Model            | MAE Test | RMSE Test | R² Test | MAE Train | RMSE Train | R² Train |
|------------------|----------|-----------|---------|-----------|------------|----------|
| Random Forest    | 3.0490   | 4.0468    | 0.9443  | 2.8856    | 3.7406     | 0.9517   |
| Gradient Boosting| 3.0063   | 3.9872    | 0.9459  | 2.8130    | 3.6771     | 0.9533   |
| XGBoost          | 2.3495   | 3.3274    | 0.9623  | 1.5999    | 2.1658     | 0.9838   |

3. ```Analisis dan Pemilihan Model Terbaik```
Berdasarkan evaluasi awal, XGBoost Regressor dipilih sebagai model terbaik karena menunjukkan performa paling unggul dengan nilai MAE dan RMSE terendah serta nilai R² tertinggi (0.9623). Ini berarti XGBoost mampu memberikan prediksi yang lebih akurat dan menjelaskan variansi data dengan baik dibandingkan model Random Forest dan Gradient Boosting standar. Selain itu, XGBoost memiliki fitur regularisasi dan efisiensi komputasi yang memungkinkan pengembangan model lebih optimal melalui proses tuning hyperparameter.
Untuk meningkatkan performa model lebih jauh, dilakukan tuning hyperparameter menggunakan metode RandomizedSearchCV pada kedua model Gradient Boosting dan XGBoost. Proses ini bertujuan menemukan kombinasi parameter terbaik yang meminimalkan kesalahan prediksi.

- Gradient Boosting mendapatkan parameter terbaik seperti n_estimators=300, max_depth=7, dan learning_rate=0.1 dengan hasil evaluasi setelah tuning menunjukkan peningkatan: MAE turun menjadi 2.2051, RMSE menjadi 3.1923, dan R² naik ke 0.9653.

- XGBoost juga melakukan tuning dengan parameter seperti n_estimators=500, max_depth=7, learning_rate=0.05, dan regularisasi reg_alpha serta reg_lambda. Hasilnya MAE menjadi 2.2174, RMSE 3.2176, dan R² 0.9648, juga menunjukkan perbaikan signifikan dari model awal.
  
| Model                    | MAE      | RMSE     | R²       |
|------------------------- | -------- | -------- | -------- |
| Random Forest (Tuned)    | 2.290675 | 3.316191 | 0.962581 |
| Gradient Boosting (Tuned)| 2.205147 | 3.192256 | 0.965326 |
| XGBoost (Tuned)          | 2.217422 | 3.217562 | 0.964774 |

Gradient Boosting dipilih sebagai model terbaik setelah tuning karena performanya menunjukkan peningkatan yang paling signifikan dengan nilai MAE dan RMSE terendah serta R² tertinggi. Hal ini menandakan model ini mampu menangkap pola dan variansi data dengan sangat baik, menghasilkan prediksi yang lebih akurat dan konsisten. Proses tuning berhasil menemukan kombinasi hyperparameter yang optimal sehingga model tidak hanya lebih tepat tetapi juga lebih stabil dalam memprediksi data baru.

Sementara itu, meskipun XGBoost sebelum tuning tampil sangat kuat, setelah proses tuning performanya sedikit menurun dibandingkan hasil awal dan juga sedikit kalah dibanding Gradient Boosting yang sudah dituning. Penurunan ini bisa terjadi karena kombinasi hyperparameter yang dipilih oleh RandomizedSearchCV mungkin kurang ideal untuk dataset ini, sehingga model mengalami sedikit overfitting atau underfitting. Selain itu, XGBoost memiliki banyak hyperparameter kompleks yang perlu penyesuaian sangat tepat, dan jika tuning tidak optimal, performanya bisa sedikit menurun dibandingkan konfigurasi awal yang sederhana namun efektif.

## Evaluation

Setelah proses pelatihan model selesai, tahap selanjutnya adalah melakukan evaluasi terhadap performa masing-masing model yang telah dibangun. Evaluasi dilakukan dengan tujuan untuk menilai seberapa baik model dalam memprediksi target, yaitu output energi CCPP, berdasarkan data uji yang belum pernah dilihat oleh model sebelumnya. Karena permasalahan yang diangkat merupakan regresi, maka digunakan beberapa metrik yang umum dalam regresi, yaitu Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), dan R² Score. Masing-masing metrik ini memberikan sudut pandang yang berbeda dalam menilai akurasi prediksi model, baik dari segi rata-rata kesalahan, sensitivitas terhadap outlier, maupun proporsi variansi yang berhasil dijelaskan oleh model.
### Metrik Evaluasi
1. **`Mean Absolute Error (MAE)`**: MAE mengukur rata-rata absolut dari selisih antara nilai aktual dan prediksi.

    <img src="https://latex.codecogs.com/svg.image?\dpi{120}&space;\bg{transparent}&space;\color{White}&space;\text{MAE}&space;=&space;\frac{1}{n}&space;\sum_{i=1}^{n}&space;\left|y_i&space;-&space;\hat{y}_i\right|">
   
    Metrik ini memberikan gambaran rata-rata kesalahan model tanpa memperhatikan arah kesalahan (positif atau negatif), sehingga mudah diinterpretasikan. Semakin kecil nilai MAE, semakin akurat prediksi model terhadap data aktual.
   
    ```python
   # Random Forest MAE
    mae_rf = mean_absolute_error(y_test, y_pred_rf)
    
    # Gradient Boosting MAE
    mae_gb = mean_absolute_error(y_test, y_pred_gb)
    
    # XGBoost MAE
    mae_xgb = mean_absolute_error(y_test, y_pred_xgb)
    ```

    Cara kerja: MAE menghitung seberapa besar kesalahan rata-rata prediksi model terhadap data aktual, tanpa mempertimbangkan arah kesalahan.

2. **`Root Mean Squared Error (RMSE)`**: RMSE mengukur akar dari rata-rata kuadrat selisih antara nilai aktual dan prediksi.

     <img src="https://latex.codecogs.com/svg.image?\dpi{120}&space;\bg{transparent}&space;\color{White}&space;\text{RMSE}&space;=&space;\sqrt{&space;\frac{1}{n}&space;\sum_{i=1}^{n}&space;(y_i&space;-&space;\hat{y}_i)^2&space;}">
        
      RMSE lebih sensitif terhadap _outlier_ dibanding MAE karena kesalahan dikuadratkan. Nilai lebih kecil menunjukkan prediksi lebih akurat secara keseluruhan. Oleh karena itu, RMSE sangat berguna untuk mendeteksi model yang sensitif terhadap _outlier_.
      ```python
      # Random Forest RMSE
    rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
    
    # Gradient Boosting RMSE
    rmse_gb = np.sqrt(mean_squared_error(y_test, y_pred_gb))
    
    # XGBoost RMSE
    rmse_xgb = np.sqrt(mean_squared_error(y_test, y_pred_xgb))

      ```
                
      Cara kerja: RMSE menghitung rata-rata kuadrat dari kesalahan prediksi, kemudian diakarkan untuk mendapatkan satuan yang sama dengan target. Semakin besar kesalahan, semakin tinggi nilainya.

3. **`R² Score (Coefficient of Determination)`**: R² mengukur seberapa besar variansi dari data target yang dapat dijelaskan oleh model.

     <img src="https://latex.codecogs.com/svg.image?\dpi{120}&space;\bg{transparent}&space;\color{White}&space;R^2&space;=&space;1&space;-&space;\frac{&space;\sum_{i=1}^{n}&space;(y_i&space;-&space;\hat{y}_i)^2&space;}{&space;\sum_{i=1}^{n}&space;(y_i&space;-&space;\bar{y})^2&space;}" alt="R Squared Equation">

     Nilai R² berkisar antara 0 hingga 1, di mana nilai yang lebih tinggi menunjukkan bahwa model mampu menjelaskan variabilitas data target dengan lebih baik. Jika R² mendekati 1, berarti model hampir sepenuhnya mampu menjelaskan variasi dalam data.
        
    ```python
        # Random Forest R² Score
    r2_rf = r2_score(y_test, y_pred_rf)
    
    # Gradient Boosting R² Score
    r2_gb = r2_score(y_test, y_pred_gb)
    
    # XGBoost R² Score
    r2_xgb = r2_score(y_test, y_pred_xgb)

     ```
    Cara kerja: R² membandingkan antara total kesalahan model dengan kesalahan baseline (rata-rata nilai aktual). Nilai 1 berarti prediksi sempurna, 0 berarti tidak lebih baik dari sekadar menebak rata-rata.

### Evaluasi model akhir dengan menggunakan _dataset test_

1. **`Random Forest Regressor`**
   
| Model                    | MAE      | RMSE     | R²       |
|--------------------------|----------|----------|----------|
| Random Forest (Tuned)    | 2.2907   | 3.3162   | 0.9626   |

    Model Random Forest (Tuned) menunjukkan performa yang sangat baik pada data *test*, dengan nilai R² sebesar **0.9626** yang menandakan bahwa model mampu menjelaskan sekitar **96.26%** variasi dalam data target. Nilai **MAE sebesar 2.2907** menunjukkan bahwa, secara rata-rata, prediksi model memiliki selisih sekitar 2.29 poin dari nilai aktual. Sementara itu, **RMSE sebesar 3.3162** mengindikasikan bahwa model memiliki kesalahan prediksi rata-rata sekitar 3.31 poin, dengan penekanan lebih besar terhadap kesalahan yang ekstrem.
    Secara keseluruhan, model ini cukup andal dalam memprediksi nilai target, meskipun masih terdapat beberapa deviasi pada beberapa titik data.


![image](https://github.com/user-attachments/assets/2eb1b2f4-773f-4ea2-a758-1a5b94675f1c)

    Grafik ini menunjukkan hubungan antara nilai aktual dan prediksi model Random Forest. Titik-titik banyak tersebar dekat dengan garis merah putus-putus (garis identitas), menandakan bahwa prediksi model cukup akurat dengan deviasi kecil.

    
 ![image](https://github.com/user-attachments/assets/18a8180f-e1ae-452f-ac36-46848c4bcf06)

    
    Plot residual menampilkan sebaran kesalahan prediksi terhadap nilai prediksi. Titik-titik residual tersebar acak di sekitar garis nol tanpa pola tertentu, menunjukkan bahwa model tidak memiliki bias sistematis.
    
![image](https://github.com/user-attachments/assets/2ed9b85b-2d26-4f28-9dbf-e365e2747088)

    
    Histogram residual menunjukkan bentuk distribusi yang mendekati normal dan simetris, dengan puncak di sekitar nol. Ini menunjukkan bahwa sebagian besar kesalahan prediksi model bersifat kecil dan acak.

2. **`Gradient Boosting`**
   
| Model                    | MAE      | RMSE     | R²       |
|------------------------- | -------- | -------- | -------- |
| Gradient Boosting (Tuned)| 2.205147 | 3.192256 | 0.965326 |

Model Gradient Boosting (Tuned) menunjukkan performa prediksi yang sangat baik dengan **R² sebesar 0.9653**, yang berarti model mampu menjelaskan sekitar **96.53%** variasi dalam data target. Nilai **MAE sebesar 2.2051** menunjukkan bahwa secara rata-rata prediksi model berbeda sekitar 2.21 poin dari nilai aktual. Sementara itu, **RMSE sebesar 3.1923** menunjukkan bahwa kesalahan prediksi model tetap rendah dan relatif stabil, meskipun masih ada beberapa titik data dengan deviasi yang lebih besar.

Secara keseluruhan, Gradient Boosting memberikan performa yang sangat baik dan mampu menangkap pola hubungan antara fitur dan target dengan akurasi tinggi, bahkan sedikit lebih baik dibanding Random Forest pada metrik yang sama.
    
 ![image](https://github.com/user-attachments/assets/7fe8d97d-010c-4a50-98ee-3851f8658538)

    
    Grafik ini memperlihatkan korelasi antara nilai aktual dan hasil prediksi model Gradient Boosting. Sebagian besar titik berada sangat dekat dengan garis identitas, menandakan bahwa prediksi model sangat akurat dan mendekati nilai sebenarnya. Jika dibandingkan dengan model lain seperti Random Forest, prediksi Gradient Boosting tampak lebih konsisten dan presisi.

![image](https://github.com/user-attachments/assets/4c01c085-0f2d-413e-ae77-c71802e32531)

    Sebaran residual dari model Gradient Boosting menunjukkan pola yang acak dan simetris di sekitar garis nol. Tidak terdapat tren atau pola tertentu, mengindikasikan bahwa kesalahan prediksi bersifat stabil dan tidak bias. Jika dibandingkan dengan model lain, Gradient Boosting menunjukkan kestabilan yang lebih tinggi dengan penyebaran residual yang lebih terkontrol.
    
![image](https://github.com/user-attachments/assets/b75e088b-f49a-4e06-b1f4-0f4ec6428fa4)

    
    Histogram residual dari model Gradient Boosting menunjukkan bentuk distribusi yang mendekati distribusi normal, dengan puncak di sekitar nol. Hal ini memperkuat bukti bahwa kesalahan prediksi model bersifat acak dan tidak condong ke satu sisi. Jika dibandingkan dengan model lain, distribusi residual Gradient Boosting terlihat lebih terkonsentrasi dan simetris.


3. **`XGBoost Regressor`**
   
| Model                    | MAE      | RMSE     | R²       |
|------------------------- | -------- | -------- | -------- |
| XGBoost (Tuned)          | 2.217422 | 3.217562 | 0.964774 |

Model XGBoost yang telah dituning menunjukkan performa yang sangat baik dengan nilai **R² sebesar 0.9648**, artinya model mampu menjelaskan sekitar **96.48%** variasi dari data target. Nilai **MAE sebesar 2.2174** menunjukkan bahwa secara rata-rata, prediksi model meleset sekitar 2.22 poin dari nilai aktual. Sementara itu, **RMSE sebesar 3.2176** mengindikasikan bahwa prediksi model cukup stabil, meskipun tetap memperhatikan adanya kesalahan besar yang mungkin terjadi.

Secara keseluruhan, performa XGBoost hampir setara dengan Gradient Boosting dan Random Forest. Meskipun secara metrik sedikit di bawah Gradient Boosting, model ini tetap memberikan hasil yang sangat kompetitif dan presisi yang tinggi dalam menangkap hubungan antara fitur dan target.
     
![image](https://github.com/user-attachments/assets/eff7265d-97a9-4a93-b133-4b0e9e656b93)

    
    Grafik ini menunjukkan hubungan antara nilai aktual dan hasil prediksi model XGBoost yang telah dituning. Titik-titik data tersebar rapat di sepanjang garis identitas (garis merah putus-putus), menandakan bahwa prediksi model sangat mendekati nilai sebenarnya. Hal ini menunjukkan performa prediksi yang sangat baik, dengan galat yang relatif kecil dan distribusi prediksi yang konsisten di seluruh rentang nilai.
    
![image](https://github.com/user-attachments/assets/695df019-10d3-4fc9-9365-49b720f1121e)

    
    Sebaran residual terhadap nilai prediksi terlihat acak dan simetris di sekitar garis nol. Tidak tampak adanya pola sistematis seperti tren menaik/menurun, yang berarti kesalahan prediksi bersifat tidak bias (unbiased). Ini menandakan bahwa model tidak overfitting maupun underfitting terhadap data, serta kesalahan tersebar merata di seluruh rentang nilai prediksi.
    
![image](https://github.com/user-attachments/assets/4fb11825-81d8-443f-977d-8f45ae457f03)

    
    Histogram residual menunjukkan bentuk distribusi yang menyerupai distribusi normal dengan puncak di sekitar nol. Ini mengindikasikan bahwa sebagian besar prediksi model memiliki kesalahan kecil, dan hanya sedikit outlier. Distribusi ini memperkuat asumsi bahwa model memiliki stabilitas dan akurasi yang baik, dengan galat yang cenderung acak.

## Conclusion
![image](https://github.com/user-attachments/assets/ccdadb7e-9796-4f35-b30b-59d1341814a6)
![image](https://github.com/user-attachments/assets/3efb673b-54a9-4bf1-82a5-38aeb0c13b7f)
Dalam proyek ini, dilakukan serangkaian tahapan mulai dari pemahaman data (data understanding), praproses data (data preprocessing), pemilihan fitur (feature selection), hingga pelatihan dan evaluasi model menggunakan tiga algoritma machine learning regresi terkemuka: Random Forest Regressor, Gradient Boosting Regressor, dan XGBoost Regressor. Tujuan utama dari proyek ini adalah membangun model prediktif yang akurat untuk memperkirakan nilai target berdasarkan fitur input yang tersedia.

Evaluasi performa model dilakukan menggunakan tiga metrik utama, yaitu MAE (Mean Absolute Error), RMSE (Root Mean Squared Error), dan R² Score. Hasil evaluasi menunjukkan bahwa:

- Gradient Boosting Regressor (Tuned) mencatat performa terbaik dengan nilai MAE terendah sebesar 2.2051, RMSE sebesar 3.1923, dan R² sebesar 0.9653. Hal ini menunjukkan bahwa model mampu meminimalkan galat prediksi dengan stabilitas tinggi serta menangkap variasi data target dengan akurasi lebih dari 96.5%. Visualisasi residual juga menunjukkan distribusi yang simetris dan tidak bias.

- XGBoost Regressor (Tuned) menempati posisi kedua dengan MAE sebesar 2.2174, RMSE sebesar 3.2176, dan R² sebesar 0.9648. Performa model ini hampir setara dengan Gradient Boosting, dengan kelebihan dalam kestabilan prediksi serta distribusi residual yang tidak menunjukkan pola sistematis.

- Random Forest Regressor (Tuned) juga menunjukkan performa sangat baik dengan R² sebesar 0.9626, MAE sebesar 2.2907, dan RMSE sebesar 3.3162. Meskipun berada sedikit di bawah dua model lainnya dari sisi metrik, model ini tetap akurat dan andal, dengan distribusi residual yang acak dan tidak menunjukkan bias yang berarti.

Berdasarkan hasil evaluasi metrik dan analisis visual residual dari ketiga model, Gradient Boosting Regressor dipilih sebagai model terbaik dalam studi ini. Model ini tidak hanya memberikan akurasi tertinggi, tetapi juga menunjukkan stabilitas prediksi yang baik, distribusi error yang sehat, serta generalisasi yang optimal terhadap data uji. Oleh karena itu, Gradient Boosting direkomendasikan untuk digunakan dalam implementasi prediksi pada domain permasalahan ini.
