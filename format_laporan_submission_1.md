# Laporan Proyek Machine Learning - Nama Anda

## Domain Proyek
Menurut International Energy Agency (IEA), sektor energi listrik merupakan tulang punggung pembangunan ekonomi modern dan bertanggung jawab atas lebih dari 40% emisi karbon global, sehingga pengoptimalan efisiensi pembangkit listrik sangat krusial dalam upaya mitigasi perubahan iklim dan pencapaian target net-zero emissions International Energy Agency. Combined Cycle Power Plants (CCPP) merupakan teknologi pembangkit yang menggabungkan turbin gas dan turbin uap untuk meningkatkan efisiensi termal hingga mencapai 60% atau lebih, jauh melampaui pembangkit berbahan bakar fosil konvensional Zhou et al.. Efisiensi tinggi ini tidak hanya mengurangi konsumsi bahan bakar tetapi juga menurunkan emisi gas rumah kaca, menjadikan CCPP sebagai salah satu solusi utama dalam transisi energi saat ini IEA.

Namun, performa CCPP sangat dipengaruhi oleh kondisi lingkungan yang dinamis seperti suhu udara, kelembaban, tekanan atmosfer, dan kecepatan angin, yang dapat berfluktuasi secara signifikan sepanjang hari dan musim. Penurunan suhu sekitar misalnya, dapat meningkatkan efisiensi pembangkit, sementara suhu tinggi dapat menurunkannya secara drastis, sehingga ketidakpastian ini menyebabkan variabilitas output daya yang cukup besar Smith dan Lee. Studi empiris menunjukkan bahwa suhu tinggi dapat menurunkan output daya CCPP hingga 10-15% dibanding kondisi optimal, yang berdampak langsung pada kestabilan pasokan listrik Kumar et al..

Ketidakpastian output daya ini menjadi tantangan besar bagi pengelolaan sistem kelistrikan, terutama dalam konteks integrasi pembangkit energi terbarukan yang juga memiliki variabilitas tinggi. Fluktuasi output daya tanpa prediksi yang akurat dapat menyebabkan gangguan keseimbangan beban dan menurunkan keandalan jaringan listrik, bahkan berpotensi menimbulkan blackout jika tidak ditangani dengan baik Zhang et al.. Selain itu, operasi pembangkit yang tidak optimal akibat kurangnya prediksi lingkungan berpotensi mempercepat kerusakan peralatan, meningkatkan biaya pemeliharaan, dan menurunkan umur teknis aset Miller dan Thompson.

Dalam upaya menghadapi tantangan tersebut, pengembangan model prediksi output daya berbasis analisis data lingkungan menjadi sangat penting. Metode prediksi yang mampu mengakomodasi data lingkungan secara real-time dan menghasilkan estimasi yang presisi dapat mendukung perencanaan operasional yang adaptif dan pengambilan keputusan yang lebih tepat oleh operator pembangkit International Renewable Energy Agency. Dengan demikian, risiko gangguan pasokan dapat diminimalisasi dan efisiensi pembangkit dapat dioptimalkan secara berkelanjutan.

Lebih lanjut, implementasi teknologi digital dan analitik data dalam sektor energi sejalan dengan konsep Smart Grid yang mengintegrasikan sumber energi terbarukan dan sistem penyimpanan energi dengan jaringan listrik secara efisien. Data historis dan real-time dari sensor lingkungan di area pembangkit menyediakan peluang besar untuk pengembangan model prediktif yang akurat dan responsif Gonzalez et al.. Oleh karena itu, pemanfaatan data lingkungan secara komprehensif bukan hanya solusi teknis, tetapi juga bagian dari strategi transisi energi menuju sistem kelistrikan yang lebih hijau, handal, dan berkelanjutan.

Proyek ini berfokus pada pengembangan model prediksi output daya CCPP dengan memanfaatkan data variabel lingkungan yang aktual dan historis. Diharapkan hasil penelitian ini dapat memberikan kontribusi signifikan terhadap pengelolaan pembangkit yang lebih efisien dan berkelanjutan, sekaligus mendukung kebijakan energi nasional dalam mencapai target pengurangan emisi karbon.

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
![image](https://github.com/user-attachments/assets/d91386a1-c971-483a-b2c8-6fa01499cb69)

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
# Function to cap outliers in a specific column using the IQR method
def cap_outliers(data, col):
    Q1 = data[col].quantile(0.25)
    Q3 = data[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    data[col] = np.where(data[col] < lower_bound, lower_bound, data[col])
    data[col] = np.where(data[col] > upper_bound, upper_bound, data[col])
    return data

# Function to detect outliers using Z-score method
def zscore_outliers(data, threshold=3):
    outliers = pd.DataFrame()
    for col in data.select_dtypes(include=[np.number]).columns:
        z_scores = zscore(data[col])
        outliers[col] = np.where(np.abs(z_scores) > threshold, 1, 0)
    return outliers

# Function to detect outliers using IQR method
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

# Apply outlier capping directly on the original data for 'AP' and 'RH' columns
data = cap_outliers(data, 'AP')
data = cap_outliers(data, 'RH')

# Display the number of outliers detected after capping using both methods
print("Outliers after capping (Z-score):")
print(zscore_outliers(data).sum())

print("\nOutliers after capping (IQR):")
print(iqr_outliers(data).sum())
```
Tahap ini mengatasi masalah outlier pada data, yaitu nilai ekstrim yang dapat memengaruhi kinerja model secara negatif. Fungsi cap_outliers menggunakan metode Interquartile Range (IQR) untuk membatasi nilai outlier pada kolom tertentu, yaitu 'AP' (Ambient Pressure) dan 'RH' (Relative Humidity), dengan cara menggantikan nilai di luar batas bawah dan atas dengan batas tersebut. Selain itu, disediakan fungsi deteksi outlier menggunakan dua metode populer, yaitu Z-score dan IQR, untuk mengevaluasi jumlah outlier yang tersisa setelah penanganan. Penanganan outlier penting untuk menjaga kestabilan model dan mencegah pengaruh nilai ekstrim yang bisa menyebabkan model overfitting atau kesalahan prediksi.

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
Pada tahap ini dilakukan pembangunan model machine learning untuk memprediksi output daya pada Combined Cycle Power Plants (CCPP) berdasarkan data yang telah melalui tahap preprocessing. Karena variabel target bersifat kontinu, maka pendekatan yang digunakan adalah regresi.
Dua model regresi yang digunakan dalam proses pemodelan ini adalah:
- ```Linear Regression: ``` Model Linear Regression merupakan algoritma dasar untuk regresi yang memodelkan hubungan linier antara fitur dan target. Model ini sederhana, cepat dalam proses pelatihan, dan mudah diinterpretasikan karena koefisiennya menjelaskan pengaruh masing-masing fitur. Namun, model ini kurang efektif pada data dengan pola non-linear, rentan terhadap outlier, dan tidak secara otomatis mengelola interaksi antar fitur.
- ```Gradient Boosting:``` Model Gradient Boosting adalah ensemble model yang membangun banyak pohon keputusan secara bertahap, di mana setiap pohon baru bertugas memperbaiki kesalahan model sebelumnya dengan mengoptimalkan fungsi loss menggunakan gradient descent. Model ini mampu menangkap pola non-linear dengan baik dan menghasilkan performa tinggi jika parameter dituning secara tepat. Namun, pelatihan model ini lebih memakan waktu, rawan overfitting tanpa tuning yang baik, dan proses tuning parameternya cukup kompleks.
Tahapan pembuatan model yang dilakukan adalah sebagai berikut:
1. ```Inisialisasi dan Pelatihan Model:```
   
Pada tahap ini, dilakukan inisialisasi dan pelatihan dua model regresi yaitu Linear Regression dan Gradient Boosting. Masing-masing model dikonfigurasi dengan parameter tertentu yang telah disesuaikan untuk meningkatkan performa model terhadap data yang digunakan.
   - Linear Regression
     ```
     # Inisialisasi model Linear Regression
      lr_model = LinearRegression()

      # Melatih model menggunakan data training
      lr_model.fit(X_train, y_train)
      ```
     Penjelasan:
      - LinearRegression() digunakan untuk membangun model regresi linear sederhana yang berusaha memetakan hubungan linier antara fitur (X_train) dan target (y_train).
      - fit(X_train, y_train) digunakan untuk melatih model berdasarkan data pelatihan.
     
    - Gradient Boosting Regressor (Tuned)
      ```
      # Inisialisasi model Gradient Boosting dengan parameter yang telah dituning
      gb_model = GradientBoostingRegressor(
          n_estimators=200,      # Jumlah pohon yang digunakan
          learning_rate=0.1,     # Seberapa besar kontribusi tiap pohon
          max_depth=6,           # Kedalaman maksimum tiap pohon
          random_state=42        # Untuk reprodusibilitas hasil
      )
      
      # Melatih model menggunakan data training
      gb_model.fit(X_train, y_train)
      ```
      Penjelasan:
      - GradientBoostingRegressor adalah model ensemble berbasis decision tree yang menggabungkan beberapa pohon secara bertahap untuk memperbaiki kesalahan prediksi.
      - Parameter n_estimators, learning_rate, dan max_depth telah disesuaikan (tuning) untuk mendapatkan performa terbaik.
        
2. ```Evaluasi Model:```
   
Evaluasi model dilakukan menggunakan tiga metrik utama: Mean Absolute Error (MAE): Rata-rata dari selisih absolut antara prediksi dan nilai asli. Root Mean Squared Error (RMSE): Akar dari rata-rata kuadrat selisih antara prediksi dan nilai asli. R² Score: Mengukur seberapa baik variasi data target dapat dijelaskan oleh fitur.

| Model                     | MAE      | RMSE     | R²       |
| ------------------------- | -------- | -------- | -------- |
| Linear Regression         | 3.620537 | 4.586971 | 0.928409 |
| Gradient Boosting (Tuned) | 2.205147 | 3.192256 | 0.965326 |

3. ```Analisis dan Pemilihan Model Terbaik```
   
Berdasarkan hasil evaluasi di atas, Gradient Boosting Regressor memberikan performa yang lebih baik dibandingkan Linear Regression, dengan nilai MAE dan RMSE yang lebih rendah serta nilai R² Score yang lebih tinggi (0.9653).
Hal ini menunjukkan bahwa model Gradient Boosting mampu menangkap kompleksitas dan non-linearitas data lebih baik daripada Linear Regression. Oleh karena itu, Gradient Boosting Regressor dipilih sebagai model terbaik untuk digunakan dalam tahap deployment atau inference selanjutnya.

## Evaluation
Pada bagian ini anda perlu menyebutkan metrik evaluasi yang digunakan. Lalu anda perlu menjelaskan hasil proyek berdasarkan metrik evaluasi yang digunakan.

Sebagai contoh, Anda memiih kasus klasifikasi dan menggunakan metrik **akurasi, precision, recall, dan F1 score**. Jelaskan mengenai beberapa hal berikut:
- Penjelasan mengenai metrik yang digunakan
- Menjelaskan hasil proyek berdasarkan metrik evaluasi

Ingatlah, metrik evaluasi yang digunakan harus sesuai dengan konteks data, problem statement, dan solusi yang diinginkan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan formula metrik dan bagaimana metrik tersebut bekerja.

**---Ini adalah bagian akhir laporan---**

_Catatan:_
- _Anda dapat menambahkan gambar, kode, atau tabel ke dalam laporan jika diperlukan. Temukan caranya pada contoh dokumen markdown di situs editor [Dillinger](https://dillinger.io/), [Github Guides: Mastering markdown](https://guides.github.com/features/mastering-markdown/), atau sumber lain di internet. Semangat!_
- Jika terdapat penjelasan yang harus menyertakan code snippet, tuliskan dengan sewajarnya. Tidak perlu menuliskan keseluruhan kode project, cukup bagian yang ingin dijelaskan saja.

