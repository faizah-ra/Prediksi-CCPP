# Laporan Proyek Machine Learning Terapan - Felix Rafael
## Project Domain
Menurut **[Hanahan & Weinberg (2011)](https://doi.org/10.1016/j.cell.2011.02.013)**, Kanker merupakan suatu penyakit yang ditandai oleh _proliferasi_ sel abnormal secara tidak terkendali yang dapat menyerang jaringan di sekitarnya dan menyebar ke organ tubuh lain melalui proses _metastasis_. Secara biologis, kanker terjadi karena mutasi genetik yang memengaruhi jalur pengaturan siklus sel, _apoptosis_, dan mekanisme perbaikan DNA, yang menyebabkan sel kehilangan kemampuan untuk mengatur pertumbuhannya secara normal. Saat ini, kanker tetap menjadi tantangan kesehatan global yang signifikan, dengan dampak yang luas terhadap individu dan sistem kesehatan masyarakat. 

Menurut laporan **[International Agency for Research on Cancer (2020) ](https://doi.org/10.3322/caac.21660)**, terdapat hampir 20 juta kasus kanker baru dan sekitar 10 juta kematian akibat kanker di seluruh dunia pada tahun tersebut. Kanker paru-paru, payudara, dan kolorektal merupakan jenis yang paling umum, dengan kanker paru-paru menjadi penyebab utama kematian akibat kanker. Proyeksi yang ditunjukkan dari laporan **[ International Agency for Research on Cancer dan American Cancer Society  (2024) ](https://www.iarc.who.int/news-events/new-report-on-global-cancer-burden-in-2022-by-world-region-and-human-development-level)** menunjukkan bahwa pada tahun 2050, jumlah kasus kanker baru tahunan dapat mencapai 35 juta, meningkat 77% dari angka tahun 2022.

Deteksi dini kanker sangat penting dalam meningkatkan hasil pengobatan dan kelangsungan hidup pasien. Namun, menurut penelitian yang dilakukan oleh **[Crosby et al. (2022) ](https://www.science.org/doi/10.1126/science.aay9040)** sekitar 50% kasus kanker didiagnosis pada stadium lanjut, yang secara signifikan mengurangi efektivitas pengobatan dan peluang kesembuhan. Studi oleh **[Cancer Research UK (2023) ](https://www.cancerresearchuk.org/about-cancer/spot-cancer-early/why-is-early-diagnosis-important)** menekankan bahwa diagnosis kanker pada tahap awal meningkatkan kemungkinan pengobatan yang berhasil dan kelangsungan hidup pasien. Oleh karena itu, strategi untuk meningkatkan deteksi dini sangat penting dalam upaya mengurangi beban kanker secara global.

Dalam konteks ini, pendekatan berbasis teknologi, khususnya _Machine Learning_ (ML), menawarkan potensi besar dalam meningkatkan deteksi dan prediksi keparahan kanker. ML dapat menganalisis data klinis dan lingkungan pasien untuk mengidentifikasi pola yang mungkin tidak terlihat oleh metode konvensional. Studi oleh **[Zhou dan Rhrissorrakrai (2024)](https://doi.org/10.48550/arXiv.2410.22387)** menunjukkan bahwa ML dapat digunakan untuk menemukan _biomarker multi-omik_ yang berkaitan dengan keparahan kanker prostat, yang dapat membantu dalam penilaian dan pengobatan pasien .

Proyek ini bertujuan untuk mengembangkan model prediksi keparahan kanker menggunakan beberapa algoritma _Machine Learning_. Dengan memanfaatkan data dari berbagai faktor genetik dan lingkungan, model diharapkan dapat memberikan prediksi yang akurat mengenai tingkat keparahan kanker pada pasien. Implementasi model dapat membantu dalam pengambilan keputusan klinis, perencanaan pengobatan, dan pada akhirnya, meningkatkan hasil kesehatan pasien secara keseluruhan.

## Business Understanding
### Problem Statements
- Tingkat keparahan kanker yang bervariasi antar pasien membuat diagnosis dan penanganan menjadi kompleks, ditambah kurangnya alat bantu berbasis kecerdasan buatan yang dapat memberikan prediksi tingkat keparahan kanker kepada tenaga medis atau pasien.
- Belum adanya sistem prediksi terintegrasi yang menggabungkan faktor genetik, gaya hidup, dan lingkungan secara bersamaan dalam memperkirakan tingkat keparahan kanker. 
- Model _baseline_ sederhana yang belum mampu memberikan akurasi prediksi yang tinggi dalam konteks regresi medis. 

### Goals
- Mengembangkan model _Machine Learning_ berbasis regresi untuk memprediksi tingkat keparahan kanker dengan akurasi tinggi.
- Mengintegrasikan berbagai fitur dari data genetik, gaya hidup, dan lingkungan guna meningkatkan akurasi dan generalisasi model.
- Melakukan eksperimen dengan beberapa algoritma tingkat lanjut untuk menemukan model terbaik dalam memprediksi skor keparahan kanker, serta memvisualisasikan hasil evaluasi model untuk analisis residual yang lebih dalam.

### Solution Statement
- Mengimplementasikan model regresi berbasis _Machine Learning_ tingkat lanjut, seperti Random Forest Regressor, XGBoost Regressor, dan LightGBM Regressor untuk memprediksi tingkat keparahan kanker berdasarkan integrasi data dari faktor genetik, gaya hidup, dan lingkungan dengan akurasi yang tinggi.
- Mengintegrasikan berbagai fitur penting dari domain genetik, gaya hidup, dan lingkungan dengan analisis korelasi berbasis statistik untuk memastikan input yang diberikan ke model relevan dan berkualitas tinggi.
- Melakukan visualisasi dan evaluasi mendalam, termasuk _plotting_ residual, _predicted vs actual scores_, dan distribusi _error_ guna memahami perilaku model dan mengidentifikasi bias atau pola ketidaksesuaian dalam hasil prediksi.

## Data Understanding
Dataset yang digunakan dalam proyek ini diperoleh dari platform Kaggle dengan judul **`Global Cancer Patients (2015–2024)`**. Dataset ini berisi data pasien kanker dari berbagai negara dan mencakup sejumlah faktor penting seperti data demografis, genetika, gaya hidup, serta kondisi lingkungan pasien. Tujuan dari penggunaan dataset ini adalah untuk membangun model regresi yang dapat memprediksi tingkat keparahan kanker berdasarkan kombinasi fitur-fitur tersebut. 

Dataset ini terdiri dari 50.000 baris data pasien dan 15 kolom fitur, disajikan dalam format tabular (CSV). Seluruh data telah melalui proses pembersihan sehingga tidak terdapat _missing values_ maupun data duplikat, menjadikannya siap untuk dianalisis dan digunakan dalam pemodelan _machine learning_.

Link Dataset: https://www.kaggle.com/datasets/zahidmughal2343/global-cancer-patients-2015-2024

### Variabel-variabel pada dataset Global Cancer Patients (2015–2024) sebagai berikut:
- **`Patient_ID`** : 	ID unik untuk mengidentifikasi setiap pasien. Tidak digunakan dalam pemodelan.
- **`Age`** :	Usia pasien pada saat data dikumpulkan (dalam tahun).
- **`Gender`** : Jenis kelamin pasien (Male, Female, Other).
- **`Country_Region`** : Negara atau wilayah asal pasien.
- **`Year`** : Tahun pencatatan data pasien, antara 2015 hingga 2024.
- **`Genetic_Risk`** : Tingkat risiko genetik terhadap kanker, dalam skala 0–10.
- **`Air_Pollution`** : 	Indeks paparan polusi udara di lingkungan pasien (semakin tinggi, semakin buruk), dalam skala 0-10.
- **`Alcohol_Use`** : Tingkat konsumsi alkohol pasien, dalam skala kuantitatif 0-10.
- **`Smoking`** : Tingkat kebiasaan merokok pasien, dalam skala 0–10.
- **`Obesity_Level`** : Tingkat obesitas pasien, dalam skala 0-10.
- **`Cancer_Type`** : 	Jenis kanker yang diderita oleh pasien (contoh: Lung, Breast, dll).
- **`Cancer_Stage`** : Tahapan kanker (contoh: Stage I, Stage II, Stage III, Stage IV).
- **`Treatment_Cost_USD`** : Perkiraan total biaya pengobatan yang telah dikeluarkan (dalam USD).
- **`Survival_Years`** : 	Estimasi waktu bertahan hidup pasien setelah diagnosis (dalam tahun).
- **`Target_Severity_Score`**: 	Skor target yang menunjukkan tingkat keparahan kanker (nilai kontinu). Merupakan variabel target (label) dalam prediksi.

### Visualisasi Distribusi Data Numerik
![Numerical_Distribution](./assets/Numerical_Visualization.png)
Distribusi variabel numerik dalam dataset ini memberikan gambaran awal yang penting untuk memahami karakteristik data yang akan digunakan dalam pemodelan. Variabel _Age_ menunjukkan distribusi yang relatif merata antara rentang usia 20 hingga 90 tahun, yang mengindikasikan bahwa dataset ini mencakup pasien dari berbagai kelompok usia tanpa dominasi signifikan pada kelompok tertentu. Distribusi _Year_ sebagai penanda waktu pencatatan data tampak merata antara tahun 2015 hingga 2024, menandakan bahwa data dikumpulkan secara konsisten selama rentang waktu satu dekade.

Pada variabel _Genetic_Risk_, _Air_Pollution_, _Alcohol_Use_, _Smoking_, dan _Obesity_Level_, distribusi tampak menyebar merata dari nilai 0 hingga 10. Ini menunjukkan bahwa tingkat risiko genetik, paparan polusi udara, konsumsi alkohol, kebiasaan merokok, serta tingkat obesitas bervariasi secara signifikan antar pasien, tanpa adanya _outlier_ yang ekstrem. Hal ini penting karena menunjukkan keragaman yang cukup pada faktor risiko yang akan dianalisis terhadap tingkat keparahan kanker.

Sementara itu, _Treatment_Cost_USD_ atau biaya pengobatan memperlihatkan sebaran yang cenderung merata dari angka 0 hingga mendekati 100.000 USD. Ini merefleksikan variasi yang tinggi dalam pembiayaan medis pasien, kemungkinan bergantung pada jenis kanker, stadium, dan lokasi geografis. Variabel _Survival_Years_, yang menunjukkan estimasi tahun bertahan hidup setelah diagnosis, juga tersebar merata antara 0 hingga 10 tahun, mencerminkan keberagaman prognosis dari masing-masing pasien.

Yang paling menarik adalah distribusi target atau label prediksi, yaitu Target_Severity_Score. Variabel ini memiliki distribusi menyerupai kurva normal (distribusi Gaussian), dengan sebagian besar pasien memiliki skor keparahan di sekitar nilai tengah (mean). Distribusi ini sangat ideal untuk masalah regresi karena dapat membantu model untuk belajar secara lebih stabil dan akurat terhadap variasi tingkat keparahan kanker berdasarkan fitur-fitur yang tersedia.

### Visualisasi Distribusi Data Kategorikal
![Categorical_Gender](./assets/Categorical_Gender.png)
Visualisasi distribusi gender menunjukkan persebaran yang hampir merata antara ketiga kategori gender dalam data pasien kanker. Persentase pasien laki-laki tercatat sebesar 33,6%, diikuti oleh pasien perempuan sebesar 33,4%, dan kategori lainnya sebesar 33%. Distribusi yang relatif seimbang ini mencerminkan keberagaman gender pada data yang digunakan, sehingga memberikan dasar representasi yang adil dalam proses analisis lebih lanjut terhadap faktor-faktor yang memengaruhi tingkat keparahan kanker.

![Categorical_Country](./assets/Categorical_Country_Region.png)
Visualisasi distribusi berdasarkan region menunjukkan bahwa data pasien kanker tersebar cukup merata di antara sepuluh negara, yaitu Australia, UK, USA, India, Germany, Russia, Brazil, Pakistan, China, dan Canada. Masing-masing negara memiliki proporsi pasien yang berada dalam rentang 9,7% hingga 10,2%. Persebaran yang relatif merata ini mengindikasikan bahwa dataset memiliki cakupan geografis yang luas dan representatif, sehingga memungkinkan analisis yang lebih komprehensif terhadap pengaruh faktor wilayah terhadap tingkat keparahan kanker.

![Categorical_CancerType](./assets/Categorical_Cancer_Type.png)
Distribusi jenis kanker dalam dataset ini menunjukkan persebaran yang cukup merata di antara delapan tipe kanker utama, yaitu: _Colon_ (Kanker Usus Besar), _Prostate_ (Kanker Prostat), _Leukemia_ (Kanker Darah), _Liver_ (Kanker Hati), _Skin_ (Kanker Kulit), _Cervical_ (Kanker Serviks), _Breast_ (Kanker Payudara), dan _Lung_ (Kanker Paru-Paru). Masing-masing jenis kanker memiliki proporsi yang berkisar antara 12,3% hingga 12,8%. Keseimbangan ini menunjukkan bahwa dataset dikurasi secara seimbang untuk tiap tipe kanker, sehingga sangat ideal untuk membangun model prediksi yang general dan tidak bias terhadap salah satu jenis kanker tertentu.

![Categorical_CancerStage](./assets/Categorical_Cancer_Stage.png)
Distribusi stadium kanker dalam dataset ini mencakup Stadium 0 hingga Stadium 4, yang menggambarkan tingkat keparahan atau progresi penyakit kanker pada pasien. Masing-masing stadium memiliki proporsi yang cukup merata, yaitu berada dalam kisaran 19,9% hingga 20,2%. Pemerataan distribusi ini menunjukkan bahwa data yang digunakan bersifat seimbang untuk setiap tingkat stadium, dari yang paling ringan (Stage 0) hingga yang paling berat (Stage 4). Hal ini penting untuk memastikan bahwa model yang dibangun tidak bias terhadap tingkat keparahan tertentu, sehingga performa prediksi dapat diandalkan di seluruh spektrum kondisi pasien.

### Visualisasi Correlation Matrix
![Correlation Matrix](./assets/Correlation_Matrix.png)

Visualisasi _Correlation matrix_ menunjukkan hubungan antar variabel numerik dalam dataset. Terlihat bahwa _Target Severity Score_ memiliki korelasi positif tertinggi dengan _Genetic Risk_ (0.48), _Smoking_ (0.48), _Air Pollution_ (0.37), dan _Alcohol Use_ (0.36), yang mengindikasikan bahwa faktor-faktor tersebut cukup berpengaruh terhadap tingkat keparahan kanker. Sebaliknya, terdapat korelasi negatif antara _Treatment Cost_ dan _Target Severity Score_ (-0.47), yang bisa jadi mengindikasikan bahwa pasien dengan tingkat keparahan tinggi mendapatkan penanganan lebih awal atau tidak mampu menanggung biaya tinggi. Korelasi antar variabel lainnya cenderung lemah atau tidak signifikan.

### Visualisasi Relasi Fitur Numerik
![KDE](./assets/KDE.png)
Visualisasi tersebut merupakan _pairplot_ yang menampilkan hubungan antar variabel numerik dalam dataset kanker. Secara umum, sebagian besar fitur tidak menunjukkan korelasi yang kuat satu sama lain, terlihat dari sebaran titik yang acak. Namun, terdapat korelasi positif yang cukup jelas antara fitur-fitur seperti _Air Pollution_, _Alcohol Use_, _Smoking_, dan _Obesity Level_ terhadap _Target Severity Score_, yang menunjukkan bahwa faktor gaya hidup dan lingkungan memiliki kontribusi terhadap tingkat keparahan kanker. Sementara itu, _Treatment Cost_  memiliki korelasi negatif terhadap _Target Severity Score_.

## Data Preparation
Berikut ini adalah beberapa tahap yang dilakukan sebagai berikut:
-  **`Feature Selection`**: Langkah pertama yang dilakukan adalah menghapus beberapa kolom yang dianggap tidak relevan terhadap proses pemodelan dan tidak memberikan kontribusi signifikan terhadap prediksi tingkat keparahan kanker. Kolom-kolom yang dihapus meliputi _Patient_ID_, Gender, _Country_Region_, dan _Year_. Kolom _Patient_ID_ merupakan _identifier_ unik yang tidak memiliki nilai prediktif, sementara kolom Gender, _Country_Region_, dan _Year_ tidak menunjukkan hubungan yang kuat dengan variabel target berdasarkan hasil eksplorasi awal serta analisis korelasi. Penghapusan ini bertujuan untuk menyederhanakan data, serta meningkatkan efisiensi proses pemodelan selanjutnya.
    ```python
    # Drop unnecessary columns: Patient_ID, Gender, Country_Region, and Year
    df_cleaned = df.drop(columns=['Patient_ID', 'Gender', 'Country_Region', 'Year'])
    df_cleaned.head()
    ```

-  **`Handling Outlier`**: Pada tahap ini, dilakukan visualisasi distribusi data numerik menggunakan _boxplot_ untuk mendeteksi keberadaan _outlier_. _Boxplot_ memberikan gambaran tentang median, kuartil, serta titik-titik data yang berada di luar rentang normal (outlier). Selain itu, digunakan juga metode _Interquartile Range_ (IQR) untuk mendeteksi _outlier_ secara kuantitatif. Pertama, dihitung kuartil pertama (Q1) dan kuartil ketiga (Q3) dari setiap fitur numerik, kemudian IQR dihitung sebagai selisih antara Q3 dan Q1. Batas bawah dan batas atas untuk mendeteksi _outlier_ ditentukan dengan rumus Q1 - 1,5 × IQR dan Q3 + 1,5 × IQR. Data yang berada di luar batas ini dikategorikan sebagai _outlier_. Jumlah _outlier_ pada setiap kolom numerik kemudian dihitung untuk mengetahui sebaran nilai ekstrim dalam _dataset_. Informasi ini menjadi dasar dalam mengambil keputusan penanganan _outlier_ agar data yang digunakan dalam pemodelan lebih bersih dan model yang dihasilkan lebih akurat serta stabil.
    ```python
    # Create boxplot for each numeric column to visualize distribution and detect outliers
    sns.set(style="whitegrid")
    for col in numeric_cols:
        plt.figure(figsize=(10, 4))
        ax = sns.boxplot(
            x=df[col],
            color="skyblue",
            width=0.5,
            fliersize=5,
            linewidth=1.5,
            boxprops=dict(facecolor='lightblue', edgecolor='black'),
            medianprops=dict(color='red', linewidth=2),
            whiskerprops=dict(color='black'),
            capprops=dict(color='black'),
            flierprops=dict(marker='o', markerfacecolor='orange', markersize=5, linestyle='none')
        )
    
        plt.title(f'Distribution and Outliers: {col}', fontsize=14, fontweight='bold')
        plt.xlabel(col, fontsize=12)
        plt.ylabel("Value Distribution", fontsize=11)
        plt.grid(axis='x', linestyle='--', linewidth=0.5, alpha=0.7)
        plt.tight_layout()
        plt.show()
    
    # Calculate the first quartile (Q1), third quartile (Q3), and interquartile range (IQR)
    Q1 = df_numeric.quantile(0.25)
    Q3 = df_numeric.quantile(0.75)
    IQR = Q3 - Q1
    
    # Calculate lower and upper bounds for detecting outliers
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Count the number of outliers in each numeric column
    outlier_counts = ((df_numeric < lower_bound) | (df_numeric > upper_bound)).sum().sort_values(ascending=False)
    print("Number of Outliers per Column:\n")
    print(outlier_counts)
    ```

- **`Standardization`**: Berikutnya dilakukan standarisasi pada fitur-fitur numerik. Standarisasi dilakukan menggunakan metode _StandardScaler_ yang mengubah distribusi data agar memiliki rata-rata (mean) nol dan standar deviasi satu. Proses ini penting untuk memastikan bahwa setiap fitur numerik memiliki skala yang sama sehingga tidak ada fitur yang mendominasi proses pemodelan hanya karena skala nilainya yang besar. Standarisasi membantu meningkatkan performa dan konvergensi algoritma _machine learning_, terutama untuk model yang sensitif terhadap skala fitur seperti regresi dan SVM.
    ```python
    # Apply standardization only to numeric features
    scaler = StandardScaler()
    df_cleaned[numeric_cols] = scaler.fit_transform(df_cleaned[numeric_cols])
    df_cleaned.head()
    ```

- **`Checking Feature Distribution After Standardization`**: Setelah proses standarisasi dilakukan terhadap fitur numerik, dilakukan visualisasi kembali menggunakan _boxplot_ untuk memastikan bahwa data telah memiliki skala yang seragam. _Boxplot_ ini menunjukkan bahwa nilai-nilai pada setiap fitur numerik kini terdistribusi di sekitar nol dengan standar deviasi satu, sesuai dengan karakteristik hasil transformasi _StandardScaler_. Langkah ini penting untuk memverifikasi keberhasilan proses standarisasi serta untuk mendeteksi jika masih terdapat _outlier_ atau anomali yang signifikan. Dengan distribusi yang lebih seimbang antar fitur, model _machine learning_ yang akan dibangun diharapkan dapat bekerja lebih optimal tanpa bias terhadap fitur tertentu yang sebelumnya memiliki skala lebih besar.
    ```python
    # Plot boxplot of numeric features after standardization
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df_numeric, orient="v", palette="Set3", width=0.6, fliersize=4, linewidth=1)
    plt.title("Boxplot of Numeric Features (Predictors) After Standardization", fontsize=16, fontweight='bold')
    plt.xlabel("Features", fontsize=12)
    plt.ylabel("Standardized Values", fontsize=12)
    plt.xticks(rotation=30)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()
    ```

- **`Categorical Feature Encoding`**: Pada tahap ini, dilakukan proses _encoding_ terhadap fitur kategorikal yaitu _Cancer_Type_ dan _Cancer_Stage_ menggunakan metode _One-Hot Encoding_. Teknik ini digunakan untuk mengubah data kategorikal menjadi format numerik biner agar dapat diproses oleh algoritma _machine learning_, yang umumnya hanya menerima input numerik. _One-Hot Encoding_ bekerja dengan membuat kolom baru untuk setiap kategori unik dalam fitur, dan memberikan nilai 1 jika baris tersebut termasuk dalam kategori tersebut, serta 0 jika tidak. Penerapan encoding ini memastikan bahwa informasi kategorikal dapat digunakan secara efektif tanpa memperkenalkan urutan atau bobot yang tidak semestinya antar kategori, yang bisa menyesatkan model.
    ```python
    # Encode categorical columns 'Cancer_Type' and 'Cancer_Stage' using one-hot encoding
    df_encoded = pd.get_dummies(df_cleaned, columns=['Cancer_Type', 'Cancer_Stage'], dtype=int)
    df_encoded.head()
    ```
- **`Train-Test-Split`**: Tahap terakhir dalam proses persiapan data adalah memisahkan fitur (variabel independen) dari target (variabel dependen), yaitu _Target_Severity_Score_. Setelah pemisahan, data dibagi menjadi dua bagian, yaitu data latih (training set) sebesar 80% dan data uji (testing set) sebesar 20% menggunakan fungsi train_test_split. Pembagian ini bertujuan untuk melatih model pada sebagian data (training set) dan menguji performanya pada data yang belum pernah dilihat sebelumnya (testing set). Dengan cara ini, evaluasi model dapat dilakukan secara objektif dan menghindari _overfitting_.
    ```python
    # Separate features and target variable
    X = df_encoded.drop(columns=['Target_Severity_Score'])
    y = df_encoded['Target_Severity_Score']
    # Split data into training (80%) and testing (20%) sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42
    )
    ```

## Modeling
Pada tahap ini, dilakukan pembangunan model _machine learning_ untuk memprediksi tingkat keparahan kanker berdasarkan kombinasi fitur numerik dan kategorikal yang telah diproses sebelumnya. Pemodelan dilakukan dengan pendekatan regresi karena variabel target bersifat kontinu. Tiga model regresi yang digunakan adalah sebagai berikut:
- **`Random Forest`**: Metode _ensemble_ yang menggabungkan sejumlah pohon keputusan (decision tree) untuk menghasilkan prediksi akhir. Model ini menggunakan teknik _bootstrap aggregating_ (bagging) untuk meningkatkan stabilitas dan akurasi. Kelebihan dari model ini adalah dapat menangkap pola non-linear yang kompleks tanpa perlu banyak _preprocessing_, tahan terhadap _overfitting_ pada dataset dengan jumlah fitur yang banyak, serta cukup _robust_ terhadap _missing value_ dan _outlier_ ringan. Akan tetapi, model ini memiliki kekurangan berupa interpretasi model menjadi lebih sulit karena banyaknya pohon, serta konsumsi memori dan waktu pelatihan relatif besar, terutama untuk dataset yang lebih besar.
- **`XGBoost`**: Teknik _boosting_ yang menggunakan pendekatan gradien untuk memperbaiki kesalahan prediksi secara bertahap. Model ini juga dilengkapi dengan mekanisme regularisasi untuk mengontrol kompleksitas. Kelebihan dari model ini adalah performa tinggi terutama pada masalah tabular serta terdapat opsi regularisasi yang membantu mengurangi risiko _overfitting_. Model ini juga Mendukung _early stopping_ dan teknik _pruning_ yang efisien. Akan tetapi, model ini memiliki kekurangan berupa _hyperparameter tuning_ yang cukup sensitif dan memerlukan eksperimen yang cermat, dan proses pelatihan relatif lambat jika dibandingkan dengan LightGBM.
- **`LightGBM`**: Algoritma _boosting_ berbasis histogram yang dikembangkan untuk kecepatan dan efisiensi. Model ini membagi data berdasarkan _leaf-wise growth_, bukan _level-wise_ seperti XGBoost, yang membuatnya lebih cepat untuk dataset besar. Kelebihan dari model ini adalah waktu pelatihan jauh lebih cepat dibandingkan model _boosting_ lainnya, dapat menangani _dataset_ skala besar dengan efisien, serta mendukung pengolahan paralel dan GPU. Kekurangan dari model ini adalah cenderung lebih sensitif terhadap distribusi fitur dan _outlier_ ekstrim dan bisa mengalami _overfitting_ jika tidak dilakukan _tuning_ dengan benar.

Tahapan pembuatan model yang dilakukan adalah sebagai berikut:

1. **`Inisialisasi Model`**: 
Pada tahap ini, dilakukan inisialisasi dan pelatihan tiga model regresi yaitu Random Forest Regressor, XGBoost Regressor, dan LightGBM Regressor. Masing-masing model dikonfigurasi dengan parameter tertentu yang telah disesuaikan untuk meningkatkan performa model terhadap data yang digunakan.
    - Random Forest Regressor:
      
        Model ini cocok untuk menangani data non-linear dan memiliki ketahanan terhadap _overfitting_ dalam jumlah fitur yang besar.
        ```python
        rf_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
        ```
        - n_estimators=100: Menentukan jumlah pohon keputusan (decision trees) yang akan digunakan dalam _ensemble_. Semakin banyak pohon, biasanya semakin stabil hasilnya, namun memerlukan waktu pelatihan yang lebih lama.
        - max_depth=10: Batas kedalaman maksimum setiap pohon untuk menghindari model menjadi terlalu kompleks (overfitting).
        - random_state=42: Digunakan untuk memastikan hasil yang konsisten di setiap eksekusi.
        
    - XGBoost Regressor:
      
        XGBoost terkenal karena efisiensinya dalam menangani data tabular dan sering digunakan dalam kompetisi _machine learning_.
        ```python
        xgb_model = XGBRegressor(n_estimators=200, learning_rate=0.1, max_depth=6, random_state=42)
        ```
        - n_estimators=200: Jumlah _boosting rounds_ atau jumlah pohon yang dibangun secara berurutan.
        - learning_rate=0.1: Ukuran langkah untuk memperbarui prediksi pada setiap iterasi. Nilai yang lebih kecil menghasilkan proses pelatihan yang lebih lambat namun akurat.
       -  max_depth=6: Mengontrol kompleksitas setiap pohon. Nilai ini membantu menjaga keseimbangan antara bias dan varians.
        - random_state=42: Digunakan untuk memastikan hasil yang konsisten di setiap eksekusi.
        
    - LightGBM Regressor:
      
        LightGBM dirancang untuk efisiensi dan kecepatan, dan sangat optimal untuk dataset besar dengan banyak fitur.
        ```python
        lgb_model = LGBMRegressor(n_estimators=200, learning_rate=0.1, max_depth=6, random_state=42)
        ```
        - n_estimators=200: Jumlah _boosting rounds_ atau jumlah pohon yang dibangun secara berurutan.
        - learning_rate=0.1: Ukuran langkah untuk memperbarui prediksi pada setiap iterasi. Nilai yang lebih kecil menghasilkan proses pelatihan yang lebih lambat namun akurat.
       -  max_depth=6: Mengontrol kompleksitas setiap pohon. Nilai ini membantu menjaga keseimbangan antara bias dan varians.
        - random_state=42: Digunakan untuk memastikan hasil yang konsisten di setiap eksekusi.
        
2. **`Pelatihan Model`**: 
Setelah proses inisialisasi model selesai, tahap selanjutnya adalah melatih (training) ketiga model regresi, yaitu Random Forest Regressor, XGBoost Regressor, dan LightGBM Regressor menggunakan data pelatihan yang telah dipersiapkan (X_train dan y_train).
    - Random Forest Regressor
        ```python
        rf_model.fit(X_train, y_train)
        ```
    - XGBoost Regressor
         ```python
        xgb_model.fit(X_train, y_train)
        ```
    - LightGBM Regressor
         ```python
        lgb_model.fit(X_train, y_train)
        ```
3. **`Evaluasi Model`**:
Pada tahap ini, dilakukan evaluasi awal terhadap ketiga model regresi menggunakan data pelatihan. Evaluasi dilakukan menggunakan tiga metrik utama: _Mean Absolute Error_ (MAE), _Root Mean Squared Error_ (RMSE), dan _R² Score_, yang mengukur akurasi prediksi model terhadap data.
   
    | Model                    | MAE  |  RMSE | R² Score |
    |--------------------------|------|-------|----------|
    | Random Forest Regressor  |0.1653|0.2074 |0.9702    |
    | XGBoost Regressor        |0.0496|0.0627 |0.9973    |
    | LightGBM Regressor       |0.0516|0.0649 |0.9971    |
     
    Berdasarkan hasil evaluasi, **XGBoost Regressor** menunjukkan performa terbaik dengan nilai MAE dan RMSE paling rendah, serta _R² Score_ tertinggi (0.9973). Hal ini menunjukkan bahwa model ini paling mampu mempelajari pola dari data pelatihan secara efektif dan memberikan prediksi yang sangat akurat. Meskipun LightGBM juga memberikan hasil yang kompetitif, XGBoost sedikit lebih unggul dari sisi akurasi. Oleh karena itu, XGBoost Regressor dipilih sebagai model terbaik.

## Evaluation
Setelah proses pelatihan model selesai, tahap selanjutnya adalah melakukan evaluasi terhadap performa masing-masing model yang telah dibangun. Evaluasi dilakukan dengan tujuan untuk menilai seberapa baik model dalam memprediksi target, yaitu tingkat keparahan kanker, berdasarkan data uji yang belum pernah dilihat oleh model sebelumnya. Karena permasalahan yang diangkat merupakan regresi, maka digunakan beberapa metrik yang umum dalam regresi, yaitu _Mean Absolute Error_ (MAE), _Root Mean Squared Error_ (RMSE), dan _R² Score_. Masing-masing metrik ini memberikan sudut pandang yang berbeda dalam menilai akurasi prediksi model, baik dari segi rata-rata kesalahan, sensitivitas terhadap _outlier_, maupun proporsi variansi yang berhasil dijelaskan oleh model.

### Metrik Evaluasi
1. **`Mean Absolute Error (MAE)`**: MAE mengukur rata-rata absolut dari selisih antara nilai aktual dan prediksi.

    <img src="https://latex.codecogs.com/svg.image?\dpi{120}&space;\bg{transparent}&space;\color{White}&space;\text{MAE}&space;=&space;\frac{1}{n}&space;\sum_{i=1}^{n}&space;\left|y_i&space;-&space;\hat{y}_i\right|">
   
    Metrik ini memberikan gambaran rata-rata kesalahan model tanpa memperhatikan arah kesalahan (positif atau negatif), sehingga mudah diinterpretasikan. Semakin kecil nilai MAE, semakin akurat prediksi model terhadap data aktual.
   
    ```python
    # Random Forest MAE
    mae_rf = mean_absolute_error(y_test, y_pred_rf)
    # XGBoost MAE
    mae_xgb = mean_absolute_error(y_test, y_pred_xgb)
    # LightGBM MAE
    mae_lgb = mean_absolute_error(y_test, y_pred_lgb)
    ```

    Cara kerja: MAE menghitung seberapa besar kesalahan rata-rata prediksi model terhadap data aktual, tanpa mempertimbangkan arah kesalahan.

2. **`Root Mean Squared Error (RMSE)`**: RMSE mengukur akar dari rata-rata kuadrat selisih antara nilai aktual dan prediksi.

     <img src="https://latex.codecogs.com/svg.image?\dpi{120}&space;\bg{transparent}&space;\color{White}&space;\text{RMSE}&space;=&space;\sqrt{&space;\frac{1}{n}&space;\sum_{i=1}^{n}&space;(y_i&space;-&space;\hat{y}_i)^2&space;}">
        
      RMSE lebih sensitif terhadap _outlier_ dibanding MAE karena kesalahan dikuadratkan. Nilai lebih kecil menunjukkan prediksi lebih akurat secara keseluruhan. Oleh karena itu, RMSE sangat berguna untuk mendeteksi model yang sensitif terhadap _outlier_.
      ```python
      # Random Forest RMSE
      rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
      # XGBoost RMSE
      rmse_xgb = np.sqrt(mean_squared_error(y_test, y_pred_xgb))
      # LightGBM RMSE
      rmse_lgb = np.sqrt(mean_squared_error(y_test, y_pred_lgb))
      ```
                
      Cara kerja: RMSE menghitung rata-rata kuadrat dari kesalahan prediksi, kemudian diakarkan untuk mendapatkan satuan yang sama dengan target. Semakin besar kesalahan, semakin tinggi nilainya.

3. **`R² Score (Coefficient of Determination)`**: R² mengukur seberapa besar variansi dari data target yang dapat dijelaskan oleh model.

     <img src="https://latex.codecogs.com/svg.image?\dpi{120}&space;\bg{transparent}&space;\color{White}&space;R^2&space;=&space;1&space;-&space;\frac{&space;\sum_{i=1}^{n}&space;(y_i&space;-&space;\hat{y}_i)^2&space;}{&space;\sum_{i=1}^{n}&space;(y_i&space;-&space;\bar{y})^2&space;}" alt="R Squared Equation">

     Nilai R² berkisar antara 0 hingga 1, di mana nilai yang lebih tinggi menunjukkan bahwa model mampu menjelaskan variabilitas data target dengan lebih baik. Jika R² mendekati 1, berarti model hampir sepenuhnya mampu menjelaskan variasi dalam data.
        
    ```python
        # Random Forest R² Score
        r2_rf = r2_score(y_test, y_pred_rf)
        # XGBoost R² Score
        r2_xgb = r2_score(y_test, y_pred_xgb)
        # LightGBM R² Score
        r2_lgb = r2_score(y_test, y_pred_lgb)
     ```
        
    Cara kerja: R² membandingkan antara total kesalahan model dengan kesalahan baseline (rata-rata nilai aktual). Nilai 1 berarti prediksi sempurna, 0 berarti tidak lebih baik dari sekadar menebak rata-rata.

### Evaluasi model akhir dengan menggunakan _dataset test_
1. **`Random Forest Regressor`**
   
    | Model                    | MAE  |  RMSE | R² Score |
    |--------------------------|------|-------|----------|
    | Random Forest Regressor  |0.2024|0.2532 |0.9549    |

    Model Random Forest menunjukkan performa yang sangat baik pada data _test_, dengan nilai R² mendekati 1 yang menandakan bahwa model mampu menjelaskan 95.49% variasi dari data target. Nilai MAE sebesar 0.2024 berarti bahwa, secara rata-rata, prediksi model berbeda sekitar 0.2 poin dari nilai keparahan aktual pada skala yang digunakan. Nilai RMSE sebesar 0.2532 menunjukkan rata-rata jarak prediksi model terhadap nilai aktual, dengan penekanan pada kesalahan yang besar.

    ![ActualVSPredicted_RF](./assets/ap_rf.png)
    
    Grafik ini menunjukkan hubungan antara nilai aktual dan prediksi model. Titik-titik yang mendekati garis merah putus-putus (garis identitas) menandakan bahwa prediksi model cukup akurat.
    
    ![ResidualPlot_RF](./assets/rr_rf.png)
    
    Plot residual menampilkan sebaran kesalahan prediksi terhadap nilai prediksi. Residual tersebar secara acak di sekitar nol, menandakan bahwa model tidak memiliki pola sistematik yang menunjukkan bias.
    
    ![ResidualDistribution_RF](./assets/rd_rf.png)
    
    Histogram residual mendekati distribusi normal simetris dengan rata-rata mendekati nol. Ini menunjukkan bahwa kesalahan prediksi model bersifat acak dan tidak terdistribusi secara berat sebelah.

2. **`XGBoost Regressor`**

    | Model                    | MAE  |  RMSE | R² Score |
    |--------------------------|------|-------|----------|
    | XGBoost Regressor  |0.0628|0.0788 |0.9956    |

    Dengan MAE sebesar 0.0628, model XGBoost hanya memiliki rata-rata deviasi sekitar 0.06 poin dari nilai sebenarnya. Ini menandakan bahwa prediksi cukup konsisten dan tidak terlalu meleset. RMSE sebesar 0.0788 juga menunjukkan bahwa model XGBoost memiliki kesalahan prediksi yang rendah dan stabil. Dengan skor R² sebesar 0.9956, model XGBoost mampu menjelaskan 99.56% variasi dalam data. Hal ini menandakan bahwa model sangat baik dalam merepresentasikan hubungan antara fitur dan target.
    
    ![ActualVSPredicted_XG](./assets/ap_xg.png)
    
    Grafik ini memperlihatkan korelasi antara nilai aktual dan hasil prediksi model. Sebagian besar titik berada sangat dekat dengan garis identitas, menandakan bahwa prediksi XGBoost sangat akurat dan lebih mendekati nilai sebenarnya dibanding model Random Forest.
    
    ![ResidualPlot_XG](./assets/rr_xg.png)
    
    Sebaran residual XGBoost menunjukkan pola yang acak dan simetris di sekitar garis nol. Tidak terdapat tren atau pola tertentu, yang mengindikasikan bahwa model ini memiliki kesalahan prediksi yang konsisten tanpa bias sistematis, lebih stabil dibanding Random Forest yang menunjukkan sedikit penyebaran residual lebih luas.
    
    ![ResidualDistribution_XG](./assets/rd_xg.png)
    
    Histogram residual dari model XGBoost menunjukkan bentuk distribusi yang sangat mendekati distribusi normal, dengan puncak di sekitar nol. Hal ini memperkuat bukti bahwa kesalahan prediksi model bersifat acak dan tidak condong ke satu sisi, serta lebih terkonsentrasi dibanding model Random Forest, yang distribusinya sedikit lebih tersebar.

3. **`LightGBM Regressor`**
   
    | Model                    | MAE  |  RMSE | R² Score |
    |--------------------------|------|-------|----------|
    | LightGBM  |0.0582|0.0732 |0.9962    |

    MAE yang sangat rendah menunjukkan bahwa rata-rata selisih antara nilai prediksi dan aktual sangat kecil, menandakan model cukup presisi. RMSE lebih sensitif terhadap kesalahan besar, dan nilai yang rendah memperkuat bahwa model jarang membuat kesalahan besar dalam prediksi. R² Score mendekati 1, menandakan bahwa hampir seluruh variasi dalam data target berhasil dijelaskan oleh model, bahkan lebih baik daripada XGBoost maupun Random Forest.
     
    ![ActualVSPredicted_LG](./assets/ap_lg.png)
    
    Sebagian besar titik berada sangat rapat mengikuti garis identitas, menunjukkan bahwa prediksi LightGBM hampir sempurna. Akurasi prediksinya lebih tinggi dibandingkan model lain yang diuji.
    
    ![ResidualPlot_LG](./assets/rr_lg.png)
    
    Residual tersebar secara acak di sekitar nol tanpa pola tertentu. Hal ini menunjukkan bahwa kesalahan prediksi bersifat acak dan tidak mengindikasikan bias sistematis yang mana menandakan model generalisasi dengan baik.
    
    ![ResidualDistribution_LG](./assets/rd_lg.png)
    
    Histogram residual menunjukkan bentuk distribusi yang sangat simetris dan mengerucut di sekitar nol. Hal ini mengindikasikan bahwa kesalahan model sangat kecil dan tersebar secara seimbang.

## Conclusion
Dalam studi kasus ini, dilakukan serangkaian tahapan mulai dari _data understanding_, _data preprocessing_, _feature engineering_, standarisasi data, hingga pemodelan menggunakan tiga algoritma _machine learning_ yaitu Random Forest Regressor, XGBoost Regressor, dan LightGBM Regressor. Tujuannya adalah untuk membangun model prediktif yang mampu memperkirakan tingkat keparahan kanker berdasarkan berbagai faktor seperti genetik, gaya hidup, dan lingkungan.

Tahap evaluasi dilakukan dengan menggunakan metrik MAE, RMSE, dan _R² Score_ baik pada data latih maupun data uji. Hasil evaluasi menunjukkan bahwa:
- Random Forest memberikan performa cukup baik namun relatif kurang akurat dibanding dua model lainnya.
- XGBoost menunjukkan hasil prediksi yang sangat presisi dengan error kecil dan kemampuan generalisasi tinggi.
- LightGBM berhasil mengungguli kedua model lainnya dengan nilai MAE, RMSE, dan _R² Score_ terbaik, serta visualisasi residual yang paling ideal.

Berdasarkan evaluasi metrik dan analisis visual, LightGBM Regressor dipilih sebagai model terbaik dalam proyek ini. Model ini tidak hanya memberikan akurasi tinggi, tetapi juga efisiensi dalam proses pelatihan, serta mampu menangani dataset kompleks dengan baik.
