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
- AT (Ambient Temperature) : Suhu udara ambien di sekitar pembangkit listrik, diukur dalam derajat Celsius.
- V (Exhaust Vacuum) : Tekanan vakum pada kondensor pembangkit listrik, diukur dalam cm Hg.
- AP (Ambient Pressure) : Tekanan udara ambien, diukur dalam milibar (mbar).
- RH (Relative Humidity) : Kelembapan relatif udara ambien, dalam persen (%).
- PE (Electrical Power Output) : Output daya listrik yang dihasilkan oleh pembangkit listrik, dalam megawatt (MW). Variabel ini merupakan target yang akan diprediksi.

### Pemahaman Data:
#### Visualisasi Distribusi Data
![image](https://github.com/user-attachments/assets/8098f6b2-4d89-4d24-b3bf-782280b4196e)




**Rubrik/Kriteria Tambahan (Opsional)**:
- Melakukan beberapa tahapan yang diperlukan untuk memahami data, contohnya teknik visualisasi data atau exploratory data analysis.

## Data Preparation
Pada bagian ini Anda menerapkan dan menyebutkan teknik data preparation yang dilakukan. Teknik yang digunakan pada notebook dan laporan harus berurutan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan proses data preparation yang dilakukan
- Menjelaskan alasan mengapa diperlukan tahapan data preparation tersebut.

## Modeling
Tahapan ini membahas mengenai model machine learning yang digunakan untuk menyelesaikan permasalahan. Anda perlu menjelaskan tahapan dan parameter yang digunakan pada proses pemodelan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan kelebihan dan kekurangan dari setiap algoritma yang digunakan.
- Jika menggunakan satu algoritma pada solution statement, lakukan proses improvement terhadap model dengan hyperparameter tuning. **Jelaskan proses improvement yang dilakukan**.
- Jika menggunakan dua atau lebih algoritma pada solution statement, maka pilih model terbaik sebagai solusi. **Jelaskan mengapa memilih model tersebut sebagai model terbaik**.

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

