# 🎧 Klasifikasi Audio UrbanSound8K dengan CNN

Proyek ini bertujuan untuk mengklasifikasikan suara-suara lingkungan dari dataset **UrbanSound8K** menggunakan teknik **ekstraksi fitur MFCC** dan **model deep learning Convolutional Neural Network (CNN)**. Dataset ini terdiri dari berbagai macam suara seperti gonggongan anjing, klakson mobil, suara pengeboran, hingga tembakan, yang dapat dimanfaatkan dalam pengembangan sistem deteksi suara otomatis.

---

## 📂 Dataset

- **Nama**: UrbanSound8K
- **Jumlah Kelas**: 10 kategori suara:
  - air_conditioner  
  - car_horn  
  - children_playing  
  - dog_bark  
  - drilling  
  - engine_idling  
  - gun_shot  
  - jackhammer  
  - siren  
  - street_music
- **Jumlah File**: 8.732 file `.wav`
- **Struktur**: Dataset terbagi dalam 10 folder (`fold1` sampai `fold10`) berdasarkan cross-validation.
- **File Metadata**: `UrbanSound8K.csv` berisi informasi seperti nama file, label kelas, dan fold keberapa file tersebut berada.
- **Sumber**: [UrbanSound8K Dataset](https://urbansounddataset.weebly.com/urbansound8k.html)

---

## 🔧 Proses Pengolahan Data

### 1. Preprocessing Audio
Untuk memastikan semua file audio memiliki representasi fitur yang konsisten, dilakukan preprocessing sebagai berikut:

- **Resampling Audio**: Semua file dikonversi ke sampling rate 22.050 Hz.
- **Durasi Konsisten**: Semua file distandarkan menjadi 4 detik.
  - Jika kurang dari 4 detik → dilakukan padding (zero-padding)
  - Jika lebih → dipotong sesuai panjang
- **Ekstraksi Fitur MFCC**:
  - Ekstraksi 40 koefisien MFCC
  - MFCC diubah ke bentuk `(173, 40)` per file
  - Disesuaikan dengan fixed-length time-step (173 frame)

### 2. Encoding Label
- Label dikodekan menggunakan `LabelEncoder` ke bentuk integer (0–9).
- Label integer diubah menjadi one-hot encoded menggunakan `to_categorical`.

### 3. Split Dataset
- Data dibagi menjadi training (80%) dan testing (20%)
- Stratifikasi dilakukan agar distribusi kelas tetap proporsional

---

## 🏗️ Arsitektur Model CNN

Model CNN dirancang untuk mendeteksi pola spasial dari representasi MFCC. Berikut detail arsitektur:

```text
Input Layer: (173, 40, 1)  ← 173 time-step × 40 MFCC × 1 channel
│
├── Conv2D (32 filters, kernel 3x3, activation=ReLU)
│   └── Output: (171, 38, 32)
│
├── MaxPooling2D (pool_size=2x2)
│   └── Output: (85, 19, 32)
│
├── Dropout (rate=0.3)
│
├── Conv2D (64 filters, kernel 3x3, activation=ReLU)
│   └── Output: (83, 17, 64)
│
├── MaxPooling2D (pool_size=2x2)
│   └── Output: (41, 8, 64)
│
├── Dropout (rate=0.3)
│
├── Flatten Layer
│   └── Output: (20992,)
│
├── Dense Layer (128 neurons, activation=ReLU)
│
├── Dropout (rate=0.3)
│
└── Dense Output Layer (10 neuron, activation=Softmax)
    → Menghasilkan probabilitas 10 kelas suara
