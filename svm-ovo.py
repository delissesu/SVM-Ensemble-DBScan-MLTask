import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.svm import SVC
from sklearn.multiclass import OneVsOneClassifier
from sklearn.inspection import DecisionBoundaryDisplay

# 1. Menyiapkan Dataset
# Kita buat 4 kelas dan 2 fitur utama agar mudah divisualisasikan dalam sumbu X dan Y
X, y = make_classification(
    n_samples=1000, 
    n_features=2,        # Diubah ke 2 fitur untuk visualisasi 2D
    n_informative=2,     
    n_redundant=0, 
    n_classes=4,         # Diset untuk 4 kelas (A, B, C, D)
    n_clusters_per_class=1,
    random_state=42
)

# 2. Mendefinisikan Model Biner (SVM)
# Kita gunakan kernel 'linear' atau 'rbf' (di sini menggunakan RBF agar batas pemisahnya lebih luwes)
base_model = SVC(kernel='rbf', random_state=42)

# 3. Menerapkan Strategi One-vs-One
ovo_model = OneVsOneClassifier(base_model)

# 4. Melatih (Fitting) Model
ovo_model.fit(X, y)

# 5. Melakukan Prediksi (Mendapatkan nilai class)
y_pred = ovo_model.predict(X)
print("Contoh 10 prediksi pertama:", y_pred[:10])
print("Contoh 10 label asli:", y[:10])

# 6. Visualisasi Batas Keputusan (Decision Boundary)
plt.figure(figsize=(10, 8))

# Menggambar batas wilayah tiap kelas
disp = DecisionBoundaryDisplay.from_estimator(
    ovo_model, X, response_method="predict",
    cmap=plt.get_cmap('RdYlBu'), alpha=0.5,
    xlabel="Fitur 1", ylabel="Fitur 2"
)

# Menggambar titik-titik data
scatter = disp.ax_.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.get_cmap('RdYlBu'), edgecolors='k', s=40)

# Menambahkan legenda dan judul
plt.legend(*scatter.legend_elements(), title="Kelas (0,1,2,3)")
plt.title("Visualisasi SVM 4 Kelas dengan Strategi One-vs-One (OvO)")
plt.show()