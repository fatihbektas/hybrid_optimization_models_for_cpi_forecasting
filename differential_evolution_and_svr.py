import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
import random

# ==========================================
# 1. VERİ YÜKLEME VE ÖN İŞLEME
# ==========================================

# Excel dosyasını oku
df = pd.read_excel("src/tufe_2005_2025.xlsx", header=0)

# Veriyi "Wide" formattan "Long" formata çevir (Melt)
df_long = df.melt(id_vars=["Yıl"], var_name="Ay", value_name="Endeks")

# Ay isimlerini sayıya çevir
aylar = {
    "Ocak": 1,
    "Şubat": 2,
    "Mart": 3,
    "Nisan": 4,
    "Mayıs": 5,
    "Haziran": 6,
    "Temmuz": 7,
    "Ağustos": 8,
    "Eylül": 9,
    "Ekim": 10,
    "Kasım": 11,
    "Aralık": 12,
}
df_long["Ay_No"] = df_long["Ay"].map(aylar)

# Tarih sütunu oluştur (Pylance hatasını önleyen yöntemle)
tarih_bilesenleri = pd.DataFrame(
    {"year": df_long["Yıl"], "month": df_long["Ay_No"], "day": 1}
)
df_long["Tarih"] = pd.to_datetime(tarih_bilesenleri)

# Sırala ve temizle
df_final = df_long.sort_values("Tarih").reset_index(drop=True)
df_final = df_final[["Tarih", "Endeks"]].dropna()

# Numpy dizisine çevir
data_values = df_final["Endeks"].to_numpy().reshape(-1, 1)

# Normalizasyon (0-1 Aralığı)
scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(data_values)


# Sliding Window (Kayan Pencere) Oluşturma
def create_dataset(dataset, look_back=3):
    X, y = [], []
    for i in range(len(dataset) - look_back):
        a = dataset[i : (i + look_back), 0]
        X.append(a)
        y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(y)


look_back = 3
X, y = create_dataset(data_scaled, look_back)

# Eğitim ve Test Ayrımı (%80 - %20)
train_size = int(len(X) * 0.8)
test_size = len(X) - train_size

X_train, X_test = X[0:train_size], X[train_size : len(X)]
y_train, y_test = y[0:train_size], y[train_size : len(y)]

print(f"Veri Hazır! Eğitim Seti: {X_train.shape}, Test Seti: {X_test.shape}")

# ==========================================
# 2. AMAÇ FONKSİYONU (FITNESS FUNCTION)
# ==========================================


def objective_function(params):
    # Parametreleri al (Negatif olmamaları için kontrol et)
    C = max(0.01, params[0])
    epsilon = max(0.0001, params[1])
    gamma = max(0.0001, params[2])

    # Modeli Kur ve Eğit
    model = SVR(kernel="rbf", C=C, epsilon=epsilon, gamma=gamma)
    model.fit(X_train, y_train)

    # Test Et
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)

    return rmse


# ==========================================
# 3. DİFERANSİYEL GELİŞİM ALGORİTMASI
# ==========================================


def differential_evolution(
    objective, bounds, pop_size=20, generations=30, F=0.5, CR=0.7
):
    dim = len(bounds)

    # Popülasyon Başlatma
    population = []
    for _ in range(pop_size):
        individual = []
        for i in range(dim):
            val = bounds[i][0] + random.random() * (bounds[i][1] - bounds[i][0])
            individual.append(val)
        population.append(individual)

    population = np.array(population)

    # İlk en iyi değeri başlat
    best_vector = population[0]
    best_fitness = objective(best_vector)

    history = []

    print(f"Optimizasyon Başladı! ({generations} Jenerasyon)")

    for gen in range(generations):
        for i in range(pop_size):
            # Mutasyon
            idxs = [idx for idx in range(pop_size) if idx != i]
            a, b, c = population[np.random.choice(idxs, 3, replace=False)]
            mutant = a + F * (b - c)

            # Sınır Kontrolü
            for k in range(dim):
                mutant[k] = np.clip(mutant[k], bounds[k][0], bounds[k][1])

            # Çaprazlama
            trial = np.copy(population[i])
            for k in range(dim):
                if random.random() < CR:
                    trial[k] = mutant[k]

            # Seçilim
            fitness_trial = objective(trial)
            fitness_target = objective(population[i])

            if fitness_trial < fitness_target:
                population[i] = trial
                fitness_target = fitness_trial

            # Global En İyiyi Güncelle
            if fitness_target < best_fitness:
                best_fitness = fitness_target
                best_vector = population[i]

        history.append(best_fitness)
        print(f"Jenerasyon {gen+1}/{generations} -> En İyi RMSE: {best_fitness:.5f}")

    return best_vector, best_fitness, history


# ==========================================
# 4. ÇALIŞTIRMA VE SONUÇLAR
# ==========================================

# Parametre Sınırları: C, Epsilon, Gamma
bounds = [(0.1, 100), (0.001, 0.5), (0.001, 5.0)]

best_params, best_score, loss_history = differential_evolution(
    objective_function, bounds, pop_size=15, generations=20
)

print("\n--- OPTİMİZASYON TAMAMLANDI ---")
# best_params dizisini güvenli şekilde yazdır
print(
    f"En İyi Parametreler:\n C (Ceza): {best_params[0]:.4f}\n Epsilon: {best_params[1]:.4f}\n Gamma: {best_params[2]:.4f}"
)
print(f"En Düşük Hata (RMSE): {best_score:.5f}")

# Grafiği Çiz
plt.plot(loss_history)
plt.title("Diferansiyel Gelişim - Optimizasyon Süreci")
plt.xlabel("Jenerasyon")
plt.ylabel("RMSE Hatası")
plt.show()

# ==========================================
# 5. SONUÇLARIN GÖRSELLEŞTİRİLMESİ VE METRİKLER
# ==========================================

# 1. En iyi parametrelerle modeli tekrar kur
# best_params dizisi önceki adımdan geliyor: [C, Epsilon, Gamma]
final_model = SVR(
    kernel="rbf", C=best_params[0], epsilon=best_params[1], gamma=best_params[2]
)
final_model.fit(X_train, y_train)

# 2. Tahmin Yap (Hem eğitim hem test seti için)
train_predict = final_model.predict(X_train)
test_predict = final_model.predict(X_test)

# 3. Ters Normalizasyon (0-1'den Gerçek Değerlere Dönüş)
# reshape(-1, 1) yapıyoruz çünkü scaler 2D array bekler
train_predict = scaler.inverse_transform(train_predict.reshape(-1, 1))
y_train_real = scaler.inverse_transform(y_train.reshape(-1, 1))

test_predict = scaler.inverse_transform(test_predict.reshape(-1, 1))
y_test_real = scaler.inverse_transform(y_test.reshape(-1, 1))


# 4. Performans Metrikleri (MAPE - Ortalama Mutlak Yüzde Hata)
def calculate_mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


mape_score = calculate_mape(y_test_real, test_predict)
rmse_real = np.sqrt(mean_squared_error(y_test_real, test_predict))

print(f"\n--- FİNAL SONUÇLAR ---")
print(f"Test Seti MAPE Değeri: %{mape_score:.2f}")
print(f"Test Seti RMSE (Gerçek Endeks Puanı): {rmse_real:.2f}")

# 5. Grafik Çizimi
# Verileri uç uca ekleyip tüm zaman serisini gösterelim
total_len = len(data_values)
train_len = len(train_predict)
test_len = len(test_predict)

# Boş bir grafik şablonu oluştur
plot_train = np.empty((total_len, 1))
plot_train[:] = np.nan
plot_train[look_back : train_len + look_back] = train_predict

plot_test = np.empty((total_len, 1))
plot_test[:] = np.nan
# Test verisi eğitimden hemen sonra başlar
plot_test[train_len + look_back : total_len] = test_predict

plt.figure(figsize=(12, 6))
plt.plot(
    scaler.inverse_transform(data_scaled),
    label="Gerçek TÜFE Verisi",
    color="blue",
    alpha=0.6,
)
plt.plot(plot_train, label="Eğitim Tahmini", color="green", linestyle="--")
plt.plot(plot_test, label="Test Tahmini (Gelecek)", color="red", linewidth=2)

plt.title(f"TÜFE Tahmin Sonucu (SVR + DE)\nMAPE: %{mape_score:.2f}")
plt.xlabel("Zaman (Ay)")
plt.ylabel("TÜFE Endeksi")
plt.legend()
plt.grid(True)
plt.show()

# ==========================================
# 6. GELECEK AY TAHMİNİ (FORECASTING)
# ==========================================

# Veri setindeki SON 3 ayı al (Çünkü look_back=3)
last_window = data_scaled[-look_back:]
# Boyutunu ayarla (1 satır, 3 sütun formatına)
last_window = last_window.reshape(1, -1)

# Tahmin yap
future_pred_scaled = final_model.predict(last_window)

# Gerçek değere çevir
future_pred_real = scaler.inverse_transform(future_pred_scaled.reshape(-1, 1))

# Son Tarihi Bul
last_date = df_final["Tarih"].iloc[-1]
# Bir sonraki ayı hesapla
next_month = last_date + pd.DateOffset(months=1)

print(f"\n--- GELECEK TAHMİNİ ---")
print(f"Veri Setindeki Son Tarih: {last_date.strftime('%Y-%m')}")
print(f"Son Gerçekleşen Endeks: {df_final['Endeks'].iloc[-1]:.2f}")
print(f"--------------------------------------------------")
print(f"Tarih: {next_month.strftime('%Y-%m')} (Tahmin)")
print(f"Öngörülen TÜFE Endeksi: {future_pred_real[0][0]:.2f}")

# Tahmini Enflasyon Oranını Hesapla (Aylık)
current_index = df_final["Endeks"].iloc[-1]
predicted_index = future_pred_real[0][0]
inflation_rate = ((predicted_index - current_index) / current_index) * 100

print(f"Öngörülen Aylık Enflasyon Artışı: %{inflation_rate:.2f}")
