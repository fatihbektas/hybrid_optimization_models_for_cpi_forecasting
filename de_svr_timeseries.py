import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
import random

# ----------------------------
# 0) Reproducibility (seed)
# ----------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# ----------------------------
# 1) Veri okuma ve long formata çevirme
# ----------------------------
df = pd.read_excel("src/tufe_2005_2025.xlsx", header=0)

df_long = df.melt(id_vars=["Yıl"], var_name="Ay", value_name="Endeks")

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

tarih_bilesenleri = pd.DataFrame(
    {
        "year": df_long["Yıl"].astype(int),
        "month": df_long["Ay_No"].astype(int),
        "day": 1,
    }
)
df_long["Tarih"] = pd.to_datetime(tarih_bilesenleri)

df_final = df_long.sort_values("Tarih").reset_index(drop=True)
df_final = df_final[["Tarih", "Endeks"]].dropna()

# ----------------------------
# 2) Ölçekleme (0-1)
# ----------------------------
data_values = df_final["Endeks"].to_numpy().reshape(-1, 1)
scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(data_values)


# ----------------------------
# 3) Sliding Window dataset
# ----------------------------
def create_dataset(dataset, look_back=3):
    X, y = [], []
    for i in range(len(dataset) - look_back):
        X.append(dataset[i : (i + look_back), 0])
        y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(y)


look_back = 3
X, y = create_dataset(data_scaled, look_back)

# ----------------------------
# 4) Train / Val / Test split (zaman sırası korunur)
# ----------------------------
N = len(X)
train_ratio = 0.70
val_ratio = 0.15
# kalan test ~ 0.15

train_end = int(N * train_ratio)
val_end = int(N * (train_ratio + val_ratio))

X_train, y_train = X[:train_end], y[:train_end]
X_val, y_val = X[train_end:val_end], y[train_end:val_end]
X_test, y_test = X[val_end:], y[val_end:]

print("Train:", X_train.shape, "Val:", X_val.shape, "Test:", X_test.shape)


# ----------------------------
# 5) Metrikler
# ----------------------------
def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def calculate_mape(y_true, y_pred):
    # y_true sıfır olamaz; TÜFE için pratikte sorun olmaz
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    return float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)


# ----------------------------
# 6) Amaç fonksiyonu: TimeSeriesSplit CV ile yalnız TRAIN üzerinde optimize et
# ----------------------------
def objective_function_cv(params, X_tr, y_tr, n_splits=4):
    C = max(0.01, float(params[0]))
    epsilon = max(1e-4, float(params[1]))
    gamma = max(1e-4, float(params[2]))

    tscv = TimeSeriesSplit(n_splits=n_splits)

    rmses = []
    for tr_idx, va_idx in tscv.split(X_tr):
        X_tr_fold, y_tr_fold = X_tr[tr_idx], y_tr[tr_idx]
        X_va_fold, y_va_fold = X_tr[va_idx], y_tr[va_idx]

        model = SVR(kernel="rbf", C=C, epsilon=epsilon, gamma=gamma)
        model.fit(X_tr_fold, y_tr_fold)
        pred = model.predict(X_va_fold)
        rmses.append(rmse(y_va_fold, pred))

    return float(np.mean(rmses))


# ----------------------------
# 7) Differential Evolution (DE)
# ----------------------------
def differential_evolution(
    objective, bounds, pop_size=20, generations=30, F=0.5, CR=0.7
):
    dim = len(bounds)

    population = []
    for _ in range(pop_size):
        individual = []
        for i in range(dim):
            low, high = bounds[i]
            val = low + random.random() * (high - low)
            individual.append(val)
        population.append(individual)
    population = np.array(population, dtype=float)

    best_vector = population[0].copy()
    best_fitness = objective(best_vector)
    history = [best_fitness]

    for gen in range(generations):
        for i in range(pop_size):
            idxs = [idx for idx in range(pop_size) if idx != i]
            a, b, c = population[np.random.choice(idxs, 3, replace=False)]
            mutant = a + F * (b - c)

            # bounds clip
            for k in range(dim):
                mutant[k] = np.clip(mutant[k], bounds[k][0], bounds[k][1])

            # crossover
            trial = population[i].copy()
            j_rand = np.random.randint(dim)  # en az 1 parametre mutant'tan gelsin
            for k in range(dim):
                if (random.random() < CR) or (k == j_rand):
                    trial[k] = mutant[k]

            fitness_trial = objective(trial)
            fitness_target = objective(population[i])

            if fitness_trial < fitness_target:
                population[i] = trial
                fitness_target = fitness_trial

            if fitness_target < best_fitness:
                best_fitness = fitness_target
                best_vector = population[i].copy()

        history.append(best_fitness)
        print(f"Gen {gen+1}/{generations} | Best CV-RMSE: {best_fitness:.6f}")

    return best_vector, best_fitness, history


# ----------------------------
# 8) Optimize edilecek aralıklar
# ----------------------------
bounds = [(0.1, 200.0), (0.001, 0.5), (0.0001, 5.0)]  # C  # epsilon  # gamma

# DE objective: sadece TRAIN CV hatası
obj = lambda p: objective_function_cv(p, X_train, y_train, n_splits=4)

best_params, best_cv_rmse, loss_history = differential_evolution(
    obj, bounds, pop_size=18, generations=25, F=0.6, CR=0.8
)

best_C, best_eps, best_gamma = best_params
print("\n--- OPTİMİZASYON TAMAMLANDI (CV üzerinden) ---")
print(f"C       = {best_C:.6f}")
print(f"epsilon = {best_eps:.6f}")
print(f"gamma   = {best_gamma:.6f}")
print(f"Best CV-RMSE (scaled) = {best_cv_rmse:.6f}")

# DE kayıp grafiği
plt.figure(figsize=(9, 4))
plt.plot(loss_history)
plt.title("DE Progress: Best CV-RMSE (scaled)")
plt.xlabel("Generation")
plt.ylabel("RMSE")
plt.grid(True)
plt.tight_layout()
plt.show()

# ----------------------------
# 9) Seçilen parametrelerle model eğitimi:
#    (train + val) ile fit, test'te değerlendir
# ----------------------------
X_trainval = np.vstack([X_train, X_val])
y_trainval = np.hstack([y_train, y_val])

final_model = SVR(
    kernel="rbf", C=float(best_C), epsilon=float(best_eps), gamma=float(best_gamma)
)
final_model.fit(X_trainval, y_trainval)

train_pred = final_model.predict(X_train)
val_pred = final_model.predict(X_val)
test_pred = final_model.predict(X_test)

# inverse transform (gerçek ölçek)
train_pred_real = scaler.inverse_transform(train_pred.reshape(-1, 1)).reshape(-1)
val_pred_real = scaler.inverse_transform(val_pred.reshape(-1, 1)).reshape(-1)
test_pred_real = scaler.inverse_transform(test_pred.reshape(-1, 1)).reshape(-1)

y_train_real = scaler.inverse_transform(y_train.reshape(-1, 1)).reshape(-1)
y_val_real = scaler.inverse_transform(y_val.reshape(-1, 1)).reshape(-1)
y_test_real = scaler.inverse_transform(y_test.reshape(-1, 1)).reshape(-1)

# rapor
test_mape = calculate_mape(y_test_real, test_pred_real)
test_rmse = rmse(y_test_real, test_pred_real)

print("\n--- TEST RAPORU (gerçek ölçekte) ---")
print(f"TEST RMSE = {test_rmse:.4f}")
print(f"TEST MAPE = %{test_mape:.2f}")

# ----------------------------
# 10) Tahminleri zaman eksenine hizalama (plot)
# ----------------------------
# Orijinal seri boyutu (data_values) ile hizalama için boş diziler
plot_train = np.full_like(data_values, fill_value=np.nan, dtype=float).reshape(-1)
plot_val = np.full_like(data_values, fill_value=np.nan, dtype=float).reshape(-1)
plot_test = np.full_like(data_values, fill_value=np.nan, dtype=float).reshape(-1)

# X ve y, look_back kadar kaydığı için y'nin ilk indeksine denk gelen yer: look_back
# y'nin indeksleri 0..N-1, gerçek seride look_back..look_back+N-1 aralığına oturur.
# train: y[0:train_end]
# val:   y[train_end:val_end]
# test:  y[val_end:]

base = look_back  # y[0]'ın serideki yeri

plot_train[base : base + train_end] = train_pred_real
plot_val[base + train_end : base + val_end] = val_pred_real
plot_test[base + val_end : base + N] = test_pred_real

dates = df_final["Tarih"].to_numpy()

plt.figure(figsize=(12, 5))
plt.plot(dates, df_final["Endeks"].to_numpy(), label="Gerçek Seri")
plt.plot(dates, plot_train, "--", label="Train Tahmin")
plt.plot(dates, plot_val, "--", label="Val Tahmin")
plt.plot(
    dates, plot_test, "-", linewidth=2, label=f"Test Tahmin (MAPE %{test_mape:.2f})"
)
plt.title("SVR (DE ile optimize) - Train/Val/Test Ayrımı (Leakage yok)")
plt.xlabel("Tarih")
plt.ylabel("Endeks")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ----------------------------
# 11) Bir adım ileri (1 ay) forecast
# ----------------------------
last_window = data_scaled[-look_back:].reshape(1, look_back)
future_scaled = final_model.predict(last_window)
future_real = scaler.inverse_transform(future_scaled.reshape(-1, 1)).reshape(-1)[0]

last_date = df_final["Tarih"].iloc[-1]
next_month = last_date + pd.DateOffset(months=1)

current_index = float(df_final["Endeks"].iloc[-1])
infl_rate = ((future_real - current_index) / current_index) * 100

print("\n--- 1 AY İLERİ TAHMİN ---")
print(f"Son tarih: {last_date.date()} | Son endeks: {current_index:.3f}")
print(f"Tahmin: {next_month.date()} | Endeks: {future_real:.3f}")
print(f"Tahmini aylık değişim: %{infl_rate:.2f}")
