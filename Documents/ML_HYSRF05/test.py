import numpy as np
from solomon import solomon_split
from kfold import kfold_linear_1d, _fit_wb, _predict, _metrics  # từ kfold.py đã tạo

# --- DỮ LIỆU CỦA BẠN ---
x_train = np.array([
    2, 3, 4, 5, 6, 8, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 
    60, 65, 70, 75, 80, 85, 90, 95, 100, 105, 110, 120, 130, 
    140, 150, 160, 170, 180, 190, 200, 220, 240, 260, 280, 
    300, 320, 340, 360, 370, 375, 380
], dtype=float)

y_train = np.array([
    2.782325, 3.25935, 4.067545, 5.04757, 6.001975, 8.326895, 
    10.236335, 15.23321, 19.16292, 23.82822, 28.634885, 33.47087, 
    38.24309, 43.087885, 47.57985, 52.06106, 56.979895, 61.867765, 
    66.620375, 71.474615, 76.4032, 81.338595, 85.72017, 90.860385, 
    95.48461, 100.976285, 104.3609942, 115.81523, 125.367865, 
    135.302295, 145.42794, 156.20736, 165.91558, 175.484755, 
    185.44674, 195.15566, 214.725685, 234.09771, 254.23034, 
    273.649395, 293.15766, 311.469805, 332.2342118, 350.567955, 
    361.148005, 365.54858, 370.504295
], dtype=float)

# --- SOLOMON SPLIT: test_ratio=0.2 => train:test = 80:20 ---
# Với dữ liệu 1D, để SOLOMON có tiêu chí corr/KMO/communalities, ta ghép x và y thành ma trận X (n_samples x 2).
X = np.column_stack([x_train, y_train])

split = solomon_split(X, test_ratio=0.2, seed=42, max_iter=2000)
train_idx, test_idx = split["idx_A"], split["idx_B"]   # idx_B có kích thước ≈ 20%

x_tr, y_tr = x_train[train_idx], y_train[train_idx]
x_te, y_te = x_train[test_idx], y_train[test_idx]

print("SOLOMON diagnostics:",
      {k: split[k] for k in ["corr_frobenius","kmo_A","kmo_B","S_A","S_B"]})

# --- K-FOLD CV TRÊN TRAIN ---
cv = kfold_linear_1d(x_tr, y_tr, k=5, shuffle=True, seed=42, standardize_per_fold=False)
print("CV averages on TRAIN:", cv["averages"])

# --- FIT LẠI TRÊN TOÀN BỘ TRAIN & ĐÁNH GIÁ TEST ---
w, b = _fit_wb(x_tr, y_tr)
y_hat_test = _predict(x_te, w, b)
mse_te, mae_te, r2_te = _metrics(y_te, y_hat_test)
print(f"FINAL TEST (SOLOMON 80:20) | MSE={mse_te:.6f} | MAE={mae_te:.6f} | R2={r2_te:.6f}")
