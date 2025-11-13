import numpy as np
import matplotlib.pyplot as plt
x_train = np.array([
    2, 3, 4, 5, 6, 8, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55,
    60, 65, 70, 75, 80, 85, 90, 95, 100, 105, 110, 120, 130,
    140, 150, 160, 170, 180, 190, 200, 220, 240, 260, 280,
    300, 320, 340, 360, 370, 375, 380
])

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
])

x_mean, y_mean = np.mean(x_train), np.mean(y_train)
w = np.sum((x_train - x_mean)*(y_train - y_mean)) / np.sum((x_train - x_mean)**2)
b = y_mean - w*x_mean
print(x_mean,y_mean,np.std(x_train))
print("Hàm hồi quy tuyến tính: y = {:.4f}x + {:.4f}".format(w, b))
y_pred = w * x_train + b

# --- Vẽ đồ thị ---
plt.figure(figsize=(8,5))
plt.scatter(x_train, y_train, color='blue', label='Dữ liệu gốc')
plt.plot(x_train, y_pred, color='red', linewidth=2, label=f'y = {w:.2f}x + {b:.2f}')

plt.title('Hồi quy tuyến tính (Least Squares Method)', fontsize=13)
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()