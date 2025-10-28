import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# dữ liệu bạn cung cấp
x_val = np.array([150., 105., 170., 15., 10., 120., 380., 25.])
y_val = np.array([145.43, 100.98, 165.92, 15.23, 10.24, 115.82, 370.5, 23.83])

# bậc 6
degree = 6

# tạo đặc trưng bậc 6
X = x_val.reshape(-1, 1)
poly = PolynomialFeatures(degree=degree, include_bias=False)
X_poly = poly.fit_transform(X)

# fit mô hình hồi quy tuyến tính
model = LinearRegression()
model.fit(X_poly, y_val)

# hệ số mô hình
coefs = model.coef_
intercept = model.intercept_

print(f"Intercept (b0): {intercept:.6f}")
for i, c in enumerate(coefs, start=1):
    print(f"w{i} (x^{i}): {c:.6e}")

# hàm hồi quy
terms = [f"{c:+.3e}*x^{i}" for i, c in enumerate(coefs, start=1)]
equation = f"y = {intercept:.3e} " + " ".join(terms)
print("\nHàm hồi quy bậc 6:")
print(equation)

# dự đoán để vẽ
x_plot = np.linspace(min(x_val), max(x_val), 200).reshape(-1, 1)
y_pred = model.predict(poly.transform(x_plot))

# vẽ biểu đồ
plt.scatter(x_val, y_val, color='r', marker='x', label='Dữ liệu thật')
plt.plot(x_plot, y_pred, color='b', label='Hồi quy bậc 6')
plt.legend(); plt.xlabel('x'); plt.ylabel('y')
plt.title('Hàm hồi quy đa thức bậc 6')
plt.show()
