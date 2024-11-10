import numpy as np
import matplotlib.pyplot as plt
from plotly.graph_objects import Figure, Surface

import pandas as pd
#Baranivska_Valeriia_KM-23
# Графік функції rxt,t' для значень t,t' з кроком 0,1.
# Графік функції Kxt,t' для значень t,t' в діапазоні [0; 4] з кроком 0,1.

# Визначаємо діапазон значень для t і t', D(U), M(U)
d_u =0.1
m_u = 1
t_vals = np.arange(0, 4, 0.1)
t_prime_vals = np.arange(0, 4, 0.1)

# Визначаємо функцію випадкового процесу
def cos_func(x):
    cos1 = np.e ** (-x**2)

    return cos1

# Визначаємо функцію кореляції K_X(t, t')
def K_X(t, t_prime):
    return d_u *cos_func(t) * cos_func(t_prime)

# Створюємо матрицю значень кореляційної функції
K_matrix = np.zeros((len(t_vals), len(t_prime_vals)))


for i, t in enumerate(t_vals):
    for j, t_prime in enumerate(t_prime_vals):
        K_matrix[i, j] = K_X(t, t_prime)

# Візуалізуємо кореляційну функцію
plt.figure(figsize=(10, 8))
plt.contourf(t_vals, t_prime_vals, K_matrix, cmap='BuPu')
plt.colorbar(label='Kx(t, t\')')
plt.title('Кореляційна функція Kx(t, t\')')
plt.xlabel('t')
plt.ylabel('t\'')
plt.grid(True)
plt.show()

# Визначаємо функцію нормованої кореляції r_X(t, t')
def r_X(t, t_prime):
    cos_t = cos_func(t)
    cos_t_prime = cos_func(t_prime)
    return np.sign(cos_t) * np.sign(cos_t_prime)

# Створюємо матрицю значень нормованої кореляційної функції
r_matrix = np.zeros((len(t_vals), len(t_prime_vals)))

for i, t in enumerate(t_vals):
    for j, t_prime in enumerate(t_prime_vals):
        r_matrix[i, j] = r_X(t, t_prime)

# Візуалізуємо нормовану кореляційну функцію
plt.figure(figsize=(10, 8))
plt.contourf(t_vals, t_prime_vals, r_matrix, cmap='Purples')
plt.colorbar(label='r_x(t, t\')')
plt.title('Нормована кореляційна функція r_x(t, t\')')
plt.xlabel('t')
plt.ylabel('t\'')
plt.grid(True)
plt.show()



# Параметри
D_U = 1
t_values = np.arange(0, 4, 0.1)
t_prime_values = np.arange(0, 4, 0.1)

# Створення сітки значень t і t'
T, T_prime = np.meshgrid(t_values, t_prime_values)

# Кореляційна функція
K_X = D_U * (np.e**(- T ** 2) ) * (np.e ** (- T_prime ** 2))


# Створення DataFrame для зберігання даних
z_data = pd.DataFrame(K_X)

# Створення 3D-поверхні
fig = Figure(data=[Surface(z=z_data.values, x=t_values, y=t_prime_values)])

# Налаштування макету графіка
fig.update_layout(title='Кореляційна функція Kx(t, t\')',
                  autosize=False, width=1000, height=1000,
                  scene=dict(
                      zaxis_title='Kx(t, t\')',
                      xaxis_title='t',
                      yaxis_title='t\'',
                      xaxis=dict(nticks=10, dtick=0.5, range=[0, 3], gridcolor='black', gridwidth=1),
                      yaxis=dict(nticks=10, dtick=0.5, range=[0, 3], gridcolor='black', gridwidth=0.5),
                      zaxis=dict(nticks=10, dtick=0.5, range=[0, 3], gridcolor='black', gridwidth=0.5)),
                  colorway=["#70CA2A", "#E4B530", "#3A339F"])
# Відображення графіка
fig.show()

# Створення сітки значень t і t'
T, T_prime = np.meshgrid(t_values, t_prime_values)

# Кореляційна функція
R_X = D_U * np.sign(np.cos(np.cos(0.1 * T**2))) * np.sign(np.cos(np.cos(0.1 * T_prime**2)))

# Створення DataFrame для зберігання даних
z_data1 = pd.DataFrame(R_X)

# Створення 3D-поверхні
fig1 = Figure(data=[Surface(z=z_data1.values)])
fig1.update_layout(title='Нормована кореляційна функція r_x(t, t\')',
                   autosize=False, width=1000, height=1000,
                   scene=dict(
                       zaxis_title='Rx(t, t\')',
                       xaxis_title='t',
                       yaxis_title='t\'',
                       xaxis=dict(nticks=10, dtick=0.5, range=[0,3], gridcolor='black', gridwidth=1),
                       yaxis=dict(nticks=10, dtick=0.5, range=[0,3], gridcolor='black', gridwidth=0.5),
                       zaxis=dict(nticks=10, dtick=0.5, range=[0,3], gridcolor='black', gridwidth=0.5)),
                   colorway=["#70CA2A", "#E4B530", "#3A339F"])
# Відображення графіка
fig1.show()