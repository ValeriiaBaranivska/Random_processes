import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from math import ceil, sqrt

# Для кореляційної матриці
from numpy.linalg import norm as np_norm

data = {
    3: [0.834, 0.000, -0.603, -0.021, 0.526, 1.160, 1.370, 0.599, 0.904, 0.712],
    7: [-0.609, -0.297, -0.565, -0.403, 0.208, 0.972, 0.777, 0.476, 0.465, -0.361],
    12: [0.081, -0.178, -0.788, -0.477, 0.471, -0.354, 1.410, 0.006, 0.315, 0.658],
    23: [0.724, 0.152, -0.220, 0.139, 0.157, -0.033, 0.535, 0.248, 0.311, 0.192],
    32: [0.563, -0.280, -0.694, 0.142, 0.851, 0.864, 1.100, 0.267, 0.359, -0.154],
    38: [0.076, 0.189, -0.845, -0.357, 0.661, 0.672, 0.647, 0.349, 0.846, -0.096],
    42: [-0.254, 0.039, -0.471, -0.045, 0.587, 0.475, 0.873, 0.468, -0.183, 0.544],
    56: [-0.229, -1.330, -0.569, -0.115, -0.021, 0.031, 0.733, 0.813, 0.542, 0.820],
    58: [-0.032, -1.290, -0.845, 0.156, 0.991, 0.381, 0.870, 0.094, 0.709, -0.224],
    68: [0.642, -1.690, -0.766, 0.341, -0.168, 0.987, 0.934, 0.770, 0.112, 0.236],
    76: [0.258, 0.099, -0.991, -0.410, 0.474, 0.962, 0.786, -0.393, 0.269, 0.055],
    89: [-0.076, -0.636, -1.100, -0.214, 0.743, 1.100, 0.942, 0.745, -0.048, -0.188]
}

df = pd.DataFrame(data)
time_points = pd.RangeIndex(start=0, stop=10, step=1)

mean_values = df.mean(axis=1) # математичне сподівання
var_values = df.var(axis=1) # дисперсія
std_values = df.std(axis=1) # середньоквадратичне відхилення

print("математичне сподівання")
print(mean_values)

print("дисперсія")
print(std_values)

print("середньоквадратичне відхилення")
print(var_values)

"""
    Побудувати графіки залежності математичного сподівання і 
середньоквадратичного відхилення від часу. 
"""

"""plt.figure(figsize=(18, 12))
for column in df.columns:
    plt.plot(time_points, df[column], label = f'Випадковий процес № {column}', alpha = 1)

plt.plot(time_points, mean_values, label="Математичне сподівання", color = 'darkgreen', linewidth = 5)
plt.fill_between(time_points, mean_values - std_values, mean_values + std_values, color='lightgreen', alpha=1, label='Середньоквадратичне відхилення')

plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.1)
plt.title('Графіки 12-ти реалізацій випадкового процесу, середньоквадратичного відхилення та математичного сподівання')
plt.xlabel('час')
plt.ylabel('значення')
plt.grid(True)
# Встановлення міток осі X
plt.xticks(ticks=np.arange(0, 10, 1))
plt.show()"""

"""
    Для кожного перерізу випадкового процесу побудувати гістограму і
приблизно оцінити вид розподілу відповідних випадкових величин.
    При побудові гістограми для вибору кількості інтервалів використовувати
формулу Стерджеса.
"""
# Побудова гістограм для кожного перерізу
# Якщо перерізи знаходяться в рядках, то транспонуємо DataFrame
df_transposed = df.T  # Транспонуємо DataFrame, щоб рядки стали стовпцями

# Нумерація стовпців після транспонування
"""for idx, col in enumerate(df_transposed.columns, 1):  # Ітерація по стовпцях транспонованого DataFrame
    plt.figure(figsize=(8, 4))

    # Побудова гістограми для кожного перерізу
    plt.hist(df_transposed[col], bins=ceil(1 + 3.322 * np.log10(len(df_transposed))), color='lightgreen',
             edgecolor='black', density=True)

    # Оцінка ймовірнісної густини (KDE)
    sns.kdeplot(df_transposed[col], color='black', label='оцінка розподілу')

    # Додаємо нумерацію у заголовок
    plt.title(f'Гістограма випадкового процесу №{idx}')  # idx — це номер стовпця

    # Оформлення графіка
    plt.xlabel('значення')
    plt.ylabel('частота')
    plt.legend()
    plt.grid(True)

    # Показ графіка
    plt.show()"""
"""df_transposed = df.T  # Транспонуємо DataFrame, щоб рядки стали стовпцями
for col in df_transposed.columns:
    stat, p_value = stats.shapiro(df_transposed[col])
    print(f'Переріз {col}: Статистика={stat}, p-value={p_value}')

    if p_value > 0.05:
        print(f'Переріз {col} можна вважати нормально розподіленим\n')
    else:
        print(f'Переріз {col} не є нормально розподіленим\n')

"""# обчислення кореляційної матриці випадкового процесу
# Преобразування в numpy масив
X = np.array(list(data.values()))

# Обчислення середніх значень для кожного моменту часу
mean_values = np.mean(X, axis=0)

# Ініціалізація порожньої матриці кореляції
num_times = X.shape[1]  # Кількість моментів часу (стовпці X)
correlation_matrix = np.zeros((num_times, num_times))

# Обчислення кореляційної матриці за формулою
for i in range(num_times):
    for j in range(num_times):
        correlation_matrix[i, j] = np.sum((X[:, i] * X[:, j]) - (mean_values[i] * mean_values[j])) / (len(X) - 1)

# Створюємо DataFrame для кореляційної матриці
df_corr = pd.DataFrame(correlation_matrix, columns=[f"t{i+1}" for i in range(num_times)], index=[f"t{i+1}" for i in range(num_times)])

# Виведення у вигляді таблиці
df_corr.to_csv('correlation_matrix.csv')
# Встановлення максимальних параметрів для відображення
pd.set_option('display.max_rows', None)  # Виводити всі рядки
pd.set_option('display.max_columns', None)  # Виводити всі стовпці
pd.set_option('display.float_format', '{:.4f}'.format)
# Ваш код для створення кореляційної матриці

print("Кореляційна матриця (як таблиця):")
print(df_corr)

# Візуалізація за допомогою heatmap
"""plt.figure(figsize=(10, 8))
sns.heatmap(df_corr, annot=True, fmt=".4f", cmap="BuPu")
plt.title("Кореляційна матриця")
plt.show()

corr_matrix = df.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='Purples', center=0)
plt.title('Нормована кореляційна матриця')
plt.show()"""
