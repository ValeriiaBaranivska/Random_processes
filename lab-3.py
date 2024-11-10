import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson
from scipy.stats import rv_continuous
from scipy.integrate import quad

# 2.1 Моделювання випадкової величини з розподілом Пуассона
lambda_param = 0.9
n_samples = 50

# Генерація вибірки з розподілу Пуассона
samples = np.random.poisson(lambda_param, n_samples)

# Крок 3: Обчислення емпіричних частот
unique, counts = np.unique(samples, return_counts=True)
empirical_frequencies = counts / n_samples  # нормалізація для відображення частот
# Обчислення теоретичних частот для тих же значень
theoretical_frequencies = poisson.pmf(unique, lambda_param)

# Побудова гістограми емпіричних частот
plt.figure(figsize=(12, 6))
# Стовпчики для емпіричних частот
plt.bar(unique, empirical_frequencies, width=0.4, alpha=0.6, label='Емпіричні частоти', color='purple', align='center')


# Крок 4: Обчислення теоретичних частот для кожного можливого значення
theoretical_probs = poisson.pmf(unique, lambda_param)  # теоретичні ймовірності для унікальних значень вибірки

# Стовпчики для теоретичних частот
plt.bar(unique + 0.1 , theoretical_frequencies, width=0.4, alpha=0.6, label='Теоретичні частоти', color='orange', align='center')

# Налаштування графіка
plt.xlabel('Значення')
plt.ylabel('Частота')
plt.title('Гістограма вибірки з розподілу Пуассона та теоретичні частоти')
plt.legend()
plt.grid()
plt.show()

# Виведення результатів
print("Значення вибірки:", samples)
print("Емпіричні частоти:", np.round(empirical_frequencies, 4))
print("Теоретичні частоти:", np.round(theoretical_probs, 4))


# Визначення функції розподілу F(x)
def Fx(x):
    result = np.zeros_like(x)
    result[x < 0] = 0
    result[(0 <= x) & (x <= np.pi / 2)] = 1 - np.cos(x[(0 <= x) & (x <= np.pi / 2)])
    result[x > np.pi / 2] = 1
    return result

# Визначення оберненої функції розподілу (інверсна функція)
def Fx_inv(y):
    result = np.zeros_like(y)
    result[(0 <= y) & (y <= 1)] = np.arccos(1 - y[(0 <= y) & (y <= 1)])
    return result

# Крок 1: Генерація вибірки через інверсну функцію розподілу
n_samples = 50
uniform_samples = np.random.uniform(0, 1, n_samples)
samples = Fx_inv(uniform_samples)  # Генеруємо вибірку значень відповідно до функції розподілу

# Крок 2: Побудова гістограми вибірки
plt.figure(figsize=(8, 6))
counts, bins, _ = plt.hist(samples, bins=10, density=True, alpha=0.6, color='purple', label='Гістограма вибірки')

# Крок 3: Додавання графіка теоретичної щільності розподілу
x_range = np.linspace(0, np.pi / 2, 1000)
theoretical_density = (np.sin(x_range)) / 2  # Теоретична щільність розподілу

plt.plot(x_range, theoretical_density, color='orange', linewidth=2, label='Графік щільності розподілу')

# Налаштування графіка
plt.title('Гістограма 2.2')
plt.xlabel('Значення X')
plt.ylabel('Щільність')
plt.legend()
plt.grid()
plt.show()

# Виведення результатів
print("значення вибірки X:", np.round(samples, 4))
"""# 2.2 Моделювання випадкової величини з заданою функцією розподілу F(x)
def generate_random_variable_F(n):
    """"""Генерація випадкових чисел з заданим розподілом F(x)""""""
    Y = np.random.uniform(0, 1, n)  # Генеруємо випадкові значення Y на [0, 1]
    X = np.zeros(n)
    for i in range(n):
        if Y[i] < 0:
            X[i] = 0
        elif 0 < Y[i] <= 1 - np.cos(np.pi / 2):  # Вибір значень за інверсною функцією розподілу
            X[i] = np.arccos(1 - Y[i])
        else:
            X[i] = np.pi / 2
    return X

# Генерація вибірки
samples_F = generate_random_variable_F(n_samples)

# Побудова гістограми емпіричних частот
plt.figure(figsize=(12, 6))
counts, bins, _ = plt.hist(samples_F, bins=7, density=True, alpha=0.6, color='purple', label='Гістограма вибірки')

# Додавання теоретичної щільності розподілу
x_range = np.linspace(0, np.pi / 2, 1000)
density = np.where((x_range >= 0) & (x_range <= np.pi / 2), (np.sin(x_range)) / 2, 0)  # теоретична щільність f(x)
plt.plot(x_range, density, color='orange', label='Теоретична щільність')

# Налаштування графіка
plt.title('Гістограма вибірки з функцією розподілу F(x) та теоретична щільність')
plt.xlabel('Значення X')
plt.ylabel('Щільність ймовірності')
plt.legend()
plt.grid()
plt.show()

# Виведення результатів
print("Вибірка з функцією розподілу F(x):", np.round(samples_F[:10], 4))  # Виводимо перші 10 значень вибірки

"""


"""# Крок 1: Генерація псевдовипадкових чисел Y
Y = np.random.uniform(0, 1, 50)  # 100 випадкових чисел від 0 до 1

# Крок 2: Обчислення значень випадкової величини X
lambda_param = 0.9  # Параметр експоненційного розподілу
X = -np.log(1 - Y) / lambda_param  # Обернена функція

# Крок 3: Побудова гістограми вибірки
plt.figure(figsize=(12, 6))
counts_1, bins_1, _ = plt.hist(X, bins=7, density=True, alpha=0.6, color='orange', label='Гістограма вибірки')

# Крок 4: Додавання графіка щільності розподілу
x_range = np.linspace(0, np.max(X), 1000)  # Значення для побудови графіка щільності
density = lambda_param * np.exp(-lambda_param * x_range)  # Експоненційна щільність
plt.plot(x_range, density, color='purple', label='Графік щільності розподілу')

# Налаштування графіка
plt.title('Імітація експоненційного розподілу')
plt.xlabel('Значення X')
plt.ylabel('Щільність ймовірності')
plt.legend()
plt.grid()
plt.show()

# Виведення результатів
print("Перші 10 значень вибірки X:", np.round(X[:10], 4))

# Крок 2: Вибираємо перші 20 значень
event_intervals = X[:20]

# Крок 3: Обчислюємо моменти виникнення подій
# Початок з часу 0
event_times = np.cumsum(event_intervals)  # Накопичуємо інтервали, щоб отримати часи подій

# Крок 4: Візуалізація моментів виникнення подій на осі часу
plt.figure(figsize=(12, 6))
plt.plot(event_times, np.zeros_like(event_times), 'ro', markersize=10)  # Позиції подій на осі X
plt.vlines(event_times, -0.1, 0.1, color='purple', alpha=0.5)  # Вертикальні лінії для позначення моментів

# Налаштування графіка
plt.title('Моменти виникнення подій з експоненційним розподілом')
plt.xlabel('Час')
plt.yticks([])  # Сховаємо вісь Y, оскільки вона не потрібна
plt.grid()
plt.xlim(0, np.max(event_times) + 5)  # Розширюємо межі графіка
plt.show()

# Виведення моментів виникнення подій
print("Моменти виникнення подій:", np.round(event_times, 4))

"""