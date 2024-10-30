import numpy as np
import matplotlib.pyplot as plt

# Крок 1: Визначення значень та ймовірностей
values = np.array([0, 1, 2, 3, 4, 5, 6])  # 7 значень (додаємо 7 для випадків > 6)
probabilities = np.array([0.15, 0.05, 0.2, 0.15, 0.1, 0.3, 0.05])  # ймовірності

# Крок 2: Генерація вибірки
n_samples = 100
samples = np.random.choice(values, size=n_samples, p=probabilities)

# Крок 3: Обчислення емпіричних частот
unique, counts = np.unique(samples, return_counts=True)
empirical_frequencies = counts / n_samples

# Крок 4: Обчислення теоретичних частот
theoretical_frequencies = probabilities[unique]  # ймовірності для унікальних значень
empirical_frequencies = np.round(empirical_frequencies, 4)  # округлення до 4 знаків
theoretical_frequencies = np.round(theoretical_frequencies, 4)  # округлення


# Крок 5: Побудова гістограми
plt.figure(figsize=(12, 6))
plt.bar(unique, empirical_frequencies, width=0.4, alpha=0.6, label='Емпіричні частоти', color='purple', align='center')
plt.bar(unique, theoretical_frequencies, width=0.4, alpha=0.6, label='Теоретичні частоти', color='orange', align='edge')

# Задаємо межі для осі Y
plt.ylim(0, max(max(empirical_frequencies), max(theoretical_frequencies)) + 0.05)
plt.xticks(values)
plt.xlabel('Значення X')
plt.ylabel('Частота')
plt.title('Гістограма вибірки та теоретичні частоти')
plt.legend()
plt.grid()
plt.show()

# Виведення результатів
print("Значення вибірки:", samples)
print("Емпіричні частоти:", empirical_frequencies)
print("Теоретичні частоти:", theoretical_frequencies) # виведемо перші 10 значень вибірки, емпіричні та теоретичні частоти


# Крок 1: Генерація псевдовипадкових чисел Y
Y = np.random.uniform(0, 1, 100)  # 100 випадкових чисел від 0 до 1

# Крок 2: Обчислення значень випадкової величини X
lambda_param = 0.2  # Параметр експоненційного розподілу
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

