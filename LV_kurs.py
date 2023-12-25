# Импорт библиотек
import numpy as np
import matplotlib.pyplot as plt
import scipy as sci
from scipy import integrate

# Определение функции ODE
def LVC(N, t, r1, r2, alpha, beta, K1, K2):
    dN_1 = r1*N[0] * (1 - (N[0] + alpha * N[1])/ K1)
    dN_2 = r2*N[1] * (1 - (N[1] + beta * N[0])/ K2)
    return(np.array([dN_1, dN_2]))

# Значения параметров и начальные условия
r1 = 0.1
r2 = 0.1
alpha = 0.72 # Измените этот параметр, чтобы получить различные сценарии модели LV
beta = 1.25 # Измените этот параметр, чтобы получить различные сценарии модели LV
K1 = 100
K2 = 100
NI = np.array([10, 10]) # Начальные условия (начальные размеры популяции)

# Вектор времени
t = np.linspace(0, 300, 1000)

# Решение системы дифференциальных уравнений
sol = sci.integrate.odeint(LVC, NI, t, args=(r1, r2, alpha, beta, K1, K2))

# Создание диаграммы
plt.plot(t, sol[:, 0], color = "blue", label = "Конкурент 1")
plt.plot(t, sol[:, 1], color = "red", label = "Конкурент 2")
plt.legend(loc = "upper right")
plt.xlabel("Время")
plt.ylabel("Конкуренция")
plt.show()

# Фазовая диаграмма. Несколько сценариев
N_1 = np.linspace(0, np.max([K1, K2/beta])+5, 100)
N_2 = np.linspace(0, np.max([K2, K1/alpha])+5, 100)
Null_N1 = (K1 - N_1)/alpha
Null_N2 = K2 - beta * N_1

N1_max = np.max(N_1) # Максимальное значение конкурента 1
N2_max = np.max(N_2) # Максимальное значение конкурента 2

x = np.linspace(0.1, N1_max, 20) # Получение вектора со значениями "X " от 0.1 до N1_max"
y = np.linspace(0.1, N2_max, 20) # Получение вектора со значениями "X " от 0.1 до N2_max"
xx, yy = np.meshgrid(x, y) # Создание сетки с точками, где будет нарисован вектор.
NN1, NN2 = LVC((xx, yy), 0, r1, r2, alpha, beta, K1, K2) # Выведите каждую точку фазовой диаграммы (это создаст векторное поле)

norm = np.sqrt(NN1**2 + NN2**2)

NN1 = NN1 / norm
NN2 = NN2 / norm

plt.quiver(xx, yy, NN1, NN2, cmap=plt.cm.gray)
plt.plot(N_1, Null_N1, color = "blue")
plt.plot(N_1, Null_N2, color = "red")
plt.plot(sol[:, 0], sol[:, 1], color = "lime")
plt.xlim(0, N1_max)
plt.ylim(0, N2_max)
plt.title("Фазовая диаграмма - Конкурентная модель")
plt.xlabel("Конкурент 1")
plt.ylabel("Конкурент 2")
plt.show()