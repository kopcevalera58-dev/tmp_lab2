"""
ЛАБОРАТОРНАЯ РАБОТА: Проверка статистических гипотез о нормальности, расчет мощности и критерии согласия

ОБЩАЯ ИНСТРУКЦИЯ ДЛЯ СТУДЕНТОВ:
1. Запустите этот скрипт в вашей среде разработки (PyCharm, VS Code, IDLE).
2. Графики будут открываться в отдельных всплывающих окнах. Чтобы программа шла дальше, просто закрывайте окно с графиком.
3. Измените значение переменной MY_VARIANT на свой номер по списку.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

# ==========================================
# ЧАСТЬ 2. ИНИЦИАЛИЗАЦИЯ И ВЫБОР ВАРИАНТА
# ==========================================
# УКАЖИТЕ СВОЙ НОМЕР ВАРИАНТА НИЖЕ:
MY_VARIANT = 7  

print(f"--- Активирован вариант №{MY_VARIANT} ---")

# ==========================================
# ЗАДАНИЕ 1. Визуальное сравнение плотностей
# ==========================================
print("[Задание 1] Расчет теоретических плотностей...")
x = np.linspace(-4, 4, 1000)

pdf_norm = stats.norm.pdf(x, loc=0, scale=1)
pdf_t_20 = stats.t.pdf(x, df=20)
pdf_t_2  = stats.t.pdf(x, df=2)

plt.figure(figsize=(10, 5))
plt.plot(x, pdf_norm, label='Нормальное N(0, 1)', color='black', linewidth=2.5)
plt.plot(x, pdf_t_20, label='Стьюдента (df = 20)', color='red', linestyle='--', linewidth=2)
plt.plot(x, pdf_t_2, label='Стьюдента (df = 2)', color='blue', linestyle=':', alpha=0.05)
plt.title('Сравнение теоретических плотностей распределений')
plt.xlabel('Значение (X)')
plt.ylabel('Плотность вероятности')
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend()
print("Окно с графиком открыто. Закройте его, чтобы продолжить расчеты...")
plt.show() 

max_diff = np.max(np.abs(pdf_norm - pdf_t_20))
print(f"Максимальная разница между N(0,1) и t(df=20): {max_diff:.5f}")

# ==========================================
# ЗАДАНИЕ 2. Парадокс больших выборок
# ==========================================
print("[Задание 2] Запуск тестов на нормальность для выборок разного объема...")
np.random.seed(MY_VARIANT)
df_student = 20

samples = {
    'sample_30': stats.t.rvs(df=df_student, size=30),
    'sample_300': stats.t.rvs(df=df_student, size=300),
    'sample_5000': stats.t.rvs(df=df_student, size=5000)
}
# samples = {
#     'sample_30': stats.norm.rvs(size=30),
#     'sample_300': stats.norm.rvs(size=300),
#     'sample_5000': stats.norm.rvs(size=5000)
# }

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
results = []

for i, (name, data) in enumerate(samples.items()):
    N = len(data)
    stat_sw, p_sw = stats.shapiro(data)
    stat_dag, p_dag = stats.normaltest(data)
    dec_sw  = "H0 принимается" if p_sw  > 0.05 else "H0 ОТВЕРГАЕТСЯ"
    dec_dag = "H0 принимается" if p_dag > 0.05 else "H0 ОТВЕРГАЕТСЯ"
    results.append({
        'Выборка': name,
        'Размер (N)': N,
        'p-value (Шапиро-Уилок)': round(p_sw, 5),
        'p-value (Д\'Агостино)': round(p_dag, 5)
    })
    
    sns.histplot(data, kde=True, stat="density", color="lightgreen", ax=axes[i], alpha=0.05)
    x_plot = np.linspace(data.min(), data.max(), 100)
    axes[i].plot(x_plot, stats.norm.pdf(x_plot, loc=data.mean(), scale=data.std()), 
                 color='red', linestyle='--', linewidth=2, label='Теор. нормальное')
    axes[i].set_title(f'{name} (N = {N})\np-val SW: {p_sw:.2e}\np-val Dag: {p_dag:.2e}\n')
    axes[i].set_title(
        f'{name} (N = {N})\n'
        f'SW:  p = {p_sw:.2e}  : {dec_sw}\n'
        f'Dag: p = {p_dag:.2e}  :  {dec_dag}'
    )
    axes[i].legend()

plt.tight_layout()
print("Окно с сеткой графиков открыто. Закройте его...")
plt.show()

df_res = pd.DataFrame(results)
print(df_res.to_string(index=False))

# ==========================================
# ЗАДАНИЕ 3. Мощность критерия (Монте-Карло)
# ==========================================
# ==========================================
# ЗАДАНИЕ 3. Мощность критерия (Монте-Карло)
# ==========================================
print("\n[Задание 3] Мощность критерия Шапиро-Уилка (Монте-Карло)...")

np.random.seed(MY_VARIANT)
n_simulations = 10000
alpha = 0.05
powers =[]
for n in [30, 300, 5000]:
    reject_count = 0
    for _ in range(n_simulations):
        sample = stats.norm.rvs(size=n)
        _, p_value = stats.shapiro(sample)
        if p_value <= alpha:
            reject_count += 1
    power = reject_count / n_simulations
    powers.append(power)
    print(f"n = {n}: мощность = {power:.4f}")

plt.bar(['30', '300', '5000'], powers)
plt.title('Мощность Шапиро-Уилка (t(20))')
plt.xlabel('Объём выборки n')
plt.ylabel('Мощность')
plt.show()



# # ==========================================
# # ЗАДАНИЕ 4. Критерий согласия Хи-квадрат
# # ==========================================
# Задача: Проверить гипотезу о нормальности выборки sample_5000 с помощью критерия Хи-квадрат Пирсона, самостоятельно построив интервальный ряд и объединив малочисленные интервалы (E𝑖<5).

# Инструкция:

#     Разбейте выборку на интервалы.
#     Рассчитайте наблюдаемые частоты (𝑂𝑖) и теоретические частоты (E𝑖) через stats.norm.cdf.
#     Реализуйте алгоритм слияния интервалов, где 𝐸𝑖<5.
#     Запустите stats.chisquare с аргументом ddof=2.

      
      
# """
# КОНТРОЛЬНЫЕ ВОПРОСЫ ДЛЯ ОТЧЕТА:
# 1. Какое максимальное расхождение плотностей получилось между N(0,1) и t(df=20)? Сильно ли меняется график при df=2?
# 2. Изменилась ли внутренняя природа данных при росте n с 30 до 5000 в Задании 2? Почему тесты выдали разный ответ?
# 3. Какую ошибку (I или II рода) совершает тест Шапиро-Уилка на малых выборках?
# 4. Инструкция для аналитика: База клиентов содержит 10 000 строк. График распределения — идеальный симметричный «колокол», но тест Шапиро-Уилка выдает p-value = 0.00001. Можно ли по этой выборке из 10000 наблюдений построить доверительные интервалы для математического ожидания или дисперсии?
# 5. Зачем в Хи-квадрат тесте объединяют малочисленные интервалы и почему ddof=2?
# Как влияет на мощность критерия Хи-квадрат наличие малочисленных интервалов и неверное задание числа степеней свободы статистики критерия?

# """
