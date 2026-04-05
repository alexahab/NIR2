import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import os


#TODO доработать сохранение метрик в файл?
def analyze_cross_correlation(ccf_df, df_norm, alpha=0.05):
    # Количество наблюдений
    n = len(df_norm)

    # Доверительный интервал
    z_score = stats.norm.ppf(1 - alpha / 2)
    confidence_interval = z_score / np.sqrt(n)

    # Находим максимальную положительную и отрицательную корреляцию
    max_positive = ccf_df.loc[ccf_df["correlation"].idxmax()]
    max_negative = ccf_df.loc[ccf_df["correlation"].idxmin()]
    max_abs = ccf_df.loc[ccf_df["correlation"].abs().idxmax()]

    # Вывод результатов
    print("=" * 60)
    print("КРОСС-КОРРЕЛЯЦИОННЫЙ АНАЛИЗ")
    print("=" * 60)

    print(f"   Количество наблюдений: {n}")
    print(f"   Уровень значимости: {alpha}")
    print(f"   Z-оценка: {z_score:.3f}")
    print(f"   Доверительный интервал: ±{confidence_interval:.4f}")

    print(f"\n Результаты корреляции:")
    print(f"   Максимальная положительная: {max_positive['correlation']:.4f} (лаг {max_positive['lag']})")
    print(f"   Максимальная отрицательная: {max_negative['correlation']:.4f} (лаг {max_negative['lag']})")
    print(f"   Максимальная по модулю: {max_abs['correlation']:.4f} (лаг {max_abs['lag']})")

    # Оценка значимости
    print(f"\n Статистическая значимость:")
    for name, corr in [("Положительная", max_positive), ("Отрицательная", max_negative), ("По модулю", max_abs)]:
        if abs(corr['correlation']) > confidence_interval:
            print(f"   ✅ {name} корреляция значима (|{corr['correlation']:.4f}| > {confidence_interval:.4f})")
        else:
            print(f"   ⚠️ {name} корреляция не значима (|{corr['correlation']:.4f}| < {confidence_interval:.4f})")

    # Создаем датафрейм с результатами
    results_df = pd.DataFrame({
        'тип': ['положительная', 'отрицательная', 'по модулю'],
        'корреляция': [max_positive['correlation'], max_negative['correlation'], max_abs['correlation']],
        'лаг': [max_positive['lag'], max_negative['lag'], max_abs['lag']],
        'значима': [
            abs(max_positive['correlation']) > confidence_interval,
            abs(max_negative['correlation']) > confidence_interval,
            abs(max_abs['correlation']) > confidence_interval
        ]
    })

    return results_df


def plot_ccf(ccf_df, df1_name, df2_name, conf_level=None):
    plt.figure(figsize=(12, 6))

    # Стем-график
    plt.stem(ccf_df["lag"], ccf_df["correlation"], basefmt=" ")

    # Доверительный интервал
    if conf_level is None:
        # Примерное значение для 95% интервала
        conf_level = 0.2

    plt.axhline(conf_level, linestyle="--", color='red', alpha=0.7)
    plt.axhline(-conf_level, linestyle="--", color='red', alpha=0.7)

    # Настройки
    plt.xlabel("Lag (месяцы)", fontsize=12)
    plt.ylabel("Коэффициент корреляции", fontsize=12)
    plt.title(f"Кросс-корреляция: {df1_name} vs {df2_name}", fontsize=14)
    plt.grid(True, alpha=0.3)

    # Добавляем текст с порогом
    plt.text(0.02, 0.98, f'95% CI: ±{conf_level:.3f}',
             transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.show()

#TODO доработать сохранение графика в проект и его закрытие после сохранения
    # base_dir = r"C:\Users\aleks\PycharmProjects\NIR2\crossCorrelation"
    # plot_path = os.path.join(base_dir, "plots", f"Кросс-корреляция: {df1_name} vs {df2_name}.png")
    # plt.savefig(plot_path, dpi=300, bbox_inches="tight")