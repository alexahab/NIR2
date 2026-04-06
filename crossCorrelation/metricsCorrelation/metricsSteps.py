import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import os
from datetime import datetime


def analyze_cross_correlation_to_file(ccf_df, df_norm, output_file, alpha=0.05):
    """
    Анализ кросс-корреляции с сохранением в файл
    """
    n = len(df_norm)
    z_score = stats.norm.ppf(1 - alpha / 2)
    confidence_interval = z_score / np.sqrt(n)

    max_positive = ccf_df.loc[ccf_df["correlation"].idxmax()]
    max_negative = ccf_df.loc[ccf_df["correlation"].idxmin()]
    max_abs = ccf_df.loc[ccf_df["correlation"].abs().idxmax()]

    # Открываем файл для записи
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("КРОСС-КОРРЕЛЯЦИОННЫЙ АНАЛИЗ\n")
        f.write(f"Дата отчета: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 60 + "\n\n")

        f.write(f"Количество наблюдений: {n}\n")
        f.write(f"Уровень значимости: {alpha}\n")
        f.write(f"Z-оценка: {z_score:.3f}\n")
        f.write(f"Доверительный интервал: ±{confidence_interval:.4f}\n\n")

        f.write("Результаты корреляции:\n")
        f.write(f"  Максимальная положительная: {max_positive['correlation']:.4f} (лаг {max_positive['lag']})\n")
        f.write(f"  Максимальная отрицательная: {max_negative['correlation']:.4f} (лаг {max_negative['lag']})\n")
        f.write(f"  Максимальная по модулю: {max_abs['correlation']:.4f} (лаг {max_abs['lag']})\n\n")

        f.write("Статистическая значимость:\n")
        for name, corr in [("Положительная", max_positive), ("Отрицательная", max_negative), ("По модулю", max_abs)]:
            if abs(corr['correlation']) > confidence_interval:
                f.write(f"  ✅ {name} корреляция значима (|{corr['correlation']:.4f}| > {confidence_interval:.4f})\n")
            else:
                f.write(
                    f"  ⚠️ {name} корреляция не значима (|{corr['correlation']:.4f}| < {confidence_interval:.4f})\n")

        # Добавляем все лаги с корреляциями
        f.write("\n" + "=" * 60 + "\n")
        f.write("ПОЛНАЯ ТАБЛИЦА КОРРЕЛЯЦИЙ\n")
        f.write("=" * 60 + "\n")
        f.write(ccf_df.to_string(index=False))

    print(f"✅ Отчет сохранен в файл: {output_file}")
    return output_file



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
    # plot_path = os.path.join(base_dir, "plots", f"name: {df1_name} vs {df2_name}.png")
    # plt.savefig(plot_path, dpi=300, bbox_inches="tight")