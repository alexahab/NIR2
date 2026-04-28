import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from scipy.stats import linregress
import warnings
import re
import os

warnings.filterwarnings('ignore')


# ============================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ============================================================

def _quarter_to_str(dt):
    """Преобразует pandas.Timestamp в строку вида '2021-Q1'."""
    quarter = (dt.month - 1) // 3 + 1
    return f"{dt.year}-Q{quarter}"


def _parse_quarter_date(quarter_str):
    """
    Парсит строки вида: 'I квартал 2021', 'II квартал 2022' и т.д.
    Возвращает pandas.Timestamp (первый месяц квартала).
    """
    s = str(quarter_str).strip()

    roman_to_int = {
        'I': 1, 'II': 2, 'III': 3, 'IV': 4,
        'І': 1, 'ІІ': 2, 'ІІІ': 3, 'ІV': 4,
    }

    pattern = r'(I{1,3}|IV|І{1,3}|ІV)\s*квартал\s*(\d{4})'
    match = re.search(pattern, s, re.IGNORECASE)

    if match:
        quarter_roman = match.group(1).upper().replace('І', 'I')
        year = int(match.group(2))
        quarter_num = roman_to_int.get(quarter_roman, None)

        if quarter_num:
            month = (quarter_num - 1) * 3 + 1
            return pd.Timestamp(year=year, month=month, day=1)

    return pd.NaT


def _fisher_g_pvalue(g, m, specific_freq=False):
    """Вычисление p-value для g-статистики теста Фишера."""
    if g <= 0 or g >= 1:
        return 1.0 if g <= 0 else 0.0

    if not specific_freq:
        p_value = m * (1 - g) ** (m - 1)
    else:
        p_value = (1 - g) ** (m - 1)

    return min(max(p_value, 0.0), 1.0)


def _fisher_g_test_core(time_series, period=4, alpha=0.05, detrend=True):
    """
    Ядро теста Фишера. Принимает numpy-массив, возвращает словарь с результатами.
    """
    y = np.array(time_series, dtype=float).flatten()
    n = len(y)

    # Удаление линейного тренда
    if detrend:
        x_trend = np.arange(n)
        slope, intercept, _, _, _ = linregress(x_trend, y)
        trend_line = slope * x_trend + intercept
        y_detrended = y - trend_line
    else:
        trend_line = np.full(n, np.mean(y))
        y_detrended = y - np.mean(y)

    # Периодограмма
    frequencies, power_spectrum = signal.periodogram(
        y_detrended, fs=1.0, window='hamming', scaling='density'
    )

    # Положительные частоты (исключая нулевую)
    pos_mask = (frequencies > 0) & (frequencies <= 0.5)
    freq_pos = frequencies[pos_mask]
    power_pos = power_spectrum[pos_mask]
    m = len(power_pos)
    total_power = np.sum(power_pos)

    # Общий g-тест
    max_power = np.max(power_pos)
    max_freq_idx = np.argmax(power_pos)
    max_freq = freq_pos[max_freq_idx]
    g_stat = max_power / total_power
    p_value_g = _fisher_g_pvalue(g_stat, m, specific_freq=False)

    # Тест на конкретную сезонную частоту
    target_freq = 1.0 / period
    closest_idx = np.argmin(np.abs(freq_pos - target_freq))
    seasonal_freq = freq_pos[closest_idx]
    seasonal_power = power_pos[closest_idx]
    w_stat = seasonal_power / total_power
    p_value_w = _fisher_g_pvalue(w_stat, m, specific_freq=True)

    return {
        'n_observations': n,
        'period': period,
        'alpha': alpha,
        'g_statistic': g_stat,
        'p_value_g': p_value_g,
        'is_any_periodicity': p_value_g < alpha,
        'dominant_period': 1 / max_freq if max_freq > 0 else np.inf,
        'w_statistic': w_stat,
        'p_value_w': p_value_w,
        'is_seasonal': p_value_w < alpha,
        'target_frequency': target_freq,
        'closest_frequency': seasonal_freq,
        'y_original': y,
        'y_detrended': y_detrended,
        'trend_line': trend_line,
        'frequencies': freq_pos,
        'power_spectrum': power_pos,
    }


def _plot_fisher_results(results, dates_pd, title_prefix=""):
    """Визуализация результатов теста Фишера."""
    period = results['period']

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'{title_prefix}Тест Фишера на сезонность (квартальные данные)',
                 fontsize=14, fontweight='bold')

    n = results['n_observations']

    # График 1: Исходный ряд
    ax1 = axes[0, 0]
    if dates_pd is not None:
        ax1.plot(dates_pd, results['y_original'], 'b-o', markersize=8, linewidth=2)
        ax1.plot(dates_pd, results['trend_line'], 'r--', linewidth=2, label='Тренд')
        ax1.tick_params(axis='x', rotation=45)
    else:
        ax1.plot(results['y_original'], 'b-o', markersize=8, linewidth=2)
        ax1.plot(results['trend_line'], 'r--', linewidth=2, label='Тренд')
    ax1.set_title('Исходный ряд с линейным трендом')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # График 2: Детрендированный ряд
    ax2 = axes[0, 1]
    if dates_pd is not None:
        ax2.plot(dates_pd, results['y_detrended'], 'g-o', markersize=8, linewidth=2)
        ax2.tick_params(axis='x', rotation=45)
    else:
        ax2.plot(results['y_detrended'], 'g-o', markersize=8, linewidth=2)
    ax2.axhline(y=0, color='red', linestyle='--', linewidth=1)
    ax2.set_title('Ряд после удаления тренда (остатки)')
    ax2.grid(True, alpha=0.3)

    # График 3: Периодограмма
    ax3 = axes[1, 0]
    freq = results['frequencies']
    power = results['power_spectrum']

    ax3.stem(freq, power, basefmt=" ", linefmt='steelblue', markerfmt='o')
    ax3.axvline(x=results['target_frequency'], color='red', linestyle='--',
                linewidth=2.5, label=f'Сезонная частота\n(период = {period} кв.)')

    # Подписываем значимые пики
    threshold = np.max(power) * 0.25
    for f, p in zip(freq, power):
        if p > threshold:
            period_val = int(round(1 / f))
            if period_val <= 20:
                ax3.annotate(f'T={period_val}кв', xy=(f, p),
                             xytext=(5, 5), textcoords="offset points",
                             fontsize=9, color='darkblue',
                             bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.7))

    ax3.set_title(f'Периодограмма\n(g-статистика = {results["g_statistic"]:.3f}, p-value = {results["p_value_g"]:.4f})')
    ax3.set_xlabel('Частота')
    ax3.set_ylabel('Спектральная мощность')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # График 4: Сезонный профиль
    ax4 = axes[1, 1]
    if results['is_seasonal']:
        quarters = np.arange(n) % period
        seasonal_profile = np.array([
            np.mean(results['y_detrended'][quarters == q]) for q in range(period)
        ])
        seasonal_profile -= np.mean(seasonal_profile)

        q_labels = ['Q1\n(янв-мар)', 'Q2\n(апр-июн)', 'Q3\n(июл-сен)', 'Q4\n(окт-дек)']
        colors = ['#2196F3', '#4CAF50', '#FF9800', '#F44336']

        bars = ax4.bar(q_labels, seasonal_profile, color=colors, alpha=0.85, edgecolor='black')
        ax4.axhline(y=0, color='black', linewidth=1)
        ax4.set_title(f'Сезонный профиль\n✅ Сезонность ЗНАЧИМА (p = {results["p_value_w"]:.4f})',
                      color='darkgreen')
        ax4.set_ylabel('Отклонение от тренда')

        for bar, val in zip(bars, seasonal_profile):
            y_pos = bar.get_height()
            offset = np.sign(val) * max(abs(seasonal_profile)) * 0.05
            ax4.text(bar.get_x() + bar.get_width() / 2., y_pos + offset,
                     f'{val:+.0f}', ha='center', fontsize=11, fontweight='bold')
    else:
        ax4.text(0.5, 0.5, '❌ Сезонность НЕ ЗНАЧИМА\nна данном уровне',
                 ha='center', va='center', fontsize=15,
                 transform=ax4.transAxes, color='red',
                 bbox=dict(boxstyle='round', facecolor='mistyrose', alpha=0.8))
        ax4.set_title('Результат теста')

    ax4.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.show()
    return fig


def _print_seasonal_indices(results):
    """Вывод сезонных индексов по кварталам."""
    n = results['n_observations']
    period = results['period']
    quarters_idx = np.arange(n) % period
    q_names = ['I квартал', 'II квартал', 'III квартал', 'IV квартал']

    print(f"\n{'─' * 70}")
    print("СЕЗОННЫЕ ИНДЕКСЫ ПО КВАРТАЛАМ")
    print(f"{'─' * 70}")

    seasonal_means = []
    for q in range(period):
        mask = quarters_idx == q
        mean_dev = np.mean(results['y_detrended'][mask])
        seasonal_means.append(mean_dev)
        original_vals = results['y_original'][mask]
        print(f"   {q_names[q]}:  откл. от тренда = {mean_dev:+.0f}  "
              f"(среднее исходное = {original_vals.mean():.0f}, "
              f"наблюдений: {len(original_vals)})")

    seasonal_means = np.array(seasonal_means)
    centered = seasonal_means - np.mean(seasonal_means)

    peak_q = np.argmax(centered)
    trough_q = np.argmin(centered)

    print(f"\n   📈 Амплитуда сезонной волны: от {centered.min():+.0f} до {centered.max():+.0f}")
    print(f"   🔺 Пик сезонности:    {q_names[peak_q]} ({centered[peak_q]:+.0f})")
    print(f"   🔻 Спад сезонности:   {q_names[trough_q]} ({centered[trough_q]:+.0f})")


# ============================================================
# ГЛАВНАЯ ФУНКЦИЯ
# ============================================================

def fisher_seasonality_test(
        file_path,
        sheet_name=0,  # 0 = первый лист
        period=4,
        alpha=0.05,
        detrend=True,
        show_plot=True
):
    """
    Проверка временного ряда на наличие сезонности с помощью g-теста Фишера.

    Файл Excel должен содержать два столбца БЕЗ заголовков:
    - первый столбец: период в формате "I квартал 2021"
    - второй столбец: числовое значение

    Параметры:
    ----------
    file_path : str — путь к Excel-файлу
    sheet_name : str/int — лист (по умолчанию 0 = первый лист)
    period : int — период сезонности (4 = квартальная, 12 = месячная)
    alpha : float — уровень значимости
    detrend : bool — удалять ли тренд
    show_plot : bool — показывать ли графики

    Возвращает:
    ----------
    dict — результаты теста
    """

    # Поиск файла
    if not os.path.exists(file_path):
        alt_paths = [
            file_path,
            os.path.join('dataExcel', file_path),
            os.path.join('../..', 'dataExcel', file_path),
        ]
        for p in alt_paths:
            if os.path.exists(p):
                file_path = p
                break
        else:
            raise FileNotFoundError(f"Файл не найден: {file_path}")

    print("=" * 70)
    print("ТЕСТ ФИШЕРА НА СЕЗОННОСТЬ")
    print("=" * 70)
    print(f"Файл: {os.path.basename(file_path)}")
    print(f"Период сезонности: {period} ({'квартальная' if period == 4 else 'месячная' if period == 12 else 'другая'})")
    print(f"Уровень значимости α: {alpha}")

    # Загрузка БЕЗ заголовков (header=None) — важно для вашего файла!
    df = pd.read_excel(file_path, sheet_name=sheet_name, header=None)
    df.columns = ['period_str', 'value']

    print(f"\nЗагружено строк: {len(df)}")

    # Парсинг дат
    df['date'] = df['period_str'].apply(_parse_quarter_date)

    # Проверка ошибок парсинга
    bad = df['date'].isna()
    if bad.any():
        print(f"⚠️ Не распознано строк: {bad.sum()}")
        df = df[~bad].reset_index(drop=True)
        print(f"Осталось строк: {len(df)}")

    if len(df) == 0:
        raise ValueError("Нет данных после очистки.")

    # Сортировка
    df = df.sort_values('date').reset_index(drop=True)

    # Данные
    y = df['value'].values.astype(float)
    dates_pd = pd.to_datetime(df['date'].values)
    n = len(y)

    print(f"\n{'─' * 70}")
    print("ВРЕМЕННОЙ РЯД")
    print(f"{'─' * 70}")
    print(f"Длина: {n} наблюдений")
    print(f"Период: {_quarter_to_str(dates_pd[0])} – {_quarter_to_str(dates_pd[-1])}")
    print(f"Среднее: {y.mean():.0f}  |  Мин: {y.min():.0f}  |  Макс: {y.max():.0f}  |  Стд.откл: {y.std():.0f}")

    # Показываем таблицу
    print(f"\nПервые 5 строк:")
    for i in range(min(5, n)):
        print(f"  {_quarter_to_str(dates_pd[i])}: {y[i]:.0f}")
    if n > 5:
        print(f"  ...")
        print(f"Последние 5 строк:")
        for i in range(max(0, n - 5), n):
            print(f"  {_quarter_to_str(dates_pd[i])}: {y[i]:.0f}")

    # Тест Фишера
    print(f"\n{'─' * 70}")
    print("ВЫПОЛНЕНИЕ ТЕСТА...")
    print(f"{'─' * 70}")

    res = _fisher_g_test_core(y, period=period, alpha=alpha, detrend=detrend)

    # Результаты
    print(f"\n{'=' * 70}")
    print("РЕЗУЛЬТАТЫ")
    print(f"{'=' * 70}")

    print(f"\n1) Общий g-тест (любая периодичность):")
    print(f"   g = {res['g_statistic']:.4f}, p-value = {res['p_value_g']:.4f}")
    if res['is_any_periodicity']:
        dp = res['dominant_period']
        print(f"   ✅ Значимая периодичность (доминирующий период: {dp:.1f} кв.)")
    else:
        print(f"   ❌ Периодичность не обнаружена")

    print(f"\n2) Тест на сезонность с периодом {period}:")
    print(f"   w = {res['w_statistic']:.4f}, p-value = {res['p_value_w']:.4f}")

    if res['is_seasonal']:
        print(f"\n   ✅✅✅ СЕЗОННОСТЬ СТАТИСТИЧЕСКИ ЗНАЧИМА ✅✅✅")
        print(f"   Внутригодовые колебания НЕ случайны (α = {alpha})")
    else:
        print(f"\n   ❌❌❌ СЕЗОННОСТЬ НЕ ЗНАЧИМА ❌❌❌")
        print(f"   Колебания могут быть случайными (α = {alpha})")

    # Сезонные индексы
    if res['is_seasonal']:
        _print_seasonal_indices(res)

    # Графики
    if show_plot:
        title_prefix = os.path.basename(file_path).replace('.xlsx', ' — ')
        _plot_fisher_results(res, dates_pd, title_prefix)

    print(f"\n{'═' * 70}")
    print("ГОТОВО")
    print(f"{'═' * 70}")

    return res