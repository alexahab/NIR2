import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
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
    """Парсит строки вида: 'I квартал 2021', 'II квартал 2022' и т.д."""
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


def _hegy_critical_values(n, trend='const', alpha=0.05):
    """
    Критические значения для HEGY-теста.
    Аппроксимация на основе таблиц Hylleberg, Engle, Granger, Yoo (1990).

    Для точных значений при малых выборках используются интерполяции.
    """
    # Базовые асимптотические критические значения (n → ∞)
    if trend == 'const':
        crit = {
            'pi1': -2.86,  # нулевая частота (тренд)
            'pi2': -2.86,  # годовая сезонность
            'pi3': -2.86,  # полугодовая сезонность
            'pi4': 3.08,  # F-тест для π₃ ∩ π₄ (комплексная часть)
            'pi2_pi4': 5.99,  # F-тест для π₂ ∩ π₄
        }
    elif trend == 'trend':
        crit = {
            'pi1': -3.41,
            'pi2': -2.86,
            'pi3': -3.41,
            'pi4': 3.08,
            'pi2_pi4': 5.99,
        }
    elif trend == 'none':
        crit = {
            'pi1': -1.95,
            'pi2': -1.95,
            'pi3': -1.95,
            'pi4': 3.08,
            'pi2_pi4': 5.99,
        }

    # Корректировка для малых выборок
    if n < 100:
        correction = 1.0 + (100 - n) / 200
        for key in crit:
            if key in ['pi1', 'pi2', 'pi3']:
                crit[key] *= correction

    return crit


def hegy_test(y, period=4, max_lags=None, trend='const'):
    """
    HEGY-тест на сезонные единичные корни.

    Параметры:
    ----------
    y : array-like
        Временной ряд.
    period : int
        Период сезонности (4 для квартальных данных).
    max_lags : int or None
        Максимальное количество лагов. Если None, определяется автоматически.
    trend : str
        'const' — константа,
        'trend' — константа + тренд,
        'none' — без детерминированных компонент.

    Возвращает:
    ----------
    dict : результаты теста.
    """
    y = np.array(y, dtype=float).flatten()
    n = len(y)

    if period != 4:
        raise NotImplementedError("Пока реализован только для квартальных данных (period=4)")

    # Сезонная разность Δ₄ y_t
    dy4 = y[period:] - y[:-period]
    n_eff = len(dy4)

    # Формируем преобразованные переменные
    y1 = np.zeros(n)
    y2 = np.zeros(n)
    y3 = np.zeros(n)

    for t in range(n):
        if t >= 0:
            y1[t] = y[t]
        if t >= 1:
            y1[t] += y[t - 1]
        if t >= 2:
            y1[t] += y[t - 2]
        if t >= 3:
            y1[t] += y[t - 3]

        if t >= 0:
            y2[t] = -y[t]
        if t >= 1:
            y2[t] += y[t - 1]
        if t >= 2:
            y2[t] += -y[t - 2]
        if t >= 3:
            y2[t] += y[t - 3]

        if t >= 0:
            y3[t] = -y[t]
        if t >= 2:
            y3[t] += y[t - 2]

    # Определяем количество лагов
    if max_lags is None:
        max_lags = min(int(4 * (n / 100) ** 0.25), n_eff // 2 - 1)
        max_lags = max(0, max_lags)

    # Построение регрессии
    # Зависимая переменная: dy4
    Y = dy4[max_lags:] if max_lags > 0 else dy4

    # Формируем матрицу регрессоров
    start_idx = period + max_lags

    X_list = []
    col_names = []

    # Детерминированные компоненты
    if trend in ['const', 'trend']:
        const = np.ones(n_eff - max_lags)
        X_list.append(const)
        col_names.append('const')

    if trend == 'trend':
        t = np.arange(start_idx, n)
        X_list.append(t - t[0] + 1)
        col_names.append('trend')

    # Сезонные дамми
    if trend in ['const', 'trend']:
        for q in range(1, period):
            dummy = np.zeros(n_eff - max_lags)
            for i in range(len(dummy)):
                if (start_idx + i) % period == q:
                    dummy[i] = 1.0
            X_list.append(dummy)
            col_names.append(f'seasonal_q{q}')

    # Преобразованные переменные (сдвинутые на period)
    y1_lag = y1[start_idx - 1:n - 1] if max_lags == 0 else y1[start_idx - 1:start_idx - 1 + len(Y)]
    y2_lag = y2[start_idx - 1:n - 1] if max_lags == 0 else y2[start_idx - 1:start_idx - 1 + len(Y)]
    y3_lag_1 = y3[start_idx - 1:n - 1] if max_lags == 0 else y3[start_idx - 1:start_idx - 1 + len(Y)]
    y3_lag_2 = y3[start_idx - 2:n - 2] if max_lags == 0 else y3[start_idx - 2:start_idx - 2 + len(Y)]

    X_list.extend([y1_lag, y2_lag, y3_lag_1, y3_lag_2])
    col_names.extend(['pi1', 'pi2', 'pi3', 'pi4'])

    # Лаги сезонной разности
    if max_lags > 0:
        for lag in range(1, max_lags + 1):
            lagged = dy4[max_lags - lag:max_lags - lag + len(Y)]
            X_list.append(lagged)
            col_names.append(f'dy4_lag{lag}')

    X = np.column_stack(X_list)

    # Оцениваем OLS
    model = sm.OLS(Y, X)
    results = model.fit()

    # Извлекаем статистики
    t_pi1 = results.tvalues[col_names.index('pi1')]
    t_pi2 = results.tvalues[col_names.index('pi2')]
    t_pi3 = results.tvalues[col_names.index('pi3')]
    t_pi4 = results.tvalues[col_names.index('pi4')]

    # F-тесты
    # F-тест для π₃ = π₄ = 0
    r_matrix = np.zeros((2, len(col_names)))
    r_matrix[0, col_names.index('pi3')] = 1
    r_matrix[1, col_names.index('pi4')] = 1
    f_pi3_pi4 = results.f_test(r_matrix)

    # F-тест для π₂ = π₄ = 0
    r_matrix2 = np.zeros((2, len(col_names)))
    r_matrix2[0, col_names.index('pi2')] = 1
    r_matrix2[1, col_names.index('pi4')] = 1
    f_pi2_pi4 = results.f_test(r_matrix2)

    # Критические значения
    crit = _hegy_critical_values(n, trend, alpha=0.05)

    return {
        'n_observations': n,
        'period': period,
        'max_lags': max_lags,
        'trend': trend,
        't_pi1': t_pi1,
        't_pi2': t_pi2,
        't_pi3': t_pi3,
        't_pi4': t_pi4,
        'f_pi3_pi4': f_pi3_pi4.fvalue,
        'f_pi3_pi4_pvalue': f_pi3_pi4.pvalue,
        'f_pi2_pi4': f_pi2_pi4.fvalue,
        'f_pi2_pi4_pvalue': f_pi2_pi4.pvalue,
        'critical_values': crit,
        'results': results,
    }


# ============================================================
# ГЛАВНАЯ ФУНКЦИЯ
# ============================================================

def hegy_seasonality_test(
        file_path,
        sheet_name=0,
        period=4,
        alpha=0.05,
        max_lags=None,
        trend='const',
        show_plot=True
):
    """
    HEGY-тест на сезонные единичные корни для квартальных данных.

    Тест проверяет тип сезонности:
    - Детерминированная (стабильная, не меняется год от года)
    - Стохастическая (меняется случайным образом)

    Параметры:
    ----------
    file_path : str
        Путь к Excel-файлу с данными.
        Файл должен содержать два столбца БЕЗ заголовков:
        первый — период ("I квартал 2021"), второй — числовое значение.

    sheet_name : str/int
        Лист в Excel (по умолчанию 0 = первый).

    period : int
        Период сезонности (4 для квартальных данных).

    alpha : float
        Уровень значимости (по умолчанию 0.05).

    max_lags : int or None
        Количество лагов в тесте. Если None — автоматически.

    trend : str
        'const' — константа,
        'trend' — константа + тренд,
        'none' — без детерминированных компонент.

    show_plot : bool
        Показывать ли графики.

    Возвращает:
    ----------
    dict : результаты теста.
    """

    # Поиск файла
    if not os.path.exists(file_path):
        alt_paths = [
            file_path,
            os.path.join('dataExcel', file_path),
            os.path.join('..', 'dataExcel', file_path),
        ]
        for p in alt_paths:
            if os.path.exists(p):
                file_path = p
                break
        else:
            raise FileNotFoundError(f"Файл не найден: {file_path}")

    print("=" * 70)
    print("HEGY-ТЕСТ НА СЕЗОННЫЕ ЕДИНИЧНЫЕ КОРНИ")
    print("=" * 70)
    print(f"Файл: {os.path.basename(file_path)}")
    print(f"Период: {period} (квартальные данные)")
    print(f"Уровень значимости α: {alpha}")
    print(f"Детерминированные компоненты: {trend}")

    # Загрузка данных
    df = pd.read_excel(file_path, sheet_name=sheet_name, header=None)
    df.columns = ['period_str', 'value']

    # Парсинг дат
    df['date'] = df['period_str'].apply(_parse_quarter_date)
    bad = df['date'].isna()
    if bad.any():
        print(f"⚠️ Не распознано строк: {bad.sum()}")
        df = df[~bad].reset_index(drop=True)

    df = df.sort_values('date').reset_index(drop=True)

    y = df['value'].values.astype(float)
    dates_pd = pd.to_datetime(df['date'].values)
    n = len(y)

    print(f"\n{'─' * 70}")
    print("ВРЕМЕННОЙ РЯД")
    print(f"{'─' * 70}")
    print(f"Длина: {n} наблюдений")
    print(f"Период: {_quarter_to_str(dates_pd[0])} – {_quarter_to_str(dates_pd[-1])}")
    print(f"Среднее: {y.mean():.0f}  |  Мин: {y.min():.0f}  |  Макс: {y.max():.0f}")

    # Таблица данных
    print(f"\nПервые 4 строки:")
    for i in range(min(4, n)):
        print(f"  {_quarter_to_str(dates_pd[i])}: {y[i]:.0f}")
    print(f"  ...")
    print(f"Последние 4 строки:")
    for i in range(max(0, n - 4), n):
        print(f"  {_quarter_to_str(dates_pd[i])}: {y[i]:.0f}")

    # Выполнение теста
    print(f"\n{'─' * 70}")
    print("ВЫПОЛНЕНИЕ HEGY-ТЕСТА...")
    print(f"{'─' * 70}")

    result = hegy_test(y, period=period, max_lags=max_lags, trend=trend)
    crit = result['critical_values']

    # ============================================================
    # ВЫВОД РЕЗУЛЬТАТОВ
    # ============================================================

    print(f"\n{'=' * 70}")
    print("РЕЗУЛЬТАТЫ HEGY-ТЕСТА")
    print(f"{'=' * 70}")

    print(f"\nИспользовано лагов: {result['max_lags']}")

    # π₁ — нулевая частота (нестационарность в уровне / тренд)
    print(f"\n{'─' * 70}")
    print("1. π₁ — НУЛЕВАЯ ЧАСТОТА (НЕСТАЦИОНАРНОСТЬ / ЕДИНИЧНЫЙ КОРЕНЬ)")
    print(f"{'─' * 70}")
    print(f"   t-статистика: {result['t_pi1']:.3f}")
    print(f"   Критическое значение (α={alpha}): {crit['pi1']:.3f}")
    if result['t_pi1'] < crit['pi1']:
        print(f"   ✅ Единичный корень ОТВЕРГАЕТСЯ → ряд стационарен в уровне")
    else:
        print(f"   ❌ Единичный корень НЕ ОТВЕРГАЕТСЯ → ряд нестационарен (есть тренд)")

    # π₂ — годовая сезонная частота (ОСНОВНОЙ РЕЗУЛЬТАТ)
    print(f"\n{'─' * 70}")
    print("2. π₂ — ГОДОВАЯ СЕЗОННАЯ ЧАСТОТА (ОСНОВНОЙ ТЕСТ НА СЕЗОННОСТЬ)")
    print(f"{'─' * 70}")
    print(f"   t-статистика: {result['t_pi2']:.3f}")
    print(f"   Критическое значение (α={alpha}): {crit['pi2']:.3f}")

    if result['t_pi2'] < crit['pi2']:
        print(f"   ✅✅✅ Сезонный единичный корень ОТВЕРГАЕТСЯ ✅✅✅")
        print(f"   Это означает: СЕЗОННОСТЬ ДЕТЕРМИНИРОВАННАЯ (СТАБИЛЬНАЯ)")
        print(f"   Сезонный паттерн устойчив год от года.")
        print(f"   Рекомендация: использовать сезонные дамми-переменные.")
        pi2_rejected = True
    else:
        print(f"   ❌ Сезонный единичный корень НЕ ОТВЕРГАЕТСЯ")
        print(f"   Это означает: сезонность СТОХАСТИЧЕСКАЯ или ОТСУТСТВУЕТ")
        print(f"   Сезонный паттерн меняется год от года случайным образом.")
        print(f"   Рекомендация: использовать сезонное дифференцирование.")
        pi2_rejected = False

    # π₃ — полугодовая частота
    print(f"\n{'─' * 70}")
    print("3. π₃ — ПОЛУГОДОВАЯ ЧАСТОТА")
    print(f"{'─' * 70}")
    print(f"   t-статистика: {result['t_pi3']:.3f}")
    print(f"   Критическое значение (α={alpha}): {crit['pi3']:.3f}")
    if result['t_pi3'] < crit['pi3']:
        print(f"   ✅ Полугодовой единичный корень ОТВЕРГАЕТСЯ → стабильный полугодовой паттерн")
    else:
        print(f"   ❌ Полугодовой единичный корень НЕ ОТВЕРГАЕТСЯ")

    # π₄ — комплексная часть
    print(f"\n{'─' * 70}")
    print("4. π₄ — КОМПЛЕКСНАЯ ЧАСТЬ ГОДОВОЙ СЕЗОННОСТИ")
    print(f"{'─' * 70}")
    print(f"   t-статистика для π₄: {result['t_pi4']:.3f}")

    # F-тест π₃ = π₄ = 0
    print(f"\n{'─' * 70}")
    print("5. F-ТЕСТ: π₃ = π₄ = 0 (полугодовая + комплексная)")
    print(f"{'─' * 70}")
    print(f"   F-статистика: {result['f_pi3_pi4']:.3f}")
    print(f"   p-value: {result['f_pi3_pi4_pvalue']:.4f}")
    print(f"   Критическое значение (α={alpha}): {crit['pi4']:.3f}")
    if result['f_pi3_pi4'] > crit['pi4']:
        print(f"   ✅ Гипотеза π₃ = π₄ = 0 ОТВЕРГАЕТСЯ → есть полугодовая или годовая динамика")
    else:
        print(f"   ❌ Гипотеза π₃ = π₄ = 0 НЕ ОТВЕРГАЕТСЯ")

    # F-тест π₂ = π₄ = 0
    print(f"\n{'─' * 70}")
    print("6. F-ТЕСТ: π₂ = π₄ = 0 (ГОДОВАЯ СЕЗОННОСТЬ В ЦЕЛОМ)")
    print(f"{'─' * 70}")
    print(f"   F-статистика: {result['f_pi2_pi4']:.3f}")
    print(f"   p-value: {result['f_pi2_pi4_pvalue']:.4f}")
    print(f"   Критическое значение (α={alpha}): {crit['pi2_pi4']:.3f}")
    if result['f_pi2_pi4'] > crit['pi2_pi4']:
        print(f"   ✅ Гипотеза π₂ = π₄ = 0 ОТВЕРГАЕТСЯ → СЕЗОННОСТЬ ЗНАЧИМА")
        seasonal_significant = True
    else:
        print(f"   ❌ Гипотеза π₂ = π₄ = 0 НЕ ОТВЕРГАЕТСЯ → недостаточно доказательств сезонности")
        seasonal_significant = False

    # ============================================================
    # ИТОГОВОЕ ЗАКЛЮЧЕНИЕ
    # ============================================================

    print(f"\n{'═' * 70}")
    print("ИТОГОВОЕ ЗАКЛЮЧЕНИЕ")
    print(f"{'═' * 70}")

    if pi2_rejected and seasonal_significant:
        print(f"\n   ✅✅✅ ВЫВОД: СЕЗОННОСТЬ СТАТИСТИЧЕСКИ ЗНАЧИМА ✅✅✅")
        print(f"   Тип сезонности: ДЕТЕРМИНИРОВАННАЯ (СТАБИЛЬНАЯ)")
        print(f"   Это означает:")
        print(f"   • сезонный паттерн устойчив и повторяется из года в год")
        print(f"   • амплитуда сезонных колебаний постоянна")
        print(f"   • для моделирования нужно использовать сезонные дамми-переменные")
    elif not pi2_rejected:
        print(f"\n   ❌ ВЫВОД: СЕЗОННОСТЬ НЕ ЗНАЧИМА (СТОХАСТИЧЕСКАЯ)")
        print(f"   Это означает:")
        print(f"   • сезонный паттерн НЕСТАБИЛЕН и меняется год от года")
        print(f"   • или сезонность вовсе отсутствует")
        print(f"   • для моделирования нужно использовать сезонное дифференцирование")
    else:
        print(f"\n   ⚠️ ВЫВОД: РЕЗУЛЬТАТ НЕОДНОЗНАЧЕН")
        print(f"   π₂ отвергается (p < {alpha}), но F-тест π₂=π₄=0 не значим.")
        print(f"   Сезонный паттерн, возможно, присутствует, но недостаточно выражен.")

    # Сравнение с тестом Фишера
    print(f"\n{'─' * 70}")
    print("СРАВНЕНИЕ С G-ТЕСТОМ ФИШЕРА")
    print(f"{'─' * 70}")
    print(f"   G-тест Фишера: проверяет, есть ли ЛЮБАЯ периодичность.")
    print(f"   HEGY-тест:      проверяет, КАКОГО ТИПА сезонность (стабильная или меняющаяся).")
    print(f"   Оба теста дополняют друг друга!")

    # Графики
    if show_plot:
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'{os.path.basename(file_path)} — HEGY-тест на сезонность',
                     fontsize=14, fontweight='bold')

        # График 1: Исходный ряд
        ax1 = axes[0, 0]
        ax1.plot(dates_pd, y, 'b-o', markersize=8, linewidth=2)
        ax1.set_title('Исходный ряд')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3)

        # График 2: Сезонная разность Δ₄
        ax2 = axes[0, 1]
        dy4_plot = y[period:] - y[:-period]
        ax2.plot(dates_pd[period:], dy4_plot, 'g-o', markersize=8, linewidth=2)
        ax2.axhline(y=0, color='red', linestyle='--', linewidth=1)
        ax2.set_title('Сезонная разность (Δ₄ y_t = y_t - y_{t-4})')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)

        # График 3: Результаты теста
        ax3 = axes[1, 0]
        ax3.axis('off')

        summary_text = f"""
        РЕЗУЛЬТАТЫ HEGY-ТЕСТА:

        π₁ (тренд):     t = {result['t_pi1']:.3f}
                         crit = {crit['pi1']:.3f}
                         {'✓ стационарен' if result['t_pi1'] < crit['pi1'] else '✗ нестационарен'}

        π₂ (год. сез.):  t = {result['t_pi2']:.3f}
                         crit = {crit['pi2']:.3f}
                         {'✓ ДЕТЕРМИНИРОВАННАЯ' if pi2_rejected else '✗ стохастическая/отсутствует'}

        π₃ (полугод.):   t = {result['t_pi3']:.3f}
                         crit = {crit['pi3']:.3f}

        F(π₃=π₄=0):     F = {result['f_pi3_pi4']:.3f}
                         crit = {crit['pi4']:.3f}

        F(π₂=π₄=0):     F = {result['f_pi2_pi4']:.3f}
                         crit = {crit['pi2_pi4']:.3f}

        ВЫВОД: {'СЕЗОННОСТЬ ЗНАЧИМА (детерминированная)' if (pi2_rejected and seasonal_significant)
        else 'СЕЗОННОСТЬ НЕ ЗНАЧИМА'}
        """
        ax3.text(0.1, 0.5, summary_text, transform=ax3.transAxes,
                 fontsize=11, verticalalignment='center', fontfamily='monospace',
                 bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

        # График 4: Сезонный профиль (если значим)
        ax4 = axes[1, 1]
        if pi2_rejected and seasonal_significant:
            quarters = np.arange(n) % period
            seasonal_profile = np.array([np.mean(y[quarters == q]) for q in range(period)])
            seasonal_profile_norm = seasonal_profile - np.mean(seasonal_profile)

            q_labels = ['Q1', 'Q2', 'Q3', 'Q4']
            colors = ['#2196F3', '#4CAF50', '#FF9800', '#F44336']

            bars = ax4.bar(q_labels, seasonal_profile_norm, color=colors, alpha=0.85, edgecolor='black')
            ax4.axhline(y=0, color='black', linewidth=1)
            ax4.set_title('Сезонный профиль\n(ДЕТЕРМИНИРОВАННАЯ сезонность)', color='darkgreen')
            ax4.set_ylabel('Отклонение от среднего')

            for bar, val in zip(bars, seasonal_profile_norm):
                ax4.text(bar.get_x() + bar.get_width() / 2.,
                         bar.get_height() + np.sign(val) * max(abs(seasonal_profile_norm)) * 0.05,
                         f'{val:+.0f}', ha='center', fontsize=11, fontweight='bold')
        else:
            ax4.text(0.5, 0.5, 'Сезонность\nне значима\n(стохастическая или отсутствует)',
                     ha='center', va='center', fontsize=14, transform=ax4.transAxes, color='red',
                     bbox=dict(boxstyle='round', facecolor='mistyrose', alpha=0.8))
            ax4.set_title('Сезонный профиль не построен')

        ax4.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.show()

    print(f"\n{'═' * 70}")
    print("ГОТОВО")
    print(f"{'═' * 70}")

    return result