import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm
import warnings
import re
import os

warnings.filterwarnings('ignore')


# ============================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ============================================================

"""
const - Изменение среднего уровня       -----------> (реализована)
trend - Изменение среднего и тренда одновременно	   -----------> (реализована)
mean - Только изменение среднего (без тренда в сегментах)
break_mean - Изменение среднего при общем тренде
break_trend - Изменение наклона тренда
full - Полная модель: меняется и среднее, и тренд
"""

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


def _compute_break_date(dates, break_index):
    """Вычисляет дату сдвига (середину между двумя соседними точками)."""
    if break_index is None or break_index <= 0 or break_index >= len(dates):
        return None

    d1 = pd.Timestamp(dates[break_index - 1])
    d2 = pd.Timestamp(dates[break_index])
    mid = d1 + (d2 - d1) / 2
    return mid


# ============================================================
# ЯДРО: АЛГОРИТМ БАЙ-ПЕРРОНА (исправленная версия)
# ============================================================

def _compute_ssr_segment(y_segment, t_segment, model_type):
    """Вычисляет SSR для одного сегмента."""
    if len(y_segment) < 2:
        return 0.0

    if model_type == 'const':
        mean_val = np.mean(y_segment)
        resid = y_segment - mean_val
    else:  # trend
        X = np.column_stack([np.ones(len(y_segment)), t_segment - t_segment[0]])
        try:
            model = sm.OLS(y_segment, X).fit()
            resid = model.resid
        except:
            mean_val = np.mean(y_segment)
            resid = y_segment - mean_val

    return np.sum(resid ** 2)


def _compute_rss_with_breaks(y, breaks, model_type):
    """Вычисляет общую RSS для заданного набора точек сдвига."""
    breaks = sorted(breaks)
    n = len(y)
    t = np.arange(n)

    segments = [0] + breaks + [n]
    total_rss = 0.0

    for i in range(len(segments) - 1):
        start = segments[i]
        end = segments[i + 1]
        y_seg = y[start:end]
        t_seg = t[start:end]
        total_rss += _compute_ssr_segment(y_seg, t_seg, model_type)

    return total_rss


def _find_optimal_breaks_brute_force(y, m, min_segment, model_type):
    """
    Поиск оптимальных точек сдвига полным перебором
    (для коротких рядов это приемлемо).
    """
    n = len(y)
    t = np.arange(n)

    # Предвычисляем SSR для всех возможных сегментов
    ssr_matrix = np.full((n + 1, n + 1), np.inf)
    for i in range(n):
        for j in range(i + min_segment, n + 1):
            if j - i >= min_segment:
                ssr_matrix[i, j] = _compute_ssr_segment(y[i:j], t[i:j], model_type)

    # Перебираем все комбинации из m точек сдвига
    best_rss = np.inf
    best_breaks = None

    # Точки сдвига — индексы, после которых начинается новый сегмент
    # Перебираем m точек из диапазона [min_segment, n - min_segment]
    from itertools import combinations

    possible_breaks = list(range(min_segment, n - min_segment + 1))

    if len(possible_breaks) < m:
        return None

    for combo in combinations(possible_breaks, m):
        # Проверяем, что все сегменты достаточно длинные
        breaks = [0] + list(combo) + [n]
        valid = True
        for i in range(len(breaks) - 1):
            if breaks[i + 1] - breaks[i] < min_segment:
                valid = False
                break

        if not valid:
            continue

        # Вычисляем общую RSS
        total_rss = 0.0
        for i in range(len(breaks) - 1):
            start = breaks[i]
            end = breaks[i + 1]
            total_rss += ssr_matrix[start, end]

        if total_rss < best_rss:
            best_rss = total_rss
            best_breaks = list(combo)

    return best_breaks


def bai_perron_test(y, max_breaks=3, trim=0.15, model_type='const'):
    """
    Алгоритм Бай-Перрона для обнаружения структурных сдвигов
    (исправленная версия с полным перебором).

    Параметры:
    ----------
    y : array-like — временной ряд.
    max_breaks : int — максимальное количество сдвигов для проверки.
    trim : float — минимальная доля наблюдений в сегменте.
    model_type : str — 'const' или 'trend'.

    Возвращает:
    ----------
    dict : результаты теста.
    """
    y = np.array(y, dtype=float).flatten()
    n = len(y)
    min_segment = max(2, int(n * trim))
    t = np.arange(n)

    results = {
        'n': n,
        'max_breaks': max_breaks,
        'trim': trim,
        'min_segment': min_segment,
        'model_type': model_type,
        'rss_by_m': {},
        'breaks_by_m': {},
        'bic_by_m': {},
        'optimal_m_bic': 0,
        'optimal_breaks_bic': [],
        'supF_stats': {},
        'supF_pvalues': {},
    }

    # Базовая модель без сдвигов (m = 0)
    if model_type == 'const':
        X_base = np.column_stack([np.ones(n)])
    else:
        X_base = np.column_stack([np.ones(n), t])

    model_base = sm.OLS(y, X_base).fit()
    rss_base = np.sum(model_base.resid ** 2)
    results['rss_by_m'][0] = rss_base

    k_base = X_base.shape[1]
    bic_base = n * np.log(rss_base / n) + k_base * np.log(n)
    results['bic_by_m'][0] = bic_base

    # Ищем оптимальные точки для m = 1, 2, ..., max_breaks
    for m in range(1, max_breaks + 1):
        breaks_found = _find_optimal_breaks_brute_force(y, m, min_segment, model_type)

        if breaks_found is not None:
            rss_m = _compute_rss_with_breaks(y, breaks_found, model_type)
            results['rss_by_m'][m] = rss_m
            results['breaks_by_m'][m] = breaks_found

            if model_type == 'const':
                k_m = (m + 1) + m  # константы + точки сдвига
            else:
                k_m = 2 * (m + 1) + m

            bic_m = n * np.log(rss_m / n) + k_m * np.log(n)
            results['bic_by_m'][m] = bic_m

    # Оптимальное m по BIC
    if len(results['bic_by_m']) > 0:
        optimal_m = min(results['bic_by_m'], key=results['bic_by_m'].get)
        results['optimal_m_bic'] = optimal_m
        results['optimal_breaks_bic'] = results['breaks_by_m'].get(optimal_m, [])

    # SupF-тесты
    for m in range(1, max_breaks + 1):
        if m in results['rss_by_m'] and 0 in results['rss_by_m']:
            rss_restricted = results['rss_by_m'][0]
            rss_unrestricted = results['rss_by_m'][m]

            if model_type == 'const':
                q = m
                k_unrestricted = (m + 1)
            else:
                q = 2 * m
                k_unrestricted = 2 * (m + 1)

            if n > k_unrestricted + q and rss_unrestricted > 0:
                F = ((rss_restricted - rss_unrestricted) / q) / (rss_unrestricted / (n - k_unrestricted))
                p_value = 1 - stats.f.cdf(F, q, n - k_unrestricted)
                results['supF_stats'][m] = F
                results['supF_pvalues'][m] = p_value

    return results


# ============================================================
# ВИЗУАЛИЗАЦИЯ
# ============================================================

def _plot_bai_perron_results(results, y, dates_pd, file_name):
    """Визуализация результатов теста Бай-Перрона."""

    m_opt = results['optimal_m_bic']
    breaks_idx = results['optimal_breaks_bic']
    n = len(y)

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(f'{file_name} — Тест Бай-Перрона на структурные сдвиги',
                 fontsize=14, fontweight='bold')

    # График 1: Исходный ряд с отмеченными сдвигами
    ax1 = axes[0, 0]
    ax1.plot(dates_pd, y, 'b-o', markersize=8, linewidth=2, label='Исходный ряд')

    segments = [0] + (breaks_idx if breaks_idx else []) + [n]
    colors_line = ['#2196F3', '#4CAF50', '#FF9800', '#F44336', '#9C27B0', '#00BCD4']

    for i in range(len(segments) - 1):
        start = segments[i]
        end = segments[i + 1]
        mean_val = np.mean(y[start:end])
        seg_dates = dates_pd[start:end]
        color = colors_line[i % len(colors_line)]
        ax1.hlines(y=mean_val, xmin=seg_dates[0], xmax=seg_dates[-1],
                   colors=color, linewidth=3, alpha=0.7,
                   label=f'Среднее сегм. {i + 1}: {mean_val:.0f}')

    # Вертикальные линии для сдвигов
    for br in breaks_idx:
        if br > 0 and br < n:
            d1 = pd.Timestamp(dates_pd[br - 1])
            d2 = pd.Timestamp(dates_pd[br])
            mid = d1 + (d2 - d1) / 2
            ax1.axvline(x=mid, color='red', linestyle='--', linewidth=2.5, alpha=0.8)

    ax1.set_title(f'Исходный ряд ({m_opt} структурных сдвигов)')
    ax1.tick_params(axis='x', rotation=45)
    ax1.legend(fontsize=8, loc='upper left')
    ax1.grid(True, alpha=0.3)

    # График 2: BIC
    ax2 = axes[0, 1]
    m_values = sorted(results['bic_by_m'].keys())
    bic_values = [results['bic_by_m'][m] for m in m_values]

    colors_bar = ['#F44336' if m == m_opt else '#2196F3' for m in m_values]
    bars = ax2.bar(m_values, bic_values, color=colors_bar, alpha=0.8, edgecolor='black')
    ax2.set_title(f'Информационный критерий BIC\n(оптимум: m = {m_opt} сдвигов)')
    ax2.set_xlabel('Количество сдвигов (m)')
    ax2.set_ylabel('BIC')
    ax2.set_xticks(m_values)
    ax2.grid(True, alpha=0.3, axis='y')

    for bar, bic in zip(bars, bic_values):
        ax2.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 1,
                 f'{bic:.1f}', ha='center', fontsize=9)

    # График 3: SupF-тесты
    ax3 = axes[1, 0]
    if results['supF_stats']:
        m_f = sorted(results['supF_stats'].keys())
        f_values = [results['supF_stats'][m] for m in m_f]
        p_values = [results['supF_pvalues'][m] for m in m_f]

        colors_f = ['#F44336' if p < 0.05 else '#4CAF50' for p in p_values]
        bars = ax3.bar(m_f, f_values, color=colors_f, alpha=0.8, edgecolor='black')
        ax3.axhline(y=3.0, color='red', linestyle='--', linewidth=1.5,
                    label='Критич. F (α=0.05) ≈ 3.0')
        ax3.set_title('SupF-тесты (H₀: 0 сдвигов vs H₁: m сдвигов)')
        ax3.set_xlabel('Количество сдвигов (m)')
        ax3.set_ylabel('F-статистика')
        ax3.legend(fontsize=8)
        ax3.grid(True, alpha=0.3, axis='y')

        for bar, f, p in zip(bars, f_values, p_values):
            ax3.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.1,
                     f'p={p:.3f}', ha='center', fontsize=8)
    else:
        ax3.text(0.5, 0.5, 'SupF-тесты недоступны',
                 ha='center', va='center', fontsize=12, transform=ax3.transAxes)
        ax3.set_title('SupF-тесты')

    # График 4: Таблица результатов
    ax4 = axes[1, 1]
    ax4.axis('off')

    summary_lines = [
        f"РЕЗУЛЬТАТЫ ТЕСТА БАЙ-ПЕРРОНА",
        f"",
        f"Оптимальное количество сдвигов (BIC): m = {m_opt}",
        f"",
    ]

    if m_opt > 0 and breaks_idx:
        summary_lines.append(f"Структурные сдвиги:")
        for i, br in enumerate(breaks_idx):
            d = _compute_break_date(dates_pd, br)
            if d is not None:
                date_str = _quarter_to_str(d)
                summary_lines.append(f"  Сдвиг {i + 1}: {date_str}")

        summary_lines.append(f"")
        segments_full = [0] + breaks_idx + [n]
        for i in range(len(segments_full) - 1):
            start = segments_full[i]
            end = segments_full[i + 1]
            seg_mean = np.mean(y[start:end])
            s_date = _quarter_to_str(dates_pd[start])
            e_date = _quarter_to_str(dates_pd[end - 1])
            summary_lines.append(f"  Сегм. {i + 1}: {s_date}–{e_date}, среднее={seg_mean:.0f}")

    summary_lines.append(f"")
    summary_lines.append(f"BIC(0) = {results['bic_by_m'][0]:.1f}")
    if m_opt > 0:
        summary_lines.append(f"BIC({m_opt}) = {results['bic_by_m'][m_opt]:.1f}")
        delta = results['bic_by_m'][0] - results['bic_by_m'][m_opt]
        summary_lines.append(f"Улучшение BIC: {delta:.1f}")

    if results['supF_stats']:
        for m in sorted(results['supF_stats'].keys()):
            p = results['supF_pvalues'][m]
            stars = ' ★' if p < 0.05 else ''
            summary_lines.append(f"SupF({m}): F={results['supF_stats'][m]:.1f}, p={p:.4f}{stars}")

    ax4.text(0.05, 0.95, "\n".join(summary_lines), transform=ax4.transAxes,
             fontsize=10, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout()
    plt.show()
    return fig


# ============================================================
# ГЛАВНАЯ ФУНКЦИЯ
# ============================================================

def bai_perron_break_test(
        file_path,
        sheet_name=0,
        max_breaks=3,
        trim=0.15,
        model_type='const',
        alpha=0.05,
        show_plot=True
):
    """
    Тест Бай-Перрона на структурные сдвиги.

    Загружает данные из Excel, выполняет тест и выводит результаты.

    Параметры:
    ----------
    file_path : str — путь к Excel-файлу.
    sheet_name : str/int — лист в Excel.
    max_breaks : int — максимальное количество сдвигов (1-5).
    trim : float — минимальная доля ряда в сегменте (0.10-0.25).
    model_type : str — 'const' (изменение среднего) или 'trend' (изменение тренда).
    alpha : float — уровень значимости.
    show_plot : bool — показывать графики.

    Возвращает:
    ----------
    dict — результаты теста.
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
    print("ТЕСТ БАЙ-ПЕРРОНА НА СТРУКТУРНЫЕ СДВИГИ")
    print("=" * 70)
    print(f"Файл: {os.path.basename(file_path)}")
    print(f"Макс. сдвигов: {max_breaks}")
    print(f"Мин. сегмент: {int(100 * trim)}% ряда")
    print(f"Модель: {model_type}")

    # Загрузка
    df = pd.read_excel(file_path, sheet_name=sheet_name, header=None)
    df.columns = ['period_str', 'value']
    df['date'] = df['period_str'].apply(_parse_quarter_date)
    bad = df['date'].isna()
    if bad.any():
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
    print(f"Среднее: {y.mean():.0f} | Мин: {y.min():.0f} | Макс: {y.max():.0f}")

    print(f"\nРяд:")
    for i in range(n):
        print(f"  {_quarter_to_str(dates_pd[i])}: {y[i]:>10.0f}")

    # Тест
    print(f"\n{'─' * 70}")
    print("ВЫПОЛНЕНИЕ ТЕСТА...")
    print(f"{'─' * 70}")

    result = bai_perron_test(y, max_breaks=max_breaks, trim=trim, model_type=model_type)

    # Вывод
    print(f"\n{'═' * 70}")
    print("РЕЗУЛЬТАТЫ")
    print(f"{'═' * 70}")

    print(f"\n1. BIC:")
    for m in sorted(result['bic_by_m'].keys()):
        mark = ' ← ОПТИМУМ' if m == result['optimal_m_bic'] else ''
        print(f"   m={m}: BIC={result['bic_by_m'][m]:.1f}{mark}")

    m_opt = result['optimal_m_bic']
    breaks_idx = result['optimal_breaks_bic']

    print(f"\n   Оптимально сдвигов: {m_opt}")

    if m_opt > 0 and breaks_idx:
        print(f"\n2. СДВИГИ:")
        for i, br in enumerate(breaks_idx):
            d = _compute_break_date(dates_pd, br)
            if d is not None:
                print(f"   Сдвиг {i + 1}: {_quarter_to_str(d)} (между точками {br}-{br + 1})")

        print(f"\n3. СЕГМЕНТЫ:")
        segs = [0] + breaks_idx + [n]
        for i in range(len(segs) - 1):
            s, e = segs[i], segs[i + 1]
            seg_y = y[s:e]
            print(f"   Сегмент {i + 1}: {_quarter_to_str(dates_pd[s])} – {_quarter_to_str(dates_pd[e - 1])}")
            print(f"   Длина: {len(seg_y)}, среднее: {np.mean(seg_y):.0f}, стд: {np.std(seg_y):.0f}")
            if i > 0:
                prev_mean = np.mean(y[segs[i - 1]:segs[i]])
                chg = np.mean(seg_y) - prev_mean
                chg_pct = chg / prev_mean * 100 if prev_mean != 0 else np.inf
                print(f"   Изменение: {chg:+.0f} ({chg_pct:+.1f}%)")
            print()
    else:
        print(f"\n2. СДВИГИ НЕ ОБНАРУЖЕНЫ")

    if result['supF_stats']:
        print(f"\n4. SupF-ТЕСТЫ:")
        for m in sorted(result['supF_stats'].keys()):
            p = result['supF_pvalues'][m]
            star = ' ★ ЗНАЧИМ' if p < alpha else ''
            print(f"   SupF({m}): F={result['supF_stats'][m]:.2f}, p={p:.4f}{star}")

    print(f"\n{'═' * 70}")
    print("ИТОГ")
    print(f"{'═' * 70}")

    if m_opt == 0:
        print(f"\n   СТРУКТУРНЫЕ СДВИГИ НЕ ОБНАРУЖЕНЫ.")
    else:
        print(f"\n   ОБНАРУЖЕНО {m_opt} СТРУКТУРНЫХ СДВИГА(ОВ).")
        print(f"   Ряд содержит {m_opt + 1} однородных сегментов.")

    if show_plot:
        _plot_bai_perron_results(result, y, dates_pd, os.path.basename(file_path))

    print(f"\n{'═' * 70}")
    print("ГОТОВО")
    print(f"{'═' * 70}")

    return result