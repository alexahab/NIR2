import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import pearsonr
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, grangercausalitytests
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
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


def _load_data(file_path, sheet_name=0):
    """Загрузка данных из Excel-файла."""
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

    df = pd.read_excel(file_path, sheet_name=sheet_name, header=None)
    df.columns = ['period_str', 'value']
    df['date'] = df['period_str'].apply(_parse_quarter_date)
    df = df.dropna(subset=['date']).sort_values('date').reset_index(drop=True)

    y = df['value'].values.astype(float)
    dates_pd = pd.to_datetime(df['date'].values)

    return y, dates_pd, os.path.basename(file_path)


def _make_stationary(y, method='diff'):
    """Приведение ряда к стационарному виду."""
    y = np.array(y, dtype=float).flatten()

    if method == 'diff':
        return np.diff(y)
    elif method == 'log_diff':
        return np.diff(np.log(np.maximum(y, 1)))
    elif method == 'detrend':
        t = np.arange(len(y))
        X = sm.add_constant(t)
        model = sm.OLS(y, X).fit()
        return model.resid
    else:
        return y - np.mean(y)


def _prewhiten(y1, y2, max_lag=4):
    """
    Предварительное отбеливание рядов для кросс-корреляции.
    Удаляет автокорреляцию из y1, применяет тот же фильтр к y2.
    """
    # Подбираем AR модель для y1
    best_aic = np.inf
    best_order = 1

    for p in range(1, max_lag + 1):
        try:
            model = ARIMA(y1, order=(p, 0, 0))
            fitted = model.fit()
            if fitted.aic < best_aic:
                best_aic = fitted.aic
                best_order = p
        except:
            continue

    # Фильтруем y1 и применяем тот же фильтр к y2
    try:
        model = ARIMA(y1, order=(best_order, 0, 0))
        fitted = model.fit()
        resid1 = fitted.resid
        resid2 = fitted.filter(y2)
        return resid1, resid2
    except:
        return y1 - np.mean(y1), y2 - np.mean(y2)


# ============================================================
# ТЕСТ 1: КРОСС-КОРРЕЛЯЦИОННЫЙ АНАЛИЗ
# ============================================================

def cross_correlation_analysis(y1, y2, name1, name2, max_lag=8, alpha=0.05):
    """
    Кросс-корреляционный анализ с предварительным отбеливанием.

    Ищет значимые корреляции между рядами на разных лагах.
    Для миграции ищем: рост y1 предшествует росту y2 (положительный пик на лаге +k).
    """

    # Приводим к стационарному виду
    y1_stat = _make_stationary(y1, method='diff')
    y2_stat = _make_stationary(y2, method='diff')

    # Выравниваем длину
    min_len = min(len(y1_stat), len(y2_stat))
    y1_stat = y1_stat[:min_len]
    y2_stat = y2_stat[:min_len]

    # Предварительное отбеливание
    y1_white, y2_white = _prewhiten(y1_stat, y2_stat)

    # Кросс-корреляция
    ccf_values = []
    lags = list(range(-max_lag, max_lag + 1))

    for lag in lags:
        if lag < 0:
            # y2 опережает y1
            corr, p_value = pearsonr(y1_white[-lag:], y2_white[:lag])
        elif lag > 0:
            # y1 опережает y2
            corr, p_value = pearsonr(y1_white[:-lag], y2_white[lag:])
        else:
            corr, p_value = pearsonr(y1_white, y2_white)
        ccf_values.append((corr, p_value))

    # Критическое значение
    n_eff = len(y1_white)
    crit_value = 1.96 / np.sqrt(n_eff)

    results = {
        'lags': lags,
        'ccf': [v[0] for v in ccf_values],
        'p_values': [v[1] for v in ccf_values],
        'crit_value': crit_value,
        'significant_lags': [],
    }

    # Находим значимые лаги
    for i, lag in enumerate(lags):
        if abs(ccf_values[i][0]) > crit_value:
            results['significant_lags'].append({
                'lag': lag,
                'corr': ccf_values[i][0],
                'p_value': ccf_values[i][1],
            })

    return results


def _plot_cross_correlation(results, name1, name2):
    """Визуализация кросс-корреляционного анализа."""
    fig, ax = plt.subplots(figsize=(14, 6))

    lags = results['lags']
    ccf = results['ccf']
    crit = results['crit_value']

    colors = ['#F44336' if abs(c) > crit else '#2196F3' for c in ccf]
    ax.bar(lags, ccf, color=colors, alpha=0.8, edgecolor='black')

    ax.axhline(y=crit, color='red', linestyle='--', linewidth=1.5, label=f'Критич. знач. (±{crit:.3f})')
    ax.axhline(y=-crit, color='red', linestyle='--', linewidth=1.5)
    ax.axhline(y=0, color='black', linewidth=0.8)

    ax.set_title(f'Кросс-корреляция: {name1} → {name2}\n(положительный лаг = {name2} реагирует на {name1} с задержкой)',
                 fontsize=13, fontweight='bold')
    ax.set_xlabel('Лаг (кварталы)')
    ax.set_ylabel('Кросс-корреляция')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Подписываем значимые пики
    for i, lag in enumerate(lags):
        if abs(ccf[i]) > crit and lag > 0:
            ax.annotate(f'лаг={lag}\nr={ccf[i]:.3f}',
                        xy=(lag, ccf[i]),
                        xytext=(10, 20 if ccf[i] > 0 else -20),
                        textcoords="offset points",
                        fontsize=9, color='darkred',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8),
                        arrowprops=dict(arrowstyle='->', color='darkred'))

    plt.tight_layout()
    plt.show()
    return fig


# ============================================================
# ТЕСТ 2: ПРИЧИННОСТЬ ПО ГРЕЙНДЖЕРУ
# ============================================================

def granger_causality_test(y1, y2, name1, name2, max_lag=4, alpha=0.05):
    """
    Тест причинности по Грейнджеру.

    H₀: y1 НЕ является причиной по Грейнджеру для y2.
    H₁: y1 является причиной по Грейнджеру для y2.
    """

    # Приводим к стационарному виду
    y1_stat = _make_stationary(y1, method='diff')
    y2_stat = _make_stationary(y2, method='diff')

    min_len = min(len(y1_stat), len(y2_stat))
    y1_stat = y1_stat[:min_len]
    y2_stat = y2_stat[:min_len]

    # Формируем двумерный ряд
    data = np.column_stack([y2_stat, y1_stat])  # y2 первая (зависимая), y1 вторая (причина)

    # Тест Грейнджера
    gc_results = grangercausalitytests(data, maxlag=max_lag, verbose=False)

    results = {
        'name_donor': name1,
        'name_acceptor': name2,
        'max_lag': max_lag,
        'tests': [],
    }

    for lag in range(1, max_lag + 1):
        f_test = gc_results[lag][0]['ssr_ftest']
        chi2_test = gc_results[lag][0]['ssr_chi2test']

        results['tests'].append({
            'lag': lag,
            'F_stat': f_test[0],
            'F_pvalue': f_test[1],
            'Chi2_stat': chi2_test[0],
            'Chi2_pvalue': chi2_test[1],
            'significant_F': f_test[1] < alpha,
            'significant_Chi2': chi2_test[1] < alpha,
        })

    return results


def _plot_granger_results(results):
    """Визуализация результатов теста Грейнджера."""
    fig, ax = plt.subplots(figsize=(12, 6))

    lags = [t['lag'] for t in results['tests']]
    f_stats = [t['F_stat'] for t in results['tests']]
    p_values = [t['F_pvalue'] for t in results['tests']]

    colors = ['#F44336' if p < 0.05 else '#4CAF50' for p in p_values]

    ax.bar(lags, f_stats, color=colors, alpha=0.8, edgecolor='black')
    ax.axhline(y=3.0, color='red', linestyle='--', linewidth=1.5, label='Критич. F (α=0.05) ≈ 3.0')

    ax.set_title(f'Тест Грейнджера: {results["name_donor"]} → {results["name_acceptor"]}\n'
                 f'(H₀: {results["name_donor"]} НЕ является причиной для {results["name_acceptor"]})',
                 fontsize=13, fontweight='bold')
    ax.set_xlabel('Лаг (кварталы)')
    ax.set_ylabel('F-статистика')
    ax.set_xticks(lags)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    for i, (lag, f, p) in enumerate(zip(lags, f_stats, p_values)):
        stars = ' ★' if p < 0.05 else ''
        ax.text(lag, f + 0.3, f'p={p:.3f}{stars}', ha='center', fontsize=10,
                color='darkred' if p < 0.05 else 'darkgreen')

    plt.tight_layout()
    plt.show()
    return fig


# ============================================================
# ТЕСТ 3: ФУНКЦИИ ИМПУЛЬСНОГО ОТКЛИКА (IRF)
# ============================================================

def impulse_response_analysis(y1, y2, name1, name2, max_lag=8, n_bootstrap=500):
    """
    Анализ функций импульсного отклика на основе VAR-модели.

    Показывает, как шок в y1 влияет на y2 с течением времени.
    """

    # Приводим к стационарному виду
    y1_stat = _make_stationary(y1, method='diff')
    y2_stat = _make_stationary(y2, method='diff')

    min_len = min(len(y1_stat), len(y2_stat))
    y1_stat = y1_stat[:min_len]
    y2_stat = y2_stat[:min_len]

    # Строим VAR модель
    data = np.column_stack([y1_stat, y2_stat])

    # Определяем оптимальный лаг по AIC
    best_aic = np.inf
    best_lag = 1

    for p in range(1, min(5, len(y1_stat) // 3)):
        try:
            model = sm.tsa.VAR(data)
            result = model.fit(p)
            if result.aic < best_aic:
                best_aic = result.aic
                best_lag = p
        except:
            continue

    # Оцениваем VAR с оптимальным лагом
    model = sm.tsa.VAR(data)
    result = model.fit(best_lag)

    # Вычисляем IRF (шок в y1 → отклик y2)
    irf = result.irf(best_lag * 2)

    # Ортогонализованные импульсные отклики
    orth_irf = irf.orth_irfs

    # Отклик y2 на шок y1
    response_y2_to_y1 = irf.irfs[:, 1, 0]  # переменная 0 → переменная 1

    # Доверительные интервалы через bootstrap
    n_obs = len(data) - best_lag
    irf_bootstrap = []

    for _ in range(n_bootstrap):
        idx = np.random.choice(n_obs, n_obs, replace=True)
        boot_data = data[idx]
        try:
            boot_model = sm.tsa.VAR(boot_data)
            boot_result = boot_model.fit(best_lag)
            boot_irf = boot_result.irf(best_lag * 2)
            irf_bootstrap.append(boot_irf.irfs[:, 1, 0])
        except:
            continue

    if irf_bootstrap:
        irf_bootstrap = np.array(irf_bootstrap)
        irf_lower = np.percentile(irf_bootstrap, 2.5, axis=0)
        irf_upper = np.percentile(irf_bootstrap, 97.5, axis=0)
    else:
        irf_lower = response_y2_to_y1 - 2 * irf.stderr()[:, 1, 0]
        irf_upper = response_y2_to_y1 + 2 * irf.stderr()[:, 1, 0]

    return {
        'lags': np.arange(len(response_y2_to_y1)),
        'response': response_y2_to_y1,
        'lower': irf_lower[:len(response_y2_to_y1)],
        'upper': irf_upper[:len(response_y2_to_y1)],
        'optimal_var_lag': best_lag,
        'name_shock': name1,
        'name_response': name2,
    }


def _plot_irf_results(irf_results, max_plot_lag=8):
    """Визуализация функций импульсного отклика."""
    fig, ax = plt.subplots(figsize=(14, 6))

    lags = irf_results['lags'][:max_plot_lag + 1]
    response = irf_results['response'][:max_plot_lag + 1]
    lower = irf_results['lower'][:max_plot_lag + 1]
    upper = irf_results['upper'][:max_plot_lag + 1]

    ax.plot(lags, response, 'b-', linewidth=2.5, label='Отклик')
    ax.fill_between(lags, lower, upper, alpha=0.3, color='blue', label='95% доверительный интервал')
    ax.axhline(y=0, color='black', linewidth=0.8, linestyle='--')

    # Отмечаем значимые отклики
    for i in range(len(lags)):
        if lower[i] * upper[i] > 0:  # оба одного знака → не пересекают 0
            ax.plot(lags[i], response[i], 'ro', markersize=10,
                    label='Значимый отклик' if i == 0 or lower[i - 1] * upper[i - 1] <= 0 else '')

    ax.set_title(
        f'Функция импульсного отклика: шок в {irf_results["name_shock"]} → отклик {irf_results["name_response"]}\n'
        f'(VAR({irf_results["optimal_var_lag"]}), 95% бутстрап-доверительные интервалы)',
        fontsize=13, fontweight='bold')
    ax.set_xlabel('Периоды после шока (кварталы)')
    ax.set_ylabel('Величина отклика')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
    return fig


# ============================================================
# ГЛАВНАЯ ФУНКЦИЯ: ПОЛНЫЙ АНАЛИЗ МИГРАЦИИ
# ============================================================

def migration_analysis(
        file_donor,
        file_acceptor,
        max_lag=4,
        alpha=0.05,
        show_plot=True
):
    """
    Полный анализ эффекта миграции между двумя каналами.

    Параметры:
    ----------
    file_donor : str — путь к файлу канала-донора.
    file_acceptor : str — путь к файлу канала-акцептора.
    max_lag : int — максимальный лаг для тестов.
    alpha : float — уровень значимости.
    show_plot : bool — показывать ли графики.

    Возвращает:
    ----------
    dict — полные результаты анализа.

    Пример:
    --------
    >>> result = migration_analysis('MohenTelGorod.xlsx', 'MohenTelMobilka.xlsx')
    """

    # Загрузка данных
    y_donor, dates_donor, name_donor = _load_data(file_donor)
    y_acceptor, dates_acceptor, name_acceptor = _load_data(file_acceptor)

    name_donor = name_donor.replace('.xlsx', '')
    name_acceptor = name_acceptor.replace('.xlsx', '')

    print("=" * 70)
    print("АНАЛИЗ ЭФФЕКТА МИГРАЦИИ МОШЕННИЧЕСТВА")
    print("=" * 70)
    print(f"Канал-донор:     {name_donor}")
    print(f"Канал-акцептор:  {name_acceptor}")
    print(f"Макс. лаг:       {max_lag} кварталов")
    print(f"Уровень значим.: α = {alpha}")

    # Информация о рядах
    print(f"\n{'─' * 70}")
    print("ИНФОРМАЦИЯ О РЯДАХ")
    print(f"{'─' * 70}")
    print(f"{name_donor}:")
    print(f"  Длина: {len(y_donor)}, среднее: {y_donor.mean():.0f}, "
          f"мин: {y_donor.min():.0f}, макс: {y_donor.max():.0f}")
    print(f"{name_acceptor}:")
    print(f"  Длина: {len(y_acceptor)}, среднее: {y_acceptor.mean():.0f}, "
          f"мин: {y_acceptor.min():.0f}, макс: {y_acceptor.max():.0f}")

    # ============================================================
    # ТЕСТ 1: КРОСС-КОРРЕЛЯЦИЯ
    # ============================================================
    print(f"\n{'═' * 70}")
    print("ТЕСТ 1: КРОСС-КОРРЕЛЯЦИОННЫЙ АНАЛИЗ")
    print(f"{'═' * 70}")

    ccf_results = cross_correlation_analysis(
        y_donor, y_acceptor, name_donor, name_acceptor, max_lag=max_lag
    )

    print(f"\nКритическое значение: ±{ccf_results['crit_value']:.3f}")

    if ccf_results['significant_lags']:
        print(f"\nЗначимые кросс-корреляции:")
        for sig in ccf_results['significant_lags']:
            direction = f"{name_acceptor} → {name_donor}" if sig['lag'] < 0 else f"{name_donor} → {name_acceptor}"
            print(f"  Лаг {sig['lag']:+d}: r = {sig['corr']:.4f}, p = {sig['p_value']:.4f} ({direction})")

        # Проверка на миграцию (положительный пик на положительном лаге)
        positive_lags = [s for s in ccf_results['significant_lags'] if s['lag'] > 0 and s['corr'] > 0]
        if positive_lags:
            print(f"\n✅ ОБНАРУЖЕНЫ ЗНАЧИМЫЕ ПОЛОЖИТЕЛЬНЫЕ СВЯЗИ НА ПОЛОЖИТЕЛЬНЫХ ЛАГАХ!")
            print(f"   Это согласуется с гипотезой миграции: изменения в {name_donor}")
            print(f"   предшествуют изменениям в {name_acceptor}.")
        else:
            print(f"\n⚠️ Значимые связи есть, но не на положительных лагах с положительной корреляцией.")
    else:
        print(f"\n❌ Значимых кросс-корреляций не обнаружено.")

    if show_plot:
        _plot_cross_correlation(ccf_results, name_donor, name_acceptor)

    # ============================================================
    # ТЕСТ 2: ПРИЧИННОСТЬ ПО ГРЕЙНДЖЕРУ
    # ============================================================
    print(f"\n{'═' * 70}")
    print("ТЕСТ 2: ПРИЧИННОСТЬ ПО ГРЕЙНДЖЕРУ")
    print(f"{'═' * 70}")

    # Тест: донор → акцептор
    gc_forward = granger_causality_test(y_donor, y_acceptor, name_donor, name_acceptor,
                                        max_lag=max_lag, alpha=alpha)

    print(f"\nНаправление: {name_donor} → {name_acceptor}")
    print(f"H₀: {name_donor} НЕ является причиной для {name_acceptor}")
    print(f"{'─' * 50}")

    for test in gc_forward['tests']:
        stars = ' ★ ЗНАЧИМ' if test['significant_F'] else ''
        print(f"  Лаг {test['lag']}: F = {test['F_stat']:.3f}, p = {test['F_pvalue']:.4f}{stars}")

    significant_forward = [t for t in gc_forward['tests'] if t['significant_F']]

    # Тест: акцептор → донор (обратное направление)
    gc_backward = granger_causality_test(y_acceptor, y_donor, name_acceptor, name_donor,
                                         max_lag=max_lag, alpha=alpha)

    print(f"\nНаправление: {name_acceptor} → {name_donor} (обратная связь)")
    print(f"H₀: {name_acceptor} НЕ является причиной для {name_donor}")
    print(f"{'─' * 50}")

    for test in gc_backward['tests']:
        stars = ' ★ ЗНАЧИМ' if test['significant_F'] else ''
        print(f"  Лаг {test['lag']}: F = {test['F_stat']:.3f}, p = {test['F_pvalue']:.4f}{stars}")

    significant_backward = [t for t in gc_backward['tests'] if t['significant_F']]

    # Вывод
    print(f"\n{'─' * 70}")
    print("ВЫВОД ПО ТЕСТУ ГРЕЙНДЖЕРА:")
    if significant_forward:
        print(f"  ✅ {name_donor} ЯВЛЯЕТСЯ причиной по Грейнджеру для {name_acceptor}")
        print(f"     на лаге(ах): {[t['lag'] for t in significant_forward]}")
        print(f"     Это ПОДТВЕРЖДАЕТ гипотезу миграции.")
    else:
        print(f"  ❌ {name_donor} НЕ является причиной по Грейнджеру для {name_acceptor}")

    if significant_backward:
        print(f"  ⚠️ Обнаружена обратная связь: {name_acceptor} → {name_donor}")

    if show_plot:
        _plot_granger_results(gc_forward)

    # ============================================================
    # ТЕСТ 3: ФУНКЦИИ ИМПУЛЬСНОГО ОТКЛИКА
    # ============================================================
    print(f"\n{'═' * 70}")
    print("ТЕСТ 3: ФУНКЦИИ ИМПУЛЬСНОГО ОТКЛИКА (IRF)")
    print(f"{'═' * 70}")

    try:
        irf_results = impulse_response_analysis(y_donor, y_acceptor, name_donor, name_acceptor,
                                                max_lag=max_lag * 2)

        print(f"\nОптимальный лаг VAR: {irf_results['optimal_var_lag']}")
        print(f"\nОтклик {name_acceptor} на шок {name_donor}:")

        significant_periods = []
        for i in range(min(9, len(irf_results['response']))):
            resp = irf_results['response'][i]
            lower = irf_results['lower'][i]
            upper = irf_results['upper'][i]
            is_significant = lower * upper > 0
            marker = ' ★ ЗНАЧИМ' if is_significant else ''
            print(f"  Период {i}: {resp:+.2f} [{lower:.2f}, {upper:.2f}]{marker}")
            if is_significant and i > 0:
                significant_periods.append(i)

        if significant_periods:
            print(f"\n  ✅ Значимый положительный отклик на периодах: {significant_periods}")
            print(f"     Это ПОДТВЕРЖДАЕТ гипотезу миграции: шок в {name_donor}")
            print(f"     вызывает рост {name_acceptor} через {significant_periods[0]} период(ов).")
        else:
            print(f"\n  ❌ Значимого положительного отклика не обнаружено.")

        if show_plot:
            _plot_irf_results(irf_results, max_plot_lag=8)

    except Exception as e:
        print(f"\n  ⚠️ IRF анализ не удался: {e}")
        irf_results = None

    # ============================================================
    # ИТОГ
    # ============================================================
    print(f"\n{'═' * 70}")
    print("ИТОГОВОЕ ЗАКЛЮЧЕНИЕ ПО ЭФФЕКТУ МИГРАЦИИ")
    print(f"{'═' * 70}")

    # Собираем доказательства
    evidence = []

    # Кросс-корреляция
    pos_lags_ccf = [s for s in ccf_results.get('significant_lags', []) if s['lag'] > 0 and s['corr'] > 0]
    if pos_lags_ccf:
        evidence.append(f"✅ Кросс-корреляция: значимые положительные связи на лагах {[s['lag'] for s in pos_lags_ccf]}")

    # Грейнджер
    if significant_forward:
        evidence.append(
            f"✅ Тест Грейнджера: {name_donor} → {name_acceptor} значим на лагах {[t['lag'] for t in significant_forward]}")

    # IRF
    if irf_results:
        sig_periods = []
        for i in range(1, min(9, len(irf_results['response']))):
            if irf_results['lower'][i] * irf_results['upper'][i] > 0 and irf_results['response'][i] > 0:
                sig_periods.append(i)
        if sig_periods:
            evidence.append(f"✅ IRF: значимый положительный отклик на периодах {sig_periods}")

    print()
    if len(evidence) >= 2:
        print(f"   ✅✅✅ ЭФФЕКТ МИГРАЦИИ ПОДТВЕРЖДЁН ✅✅✅")
        print(f"   {len(evidence)} из 3 тестов указывают на миграцию.")
    elif len(evidence) == 1:
        print(f"   ⚠️ ЧАСТИЧНОЕ ПОДТВЕРЖДЕНИЕ")
        print(f"   Только 1 из 3 тестов указывает на миграцию.")
    else:
        print(f"   ❌ ЭФФЕКТ МИГРАЦИИ НЕ ПОДТВЕРЖДЁН")
        print(f"   Ни один тест не обнаружил значимой связи.")

    for e in evidence:
        print(f"   {e}")

    print(f"\n{'═' * 70}")
    print("АНАЛИЗ ЗАВЕРШЁН")
    print(f"{'═' * 70}")

    return {
        'donor': name_donor,
        'acceptor': name_acceptor,
        'ccf': ccf_results,
        'granger_forward': gc_forward,
        'granger_backward': gc_backward,
        'irf': irf_results,
        'evidence': evidence,
        'conclusion': 'confirmed' if len(evidence) >= 2 else ('partial' if len(evidence) == 1 else 'rejected'),
    }
