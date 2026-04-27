import numpy as np
from scipy import stats
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import warnings
warnings.filterwarnings('ignore')
import statsmodels.api as sm


def engle_granger_cointegration(y, x, alpha=0.05):
    """
    Тест коинтеграции Энгла-Грейнджера

    Parameters:
    -----------
    y : pandas.Series
        Зависимая переменная
    x : pandas.Series
        Независимая переменная
    alpha : float
        Уровень значимости (по умолчанию 0.05)

    Returns:
    --------
    dict : Результаты теста
    """
    # Коинтеграционная регрессия y = α + β*x + ε
    X = sm.add_constant(x)
    model = sm.OLS(y, X).fit()
    residuals = model.resid

    # Получаем параметры (коэффициенты)
    intercept = model.params.iloc[0]
    slope = model.params.iloc[1]

    # ADF тест на остатках
    adf_stat, p_value, used_lag, nobs, crit_values, icbest = adfuller(residuals, autolag='AIC')

    # Результат
    is_cointegrated = p_value < alpha

    # Вывод
    print("=" * 60)
    print("ТЕСТ КОИНТЕГРАЦИИ ЭНГЛА-ГРЕЙНДЖЕРА")
    print("=" * 60)
    print(f"\nКоинтеграционная регрессия:")
    print(f"   y = {slope:.6f} * x + {intercept:.6f}")
    print(f"   R² = {model.rsquared:.4f}")
    print(f"   Стандартная ошибка: {model.mse_resid:.4f}")

    print(f"\nADF тест остатков:")
    print(f"   ADF статистика: {adf_stat:.6f}")
    print(f"   p-value: {p_value:.6f}")

    print(f"\nКритические значения:")
    for key, value in crit_values.items():
        print(f"   {key}: {value:.4f}")

    print(f"\nВЫВОД:")
    if is_cointegrated:
        print(f"   ✅ Ряды КОИНТЕГРИРОВАНЫ (p-value={p_value:.4f} < {alpha})")
        print(f"   → Существует долгосрочная равновесная связь")
    else:
        print(f"   ❌ Ряды НЕ КОИНТЕГРИРОВАНЫ (p-value={p_value:.4f} ≥ {alpha})")
        print(f"   → Долгосрочная связь отсутствует")

    return {
        'is_cointegrated': is_cointegrated,
        'cointegration_eq': {'slope': slope, 'intercept': intercept},
        'residuals': residuals,
        'r_squared': model.rsquared,
        'adf_stat': adf_stat,
        'p_value': p_value,
        'critical_values': crit_values,
        'model': model
    }


def adf_test(series, series_name, alpha=0.05):
    """
    Проверка ряда на стационарность (ADF тест)
    """
    data = series.dropna()
    adf_stat, p_value, used_lag, nobs, crit_values, icbest = adfuller(data, autolag='AIC')

    is_stationary = p_value < alpha

    print(f"\n📊 ADF ТЕСТ: {series_name}")
    print(f"   Количество наблюдений: {nobs}")
    print(f"   ADF статистика: {adf_stat:.6f}")
    print(f"   p-value: {p_value:.6f}")
    print(f"   Критические значения:")
    for key, value in crit_values.items():
        print(f"      {key}: {value:.4f}")

    if is_stationary:
        print(f"✅ ВЫВОД: Ряд СТАЦИОНАРЕН")
    else:
        print(f"❌ ВЫВОД: Ряд НЕ СТАЦИОНАРЕН")

    return {
        'is_stationary': is_stationary,
        'adf_stat': adf_stat,
        'p_value': p_value,
        'critical_values': crit_values
    }

def plot_original_time_series(df, first_df, first_name_df, second_df, second_name_df):
    plt.figure(figsize=(12, 6))

    plt.plot(df.index, df[first_df], label=first_name_df)
    plt.plot(df.index, df[second_df], label=second_name_df)

    plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.xticks(rotation=45)

    plt.title('Исходные временные ряды')
    plt.xlabel('Даты')
    plt.ylabel('Значение')
    plt.legend()
    plt.grid()

    plt.show()