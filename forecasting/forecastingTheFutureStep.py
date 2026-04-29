import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing, SimpleExpSmoothing
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
import warnings
import re
import os

warnings.filterwarnings('ignore')


# ============================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ============================================================

def _quarter_to_str(dt):
    quarter = (dt.month - 1) // 3 + 1
    return f"{dt.year}-Q{quarter}"


def _parse_quarter_date(quarter_str):
    s = str(quarter_str).strip()
    roman_to_int = {'I': 1, 'II': 2, 'III': 3, 'IV': 4, 'І': 1, 'ІІ': 2, 'ІІІ': 3, 'ІV': 4}
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
    if not os.path.exists(file_path):
        for alt in [file_path, os.path.join('dataExcel', file_path), os.path.join('..', 'dataExcel', file_path)]:
            if os.path.exists(alt):
                file_path = alt
                break
        else:
            raise FileNotFoundError(f"Файл не найден: {file_path}")
    df = pd.read_excel(file_path, sheet_name=sheet_name, header=None)
    df.columns = ['period_str', 'value']
    df['date'] = df['period_str'].apply(_parse_quarter_date)
    df = df.dropna(subset=['date']).sort_values('date').reset_index(drop=True)
    return df['value'].values.astype(float), pd.to_datetime(df['date'].values), os.path.basename(file_path)


def _get_future_dates(last_date, n_future):
    """Генерирует даты будущих кварталов."""
    future_dates = []
    current = last_date + pd.DateOffset(months=3)
    for _ in range(n_future):
        future_dates.append(current)
        current = current + pd.DateOffset(months=3)
    return pd.DatetimeIndex(future_dates)


# ============================================================
# МОДЕЛИ ДЛЯ БУДУЩЕГО ПРОГНОЗА
# ============================================================

def forecast_future_arima(y, n_future, max_p=1, max_d=2, max_q=1):
    """ARIMA с ограниченной сложностью."""
    best_aic = np.inf
    best_order = (0, 1, 0)
    best_model = None

    for p in range(max_p + 1):
        for d in range(max_d + 1):
            for q in range(max_q + 1):
                if p == 0 and q == 0:
                    continue
                if p + q > 2:
                    continue
                try:
                    model = ARIMA(y, order=(p, d, q))
                    fitted = model.fit(method_kwargs={'maxiter': 200})
                    if fitted.aic < best_aic:
                        best_aic = fitted.aic
                        best_order = (p, d, q)
                        best_model = fitted
                except:
                    continue

    if best_model is not None:
        forecast = best_model.forecast(n_future)
        return forecast, best_order
    else:
        drift = (y[-1] - y[0]) / (len(y) - 1)
        forecast = y[-1] + drift * np.arange(1, n_future + 1)
        return forecast, 'drift_fallback'


def forecast_future_theta(y, n_future, theta=2.0):
    """Theta-метод."""
    t = np.arange(len(y))
    trend_coef = np.polyfit(t, y, 1)[0]
    trend = trend_coef * t
    detrended = y - trend

    ses_alpha = 2.0 / (theta + 1)
    model = SimpleExpSmoothing(detrended)
    fitted = model.fit(smoothing_level=ses_alpha, optimized=False)

    forecast_detrended = fitted.forecast(n_future)
    forecast_trend = trend_coef * np.arange(len(y), len(y) + n_future)
    forecast = forecast_detrended + forecast_trend

    return forecast


def forecast_future_ets(y, n_future, seasonal=False, seasonal_periods=4):
    """ETS модель."""
    try:
        if seasonal and len(y) >= 2 * seasonal_periods:
            model = ExponentialSmoothing(
                y, seasonal_periods=seasonal_periods,
                trend='add', seasonal='add',
                initialization_method='estimated'
            )
        else:
            model = ExponentialSmoothing(
                y, trend='add',
                initialization_method='estimated'
            )
        fitted = model.fit()
        forecast = fitted.forecast(n_future)
        return forecast
    except:
        alpha = 0.3
        forecast = np.array([y[-1]])
        for i in range(1, n_future):
            forecast = np.append(forecast, alpha * forecast[-1] + (1 - alpha) * forecast[-1])
        return forecast


def forecast_future_naive(y, n_future, method='last'):
    """Наивные бенчмарки."""
    if method == 'last':
        return np.repeat(y[-1], n_future)
    elif method == 'drift':
        drift_val = (y[-1] - y[0]) / (len(y) - 1)
        return y[-1] + drift_val * np.arange(1, n_future + 1)
    elif method == 'mean':
        return np.repeat(np.mean(y), n_future)


# ============================================================
# ГЛАВНАЯ ФУНКЦИЯ
# ============================================================

def forecast_all_future(
        n_future=4,
        show_plot=True
):
    """
    Прогноз на n_future кварталов вперёд для всех показателей
    с использованием лучших моделей.
    """

    # Спецификации: файл + лучшая модель
    specs = {
        'DboFizLKolObs': {
            'file': 'DboFizLKolObs.xlsx',
            'model': 'arima',
            'description': 'Физ. лица, количество',
        },
        'DboFizObTic': {
            'file': 'DboFizObTic.xlsx',
            'model': 'theta',
            'description': 'Физ. лица, суммы',
        },
        'InterResBezlicenzii': {
            'file': 'InterResBezlicenzii.xlsx',
            'model': 'theta',
            'description': 'Интернет-мошенничество',
        },
        'InterResPiramid': {
            'file': 'InterResPiramid.xlsx',
            'model': 'naive_last',
            'description': 'Финансовые пирамиды',
        },
        'MohenTel8800': {
            'file': 'MohenTel8800.xlsx',
            'model': 'naive_last',
            'description': 'Телефонное (8-800)',
        },
        'MohenTelGorod': {
            'file': 'MohenTelGorod.xlsx',
            'model': 'naive_drift',
            'description': 'Телефонное (городские)',
        },
        'MohenTelMobilka': {
            'file': 'MohenTelMobilka.xlsx',
            'model': 'arima',
            'description': 'Телефонное (мобильные)',
        },
        'ObhKartinaKolObs': {
            'file': 'ObhKartinaKolObs.xlsx',
            'model': 'ets',
            'description': 'Агрегат, количество',
        },
        'ObhKartinaObTic': {
            'file': 'ObhKartinaObTic.xlsx',
            'model': 'theta',
            'description': 'Агрегат, суммы (сезонный)',
        },
    }

    all_forecasts = {}

    print("=" * 90)
    print("ПРОГНОЗ НА 4 КВАРТАЛА ВПЕРЁД (2025-Q4 – 2026-Q3)")
    print("=" * 90)

    # Таблица результатов
    print(f"\n{'Показатель':<30} {'Модель':<15} {'2025-Q4':>15} {'2026-Q1':>15} {'2026-Q2':>15} {'2026-Q3':>15}")
    print(f"{'─' * 90}")

    for name, spec in specs.items():
        try:
            # Загрузка данных
            y, dates_pd, _ = _load_data(spec['file'])
            last_date = dates_pd[-1]
            future_dates = _get_future_dates(last_date, n_future)

            # Прогноз в зависимости от модели
            model_name = spec['model']

            if model_name == 'arima':
                forecast, order = forecast_future_arima(y, n_future)
                model_str = f'ARIMA{order}'
            elif model_name == 'theta':
                forecast = forecast_future_theta(y, n_future)
                model_str = 'Theta'
            elif model_name == 'ets':
                forecast = forecast_future_ets(y, n_future, seasonal=('ObTic' in spec['file']))
                model_str = 'ETS'
            elif model_name == 'naive_last':
                forecast = forecast_future_naive(y, n_future, method='last')
                model_str = 'Naïve(last)'
            elif model_name == 'naive_drift':
                forecast = forecast_future_naive(y, n_future, method='drift')
                model_str = 'Naïve(drift)'
            else:
                forecast = np.repeat(y[-1], n_future)
                model_str = 'Unknown'

            # Сохраняем
            all_forecasts[name] = {
                'forecast': forecast,
                'model': model_str,
                'future_dates': future_dates,
                'y': y,
                'dates': dates_pd,
                'description': spec['description'],
            }

            # Вывод строки таблицы
            print(
                f"{name:<30} {model_str:<15} {forecast[0]:>15.0f} {forecast[1]:>15.0f} {forecast[2]:>15.0f} {forecast[3]:>15.0f}")

        except Exception as e:
            print(f"{name:<30} {'ERROR':<15} {'─':>15} {'─':>15} {'─':>15} {'─':>15}  ← {e}")

    # Визуализация
    if show_plot and all_forecasts:
        fig, axes = plt.subplots(3, 3, figsize=(20, 15))
        fig.suptitle('Прогноз на 2025-Q4 – 2026-Q3 для всех показателей',
                     fontsize=16, fontweight='bold')

        for idx, (name, res) in enumerate(all_forecasts.items()):
            ax = axes[idx // 3, idx % 3]

            # Исторические данные
            ax.plot(res['dates'], res['y'], 'b-', linewidth=2, label='История')

            # Прогноз
            future_dates = res['future_dates']
            forecast = res['forecast']

            # Соединяем последнюю точку с прогнозом
            all_dates = list(res['dates']) + list(future_dates)
            all_values = list(res['y']) + list(forecast)
            ax.plot(all_dates[-5:], all_values[-5:], 'r--o', linewidth=2.5, markersize=8, label='Прогноз')

            # Разделительная линия
            ax.axvline(x=res['dates'][-1], color='gray', linestyle=':', linewidth=1.5, alpha=0.7)

            ax.set_title(f"{name}\n({res['description']}, {res['model']})", fontsize=10, fontweight='bold')
            ax.tick_params(axis='x', rotation=45, labelsize=7)
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    return all_forecasts


# ============================================================
# ЗАПУСК
# ============================================================

if __name__ == "__main__":
    forecasts = forecast_all_future(n_future=4, show_plot=True)