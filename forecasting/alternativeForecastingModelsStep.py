import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from scipy import stats
import warnings
import re
import os

warnings.filterwarnings('ignore')

# Попытка импорта Prophet
try:
    from prophet import Prophet

    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    print("⚠️ Prophet не установлен. Установите: pip install prophet")


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


# ============================================================
# МОДЕЛИ
# ============================================================

def forecast_ets(y_train, y_test, seasonal=False, seasonal_periods=4):
    """
    Экспоненциальное сглаживание (ETS).
    Автоматически выбирает модель (A,N,N) или (A,A,N) и т.д.
    """
    try:
        if seasonal and len(y_train) >= 2 * seasonal_periods:
            model = ExponentialSmoothing(
                y_train,
                seasonal_periods=seasonal_periods,
                trend='add',
                seasonal='add',
                initialization_method='estimated'
            )
        else:
            model = ExponentialSmoothing(
                y_train,
                trend='add',
                initialization_method='estimated'
            )

        fitted = model.fit()
        forecast = fitted.forecast(len(y_test))
        return forecast, fitted
    except Exception as e:
        # Fallback: простое экспоненциальное сглаживание
        print(f"  ⚠️ ETS error: {e}, fallback to simple smoothing")
        alpha = 0.3
        forecast = np.zeros(len(y_test))
        last_val = y_train[-1]
        for i in range(len(y_test)):
            forecast[i] = alpha * last_val + (1 - alpha) * forecast[i - 1] if i > 0 else last_val
        return forecast, None


def forecast_arima_simple(y_train, y_test, max_p=1, max_d=1, max_q=1):
    """
    Простая ARIMA с ограниченной сложностью (p+q ≤ 2).
    """
    best_aic = np.inf
    best_order = (0, 1, 0)
    best_model = None

    for p in range(max_p + 1):
        for d in range(max_d + 1):
            for q in range(max_q + 1):
                if p == 0 and q == 0:
                    continue
                if p + q > 2:  # ограничение сложности
                    continue
                try:
                    model = ARIMA(y_train, order=(p, d, q))
                    fitted = model.fit(method_kwargs={'maxiter': 200})
                    if fitted.aic < best_aic:
                        best_aic = fitted.aic
                        best_order = (p, d, q)
                        best_model = fitted
                except:
                    continue

    if best_model is not None:
        forecast = best_model.forecast(len(y_test))
        return forecast, best_order
    else:
        # Fallback: дрейф
        drift = (y_train[-1] - y_train[0]) / len(y_train)
        forecast = y_train[-1] + drift * np.arange(1, len(y_test) + 1)
        return forecast, 'drift_fallback'


def forecast_prophet(y_train_dates, y_train, y_test_dates, changepoints=None):
    """
    Модель Prophet с автоматическим обнаружением точек перелома.
    """
    if not PROPHET_AVAILABLE:
        return None, None

    try:
        df_train = pd.DataFrame({
            'ds': y_train_dates,
            'y': y_train
        })

        model_kwargs = {
            'yearly_seasonality': False,
            'weekly_seasonality': False,
            'daily_seasonality': False,
            'changepoint_prior_scale': 0.5,  # гибкость обнаружения сдвигов
            'seasonality_mode': 'additive',
        }

        if changepoints:
            model_kwargs['changepoints'] = pd.to_datetime(changepoints)

        model = Prophet(**model_kwargs)
        model.fit(df_train)

        df_future = pd.DataFrame({'ds': y_test_dates})
        forecast_df = model.predict(df_future)

        return forecast_df['yhat'].values, model
    except Exception as e:
        print(f"  ⚠️ Prophet error: {e}")
        return None, None


def forecast_naive(y_train, y_test, method='last'):
    """
    Бенчмарк-модели.
    method: 'last' — наивный (последнее значение),
            'mean' — среднее по обучающей выборке,
            'drift' — дрейф (линейный тренд).
    """
    if method == 'last':
        return np.repeat(y_train[-1], len(y_test))
    elif method == 'mean':
        return np.repeat(np.mean(y_train), len(y_test))
    elif method == 'drift':
        drift_val = (y_train[-1] - y_train[0]) / (len(y_train) - 1)
        return y_train[-1] + drift_val * np.arange(1, len(y_test) + 1)
    else:
        return np.repeat(y_train[-1], len(y_test))


def forecast_theta(y_train, y_test, theta=2.0):
    """
    Theta-метод: разложение на краткосрочную и долгосрочную компоненты.
    Хорош для рядов с пиком и последующим спадом (Λ-образные).
    """
    # Простая реализация: экспоненциальное сглаживание с дрейфом
    from statsmodels.tsa.holtwinters import SimpleExpSmoothing

    # Оценка тренда
    t = np.arange(len(y_train))
    trend_coef = np.polyfit(t, y_train, 1)[0]

    # Детрендирование
    trend = trend_coef * t
    detrended = y_train - trend

    # Экспоненциальное сглаживание детрендированного ряда с theta
    ses_alpha = 2.0 / (theta + 1)  # theta-преобразование
    model = SimpleExpSmoothing(detrended)
    fitted = model.fit(smoothing_level=ses_alpha, optimized=False)

    # Прогноз
    forecast_detrended = fitted.forecast(len(y_test))
    forecast_trend = trend_coef * np.arange(len(y_train), len(y_train) + len(y_test))
    forecast = forecast_detrended + forecast_trend

    return forecast


# ============================================================
# ГЛАВНАЯ ФУНКЦИЯ СРАВНЕНИЯ МОДЕЛЕЙ
# ============================================================

def compare_forecast_models(
        file_path,
        test_size=4,
        seasonal=False,
        break_dates=None,
        show_plot=True
):
    """
    Сравнение альтернативных моделей прогнозирования.

    Модели:
    - Naïve (last) — бенчмарк
    - Naïve (drift) — бенчмарк с трендом
    - ETS — экспоненциальное сглаживание
    - ARIMA (простая) — с ограничением p+q ≤ 2
    - Theta — для Λ-образных рядов
    - Prophet — с автообнаружением сдвигов
    """

    y, dates_pd, name = _load_data(file_path)
    name = name.replace('.xlsx', '')
    n = len(y)

    train_idx = n - test_size
    y_train = y[:train_idx]
    y_test = y[train_idx:]
    dates_train = dates_pd[:train_idx]
    dates_test = dates_pd[train_idx:]

    print("=" * 70)
    print(f"СРАВНЕНИЕ МОДЕЛЕЙ: {name}")
    print("=" * 70)
    print(f"Длина ряда: {n}, обучение: {len(y_train)}, тест: {len(y_test)}")
    print(f"Сезонность: {'да' if seasonal else 'нет'}")

    results = {}

    # 1. Naïve (last)
    fc_naive_last = forecast_naive(y_train, y_test, method='last')
    results['Naïve (last)'] = fc_naive_last

    # 2. Naïve (drift)
    fc_naive_drift = forecast_naive(y_train, y_test, method='drift')
    results['Naïve (drift)'] = fc_naive_drift

    # 3. ETS
    fc_ets, _ = forecast_ets(y_train, y_test, seasonal=seasonal)
    results['ETS'] = fc_ets

    # 4. ARIMA простая
    fc_arima, arima_order = forecast_arima_simple(y_train, y_test, max_p=1, max_d=2, max_q=1)
    results[f'ARIMA{arima_order}'] = fc_arima

    # 5. Theta
    fc_theta = forecast_theta(y_train, y_test, theta=2.0)
    results['Theta'] = fc_theta

    # 6. Prophet
    if PROPHET_AVAILABLE:
        fc_prophet, _ = forecast_prophet(dates_train, y_train, dates_test)
        if fc_prophet is not None:
            results['Prophet'] = fc_prophet

    # Вычисление метрик
    print(f"\n{'─' * 80}")
    print(f"{'Модель':<22} {'MAE':>12} {'RMSE':>12} {'MAPE':>10} {'Рейтинг':>10}")
    print(f"{'─' * 80}")

    metrics = {}
    for model_name, forecast in results.items():
        if forecast is None:
            continue

        mae = mean_absolute_error(y_test, forecast)
        rmse = np.sqrt(mean_squared_error(y_test, forecast))

        with np.errstate(divide='ignore', invalid='ignore'):
            mape = np.mean(np.abs((y_test - forecast) / np.where(np.abs(y_test) > 1e-10, y_test, np.nan))) * 100
            if np.isnan(mape):
                mape = np.inf

        metrics[model_name] = {'mae': mae, 'rmse': rmse, 'mape': mape}

    # Сортировка по MAPE
    sorted_models = sorted(metrics.items(), key=lambda x: x[1]['mape'])

    for rank, (model_name, m) in enumerate(sorted_models, 1):
        stars = '★★★' if rank == 1 else ('★★' if rank == 2 else ('★' if rank == 3 else ''))
        print(f"{model_name:<22} {m['mae']:>12.1f} {m['rmse']:>12.1f} {m['mape']:>9.1f}% {stars:>10}")

    # Сравнение прогнозов
    print(f"\n{'─' * 80}")
    print("СРАВНЕНИЕ ПРОГНОЗОВ ПО КВАРТАЛАМ")
    print(f"{'─' * 80}")

    header = f"{'Квартал':<12} {'Факт':>10}"
    for model_name in [m[0] for m in sorted_models[:4]]:
        header += f" {model_name:>12}"
    print(header)
    print(f"{'─' * 80}")

    for i in range(test_size):
        row = f"{_quarter_to_str(dates_test[i]):<12} {y_test[i]:>10.0f}"
        for model_name in [m[0] for m in sorted_models[:4]]:
            row += f" {results[model_name][i]:>12.0f}"
        print(row)

    # Визуализация
    if show_plot:
        fig, axes = plt.subplots(2, 2, figsize=(18, 10))
        fig.suptitle(f'{name}: Сравнение моделей прогнозирования', fontsize=14, fontweight='bold')

        # График 1: Все модели
        ax1 = axes[0, 0]
        ax1.plot(dates_train, y_train, 'b-', linewidth=2, label='Обучение')
        ax1.plot(dates_test, y_test, 'g-o', markersize=10, linewidth=2, label='Факт')

        colors = ['red', 'orange', 'purple', 'brown', 'pink', 'cyan']
        for i, (model_name, _) in enumerate(sorted_models[:6]):
            if model_name in results:
                ax1.plot(dates_test, results[model_name], f'--o',
                         color=colors[i % len(colors)], markersize=6,
                         label=f'{model_name}', alpha=0.8)

        if break_dates:
            for bd in break_dates:
                year, q_str = bd.split('-Q')
                bdate = pd.Timestamp(year=int(year), month=(int(q_str) - 1) * 3 + 1, day=1)
                ax1.axvline(x=bdate, color='gray', linestyle=':', linewidth=1.5, alpha=0.7)

        ax1.set_title('Все модели')
        ax1.tick_params(axis='x', rotation=45)
        ax1.legend(fontsize=7, loc='upper left')
        ax1.grid(True, alpha=0.3)

        # График 2: Лучшая модель vs бенчмарк
        ax2 = axes[0, 1]
        best_model = sorted_models[0][0]
        ax2.plot(dates_train, y_train, 'b-', linewidth=2, label='Обучение')
        ax2.plot(dates_test, y_test, 'g-o', markersize=10, linewidth=2, label='Факт')
        ax2.plot(dates_test, results[best_model], 'r--o', markersize=10, linewidth=2,
                 label=f'Лучшая: {best_model}')
        ax2.plot(dates_test, results['Naïve (last)'], 'gray', linestyle=':', linewidth=2,
                 label='Naïve (бенчмарк)')
        ax2.set_title(f'Лучшая модель: {best_model} (MAPE={metrics[best_model]["mape"]:.1f}%)')
        ax2.tick_params(axis='x', rotation=45)
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # График 3: Сравнение MAPE
        ax3 = axes[1, 0]
        model_names = [m[0] for m in sorted_models]
        mape_values = [metrics[m]['mape'] for m in model_names]

        # Обрезаем экстремальные значения для визуализации
        max_display = min(200, np.percentile([m for m in mape_values if m < 1000], 90) * 2)
        mape_display = [min(m, max_display) for m in mape_values]

        colors_bar = ['#4CAF50' if m == min(mape_values) else '#2196F3' for m in mape_values]
        bars = ax3.barh(model_names, mape_display, color=colors_bar, alpha=0.8)
        ax3.set_title('MAPE по моделям (чем меньше, тем лучше)')
        ax3.set_xlabel('MAPE %')
        ax3.grid(True, alpha=0.3, axis='x')

        for bar, mape_val in zip(bars, mape_values):
            ax3.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                     f'{mape_val:.1f}%', va='center', fontsize=9)

        # График 4: Ошибки лучшей модели
        ax4 = axes[1, 1]
        best_forecast = results[best_model]
        errors = best_forecast - y_test

        colors_err = ['#F44336' if e > 0 else '#4CAF50' for e in errors]
        ax4.bar(range(test_size), errors, color=colors_err, alpha=0.7)
        ax4.axhline(y=0, color='black', linewidth=0.8)
        ax4.set_title(f'Ошибки лучшей модели ({best_model})')
        ax4.set_xlabel('Шаг прогноза')
        ax4.set_ylabel('Ошибка')
        ax4.set_xticks(range(test_size))
        ax4.set_xticklabels([_quarter_to_str(d) for d in dates_test], rotation=45)
        ax4.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.show()

    return {
        'name': name,
        'results': results,
        'metrics': metrics,
        'best_model': sorted_models[0][0] if sorted_models else None,
        'best_mape': sorted_models[0][1]['mape'] if sorted_models else None,
    }


# ============================================================
# ЗАПУСК ДЛЯ ВСЕХ ПОКАЗАТЕЛЕЙ
# ============================================================

if __name__ == "__main__":

    print("\n" + "█" * 70)
    print("█  СРАВНЕНИЕ АЛЬТЕРНАТИВНЫХ МОДЕЛЕЙ ПРОГНОЗИРОВАНИЯ")
    print("█" * 70)

    specs = {
        'DboFizLKolObs': {
            'file': 'DboFizLKolObs.xlsx',
            'seasonal': False,
            'breaks': ['2023-Q2', '2024-Q4'],
        },
        'DboFizObTic': {
            'file': 'DboFizObTic.xlsx',
            'seasonal': False,
            'breaks': ['2022-Q3', '2023-Q2', '2024-Q3'],
        },
        'InterResBezlicenzii': {
            'file': 'InterResBezlicenzii.xlsx',
            'seasonal': False,
            'breaks': ['2023-Q1', '2025-Q1'],
        },
        'InterResPiramid': {
            'file': 'InterResPiramid.xlsx',
            'seasonal': False,
            'breaks': ['2023-Q1'],
        },
        'MohenTel8800': {
            'file': 'MohenTel8800.xlsx',
            'seasonal': False,
            'breaks': None,
        },
        'MohenTelGorod': {
            'file': 'MohenTelGorod.xlsx',
            'seasonal': False,
            'breaks': ['2021-Q4', '2022-Q4'],
        },
        'MohenTelMobilka': {
            'file': 'MohenTelMobilka.xlsx',
            'seasonal': False,
            'breaks': ['2022-Q3', '2023-Q4'],
        },
        'ObhKartinaKolObs': {
            'file': 'ObhKartinaKolObs.xlsx',
            'seasonal': False,
            'breaks': ['2023-Q2'],
        },
        'ObhKartinaObTic': {
            'file': 'ObhKartinaObTic.xlsx',
            'seasonal': True,
            'breaks': ['2024-Q3'],
        },
    }

    all_results = {}

    for name, spec in specs.items():
        print(f"\n{'▶' * 35}")
        print(f"▶  {name}")
        print(f"{'▶' * 35}")

        try:
            result = compare_forecast_models(
                file_path=spec['file'],
                test_size=4,
                seasonal=spec.get('seasonal', False),
                break_dates=spec.get('breaks'),
                show_plot=True,
            )
            all_results[name] = result
        except Exception as e:
            print(f"  ⚠️ Ошибка: {e}")

    # Итоговая сводка
    if all_results:
        print(f"\n{'═' * 80}")
        print("ИТОГОВАЯ СВОДКА: ЛУЧШИЕ МОДЕЛИ")
        print(f"{'═' * 80}")
        print(f"{'Показатель':<30} {'Лучшая модель':<20} {'MAPE':>10} {'Лучше SARIMAX?':>20}")
        print(f"{'─' * 80}")

        for name, res in all_results.items():
            improvement = ""
            if res['best_model'] and 'SARIMAX' not in res['best_model']:
                improvement = "✅ Да"
            elif res['best_model'] and 'SARIMAX' in res['best_model']:
                improvement = "═ Равно"
            print(f"{name:<30} {res['best_model']:<20} {res['best_mape']:>9.1f}% {improvement:>20}")