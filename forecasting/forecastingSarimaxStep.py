import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error
from itertools import product
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


def _add_break_dummies(n, dates_pd, break_dates_str):
    """Создаёт дамми-переменные для структурных сдвигов (ступенька)."""
    dummies = []
    for bd in break_dates_str:
        year, q_str = bd.split('-Q')
        quarter = int(q_str)
        month = (quarter - 1) * 3 + 1
        break_date = pd.Timestamp(year=int(year), month=month, day=1)
        dummy = np.zeros(n)
        for i, d in enumerate(dates_pd):
            if d >= break_date:
                dummy[i] = 1.0
        dummies.append(dummy)
    return np.column_stack(dummies) if dummies else None


def _add_lagged_exog(y_exog, n, max_lag):
    """Создаёт лагированные значения экзогенной переменной."""
    y_exog = np.array(y_exog, dtype=float).flatten()
    lags = []
    for lag in range(1, max_lag + 1):
        lagged = np.zeros(n)
        if lag < n:
            lagged[lag:] = y_exog[:n - lag]
        lags.append(lagged)
    return np.column_stack(lags) if lags else None


# ============================================================
# SARIMAX МОДЕЛЬ (ИСПРАВЛЕННАЯ)
# ============================================================

def build_sarimax_model(
        file_path,
        break_dates=None,
        donor_file=None,
        donor_lags=None,
        seasonal=False,
        test_size=4,
        max_p=3, max_d=2, max_q=3,
        max_P=1, max_D=1, max_Q=1,
        verbose=True
):
    """
    Построение SARIMAX модели с дамми сдвигов и лагами донора.
    """

    # Загрузка целевого ряда
    y_target, dates_pd, name_target = _load_data(file_path)
    name_target = name_target.replace('.xlsx', '')
    n = len(y_target)

    if verbose:
        print("=" * 70)
        print("SARIMAX МОДЕЛЬ ПРОГНОЗИРОВАНИЯ")
        print("=" * 70)
        print(f"Целевой показатель: {name_target}")
        print(f"Длина ряда: {n} наблюдений")
        print(f"Сезонность: {'да' if seasonal else 'нет'}")
        print(f"Структурные сдвиги: {break_dates if break_dates else 'нет'}")
        print(f"Донор: {os.path.basename(donor_file) if donor_file else 'нет'}")
        print(f"Лаги донора: {donor_lags if donor_lags else 'нет'}")

    # Разделение на train/test
    train_idx = n - test_size
    y_train = y_target[:train_idx]
    y_test = y_target[train_idx:]
    dates_train = dates_pd[:train_idx]
    dates_test = dates_pd[train_idx:]

    # Формируем экзогенные переменные
    exog_train_list = []
    exog_test_list = []
    exog_names = []

    # 1. Дамми структурных сдвигов
    if break_dates:
        dummies_all = _add_break_dummies(n, dates_pd, break_dates)
        if dummies_all is not None:
            dummies_train = dummies_all[:train_idx]
            dummies_test = dummies_all[train_idx:]
            exog_train_list.append(dummies_train)
            exog_test_list.append(dummies_test)
            exog_names.extend([f'D_{bd}' for bd in break_dates])

    # 2. Лаги донора
    if donor_file and donor_lags:
        y_donor, _, name_donor = _load_data(donor_file)
        min_len = min(n, len(y_donor))
        donor_lagged_all = _add_lagged_exog(y_donor[:min_len], min_len, max(donor_lags))
        if donor_lagged_all is not None:
            selected_lags = donor_lagged_all[:, [l - 1 for l in donor_lags]]
            donor_train = selected_lags[:train_idx]
            donor_test = selected_lags[train_idx:]
            exog_train_list.append(donor_train)
            exog_test_list.append(donor_test)
            donor_short = os.path.basename(donor_file).replace('.xlsx', '')
            exog_names.extend([f'{donor_short}_lag{l}' for l in donor_lags])

    # Объединяем
    if exog_train_list:
        exog_train = np.column_stack(exog_train_list)
        exog_test = np.column_stack(exog_test_list)
    else:
        exog_train = None
        exog_test = None

    # Автоматический подбор параметров
    if verbose:
        print(f"\n{'─' * 70}")
        print("ПОДБОР ОПТИМАЛЬНЫХ ПАРАМЕТРОВ...")
        print(f"{'─' * 70}")

    best_aic = np.inf
    best_order = (1, 1, 1)
    best_seasonal_order = (0, 0, 0, 4) if seasonal else None

    if seasonal:
        for p, d, q in product(range(max_p + 1), range(1, max_d + 1), range(max_q + 1)):
            for P, D, Q in product(range(max_P + 1), range(max_D + 1), range(max_Q + 1)):
                try:
                    model = SARIMAX(y_train, exog=exog_train,
                                    order=(p, d, q),
                                    seasonal_order=(P, D, Q, 4))
                    fitted = model.fit(disp=False, maxiter=200)
                    if fitted.aic < best_aic:
                        best_aic = fitted.aic
                        best_order = (p, d, q)
                        best_seasonal_order = (P, D, Q, 4)
                except:
                    continue
    else:
        for p, d, q in product(range(max_p + 1), range(1, max_d + 1), range(max_q + 1)):
            try:
                # Используем SARIMAX даже без сезонности — у него API совместим
                model = SARIMAX(y_train, exog=exog_train, order=(p, d, q))
                fitted = model.fit(disp=False, maxiter=200)
                if fitted.aic < best_aic:
                    best_aic = fitted.aic
                    best_order = (p, d, q)
            except:
                continue

    if verbose:
        print(f"Оптимальный порядок: {best_order}")
        if seasonal:
            print(f"Сезонный порядок: {best_seasonal_order}")
        print(f"AIC: {best_aic:.2f}")

    # Финальная модель
    if verbose:
        print(f"\n{'─' * 70}")
        print("ОЦЕНКА ФИНАЛЬНОЙ МОДЕЛИ...")
        print(f"{'─' * 70}")

    if seasonal:
        final_model = SARIMAX(y_train, exog=exog_train,
                              order=best_order,
                              seasonal_order=best_seasonal_order)
    else:
        final_model = SARIMAX(y_train, exog=exog_train, order=best_order)

    final_fitted = final_model.fit(disp=False, maxiter=300)

    if verbose:
        print(final_fitted.summary().tables[1])

    # Прогноз
    forecast_result = final_fitted.get_forecast(steps=test_size, exog=exog_test)
    forecast = forecast_result.predicted_mean
    forecast_ci = forecast_result.conf_int(alpha=0.05)

    # Метрики
    mae = mean_absolute_error(y_test, forecast)
    rmse = np.sqrt(mean_squared_error(y_test, forecast))

    with np.errstate(divide='ignore', invalid='ignore'):
        mape = np.mean(np.abs((y_test - forecast) / np.where(np.abs(y_test) > 1e-10, y_test, np.nan))) * 100
        if np.isnan(mape):
            mape = np.nan

    if verbose:
        print(f"\n{'─' * 70}")
        print("МЕТРИКИ КАЧЕСТВА ПРОГНОЗА")
        print(f"{'─' * 70}")
        print(f"MAE:  {mae:.2f}")
        print(f"RMSE: {rmse:.2f}")
        if not np.isnan(mape):
            print(f"MAPE: {mape:.2f}%")

        print(f"\n{'─' * 70}")
        print("СРАВНЕНИЕ ПРОГНОЗА И ФАКТА")
        print(f"{'─' * 70}")
        for i in range(test_size):
            err = forecast[i] - y_test[i]
            print(f"  {_quarter_to_str(dates_test[i])}: факт={y_test[i]:>10.0f}, "
                  f"прогноз={forecast[i]:>10.0f}, ошибка={err:>+10.0f}")

    # Визуализация
    if verbose:
        fig, axes = plt.subplots(1, 2, figsize=(16, 5))

        ax1 = axes[0]
        ax1.plot(dates_train, y_train, 'b-', linewidth=2, label='Обучение')
        ax1.plot(dates_test, y_test, 'g-o', markersize=8, linewidth=2, label='Факт')
        ax1.plot(dates_test, forecast, 'r--o', markersize=8, linewidth=2, label='Прогноз')
        ax1.fill_between(dates_test, forecast_ci[:, 0], forecast_ci[:, 1],
                         alpha=0.2, color='red', label='95% ДИ')

        if break_dates:
            for bd in break_dates:
                year, q_str = bd.split('-Q')
                bdate = pd.Timestamp(year=int(year), month=(int(q_str) - 1) * 3 + 1, day=1)
                ax1.axvline(x=bdate, color='gray', linestyle=':', linewidth=1.5, alpha=0.7)

        ax1.set_title(f'{name_target}: Прогноз SARIMAX{"" if not seasonal else " (сезонная)"}')
        ax1.tick_params(axis='x', rotation=45)
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2 = axes[1]
        errors = forecast - y_test
        colors = ['#F44336' if e > 0 else '#4CAF50' for e in errors]
        ax2.bar(range(test_size), errors, color=colors, alpha=0.7)
        ax2.axhline(y=0, color='black', linewidth=0.8)
        ax2.set_title('Ошибки прогноза')
        ax2.set_xlabel('Шаг прогноза')
        ax2.set_ylabel('Ошибка')
        ax2.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.show()

    return {
        'name': name_target,
        'order': best_order,
        'seasonal_order': best_seasonal_order,
        'aic': best_aic,
        'forecast': forecast,
        'forecast_ci': forecast_ci,
        'actual': y_test,
        'mae': mae,
        'rmse': rmse,
        'mape': mape,
        'model': final_fitted,
        'aic': best_aic,
        'train_idx': train_idx,
        'y_train': y_train,
        'y_test': y_test,
        'dates_train': dates_train,
        'dates_test': dates_test,
    }


# ============================================================
# ЗАПУСК ДЛЯ ВСЕХ ПОКАЗАТЕЛЕЙ
# ============================================================

if __name__ == "__main__":

    print("\n" + "█" * 70)
    print("█  МОДЕЛИ SARIMAX ДЛЯ ВСЕХ ПОКАЗАТЕЛЕЙ")
    print("█" * 70)

    models_specs = {
        'DboFizLKolObs': {
            'file': 'DboFizLKolObs.xlsx',
            'breaks': ['2023-Q2', '2024-Q4'],
            'seasonal': False,
        },
        'DboFizObTic': {
            'file': 'DboFizObTic.xlsx',
            'breaks': ['2022-Q3', '2023-Q2', '2024-Q3'],
            'seasonal': False,
        },
        'InterResBezlicenzii': {
            'file': 'InterResBezlicenzii.xlsx',
            'breaks': ['2023-Q1', '2025-Q1'],
            'seasonal': False,
        },
        'InterResPiramid': {
            'file': 'InterResPiramid.xlsx',
            'breaks': ['2023-Q1'],
            'seasonal': False,
        },
        'MohenTel8800': {
            'file': 'MohenTel8800.xlsx',
            'breaks': None,
            'seasonal': False,
        },
        'MohenTelGorod': {
            'file': 'MohenTelGorod.xlsx',
            'breaks': ['2021-Q4', '2022-Q4'],
            'seasonal': False,
        },
        'MohenTelMobilka_base': {
            'file': 'MohenTelMobilka.xlsx',
            'breaks': ['2022-Q3', '2023-Q4'],
            'seasonal': False,
        },
        'MohenTelMobilka_migration': {
            'file': 'MohenTelMobilka.xlsx',
            'breaks': ['2022-Q3', '2023-Q4'],
            'donor_file': 'MohenTelGorod.xlsx',
            'donor_lags': [3, 4],
            'seasonal': False,
        },
        'ObhKartinaKolObs': {
            'file': 'ObhKartinaKolObs.xlsx',
            'breaks': ['2023-Q2'],
            'seasonal': False,
        },
        'ObhKartinaObTic': {
            'file': 'ObhKartinaObTic.xlsx',
            'breaks': ['2024-Q3'],
            'seasonal': True,
        },
    }

    results_all = {}

    for name, spec in models_specs.items():
        print(f"\n{'▶' * 35}")
        print(f"▶  Модель: {name}")
        print(f"{'▶' * 35}")

        try:
            result = build_sarimax_model(
                file_path=spec['file'],
                break_dates=spec.get('breaks'),
                donor_file=spec.get('donor_file'),
                donor_lags=spec.get('donor_lags'),
                seasonal=spec.get('seasonal', False),
                test_size=4,
                verbose=True,
            )
            results_all[name] = result
        except Exception as e:
            print(f"  ⚠️ Ошибка: {e}")

    # Сводная таблица
    if results_all:
        print(f"\n{'═' * 80}")
        print("СВОДНАЯ ТАБЛИЦА РЕЗУЛЬТАТОВ")
        print(f"{'═' * 80}")
        print(f"{'Модель':<32} {'Порядок':<15} {'RMSE':>12} {'MAPE':>10}")
        print(f"{'─' * 70}")

        for name, res in results_all.items():
            order_str = f"ARIMA{res['order']}"
            if res['seasonal_order']:
                order_str += f"xSARIMA{res['seasonal_order']}"
            mape_str = f"{res['mape']:.1f}%" if not np.isnan(res['mape']) else "N/A"
            print(f"{name:<32} {order_str:<15} {res['rmse']:>12.1f} {mape_str:>10}")