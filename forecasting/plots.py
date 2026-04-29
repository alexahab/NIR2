import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import FancyBboxPatch
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
    future_dates = []
    current = last_date + pd.DateOffset(months=3)
    for _ in range(n_future):
        future_dates.append(current)
        current = current + pd.DateOffset(months=3)
    return pd.DatetimeIndex(future_dates)


# ============================================================
# ДАННЫЕ ПРОГНОЗОВ (из предыдущего запуска)
# ============================================================

# Исправленные прогнозы с ограничениями
forecasts_data = {
    'DboFizLKolObs': {
        'file': 'DboFizLKolObs.xlsx',
        'forecast': np.array([98209, 98209, 98209, 98209]),  # Naïve(last)
        'model': 'Naïve(last)',
        'description': 'Физ. лица, количество',
        'group': 'Физические лица',
    },
    'DboFizObTic': {
        'file': 'DboFizObTic.xlsx',
        'forecast': np.array([2742636, 2808217, 2873799, 2939381]),
        'model': 'Theta',
        'description': 'Физ. лица, суммы',
        'group': 'Физические лица',
    },
    'InterResBezlicenzii': {
        'file': 'InterResBezlicenzii.xlsx',
        'forecast': np.array([3280, 3451, 3622, 3794]),
        'model': 'Theta',
        'description': 'Интернет-мошенничество',
        'group': 'Интернет-каналы',
    },
    'InterResPiramid': {
        'file': 'InterResPiramid.xlsx',
        'forecast': np.array([1922, 1922, 1922, 1922]),
        'model': 'Naïve(last)',
        'description': 'Финансовые пирамиды',
        'group': 'Интернет-каналы',
    },
    'MohenTel8800': {
        'file': 'MohenTel8800.xlsx',
        'forecast': np.array([133, 133, 133, 133]),
        'model': 'Naïve(last)',
        'description': 'Телефонное (8-800)',
        'group': 'Телефонные каналы',
    },
    'MohenTelGorod': {
        'file': 'MohenTelGorod.xlsx',
        'forecast': np.array([469, 274, 78, 0]),
        'model': 'Naïve(drift)',
        'description': 'Телефонное (городские)',
        'group': 'Телефонные каналы',
    },
    'MohenTelMobilka': {
        'file': 'MohenTelMobilka.xlsx',
        'forecast': np.array([11726, 7750, 3484, 0]),
        'model': 'ARIMA(1,2,1)',
        'description': 'Телефонное (мобильные)',
        'group': 'Телефонные каналы',
    },
    'ObhKartinaKolObs': {
        'file': 'ObhKartinaKolObs.xlsx',
        'forecast': np.array([326734, 330664, 334594, 338524]),
        'model': 'ETS',
        'description': 'Агрегат, количество',
        'group': 'Агрегаты',
    },
    'ObhKartinaObTic': {
        'file': 'ObhKartinaObTic.xlsx',
        'forecast': np.array([8123709, 8414331, 8704953, 8995575]),
        'model': 'Theta',
        'description': 'Агрегат, суммы',
        'group': 'Агрегаты',
    },
}

# Загружаем исторические данные
historical_data = {}
for name, spec in forecasts_data.items():
    y, dates_pd, _ = _load_data(spec['file'])
    historical_data[name] = {
        'y': y,
        'dates': dates_pd,
        'last_date': dates_pd[-1],
    }


# ============================================================
# ГРАФИК 1: ТЕЛЕФОННЫЕ КАНАЛЫ
# ============================================================

def plot_phone_channels():
    fig, ax = plt.subplots(figsize=(16, 7))
    fig.suptitle('ТЕЛЕФОННЫЕ КАНАЛЫ: история и прогноз (2025-Q4 – 2026-Q3)',
                 fontsize=15, fontweight='bold')

    channels = ['MohenTelGorod', 'MohenTelMobilka', 'MohenTel8800']
    colors = ['#F44336', '#FF9800', '#2196F3']
    labels = ['Городские номера', 'Мобильные номера', '8-800']

    for ch, color, label in zip(channels, colors, labels):
        hist = historical_data[ch]
        fc = forecasts_data[ch]

        # История
        ax.plot(hist['dates'], hist['y'], '-', color=color, linewidth=2.5, label=f'{label} (история)')

        # Прогноз
        future_dates = _get_future_dates(hist['last_date'], 4)
        all_dates = [hist['dates'][-1]] + list(future_dates)
        all_values = [hist['y'][-1]] + list(fc['forecast'])
        ax.plot(all_dates, all_values, '--o', color=color, linewidth=2.5, markersize=8,
                label=f'{label} (прогноз: {fc["model"]})')

    # Разделительная линия
    ax.axvline(x=historical_data['MohenTelGorod']['last_date'], color='black',
               linestyle=':', linewidth=2, alpha=0.7, label='Начало прогноза')

    # Аннотации
    ax.annotate('Городские: ☠️\nполное исчезновение',
                xy=(_get_future_dates(historical_data['MohenTelGorod']['last_date'], 4)[-1], 100),
                fontsize=10, color='#F44336', fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='mistyrose', alpha=0.8))

    ax.annotate('Мобильные: ⚠️\nпродолжение сжатия',
                xy=(_get_future_dates(historical_data['MohenTelMobilka']['last_date'], 4)[1], 8000),
                fontsize=10, color='#FF9800', fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    ax.annotate('8-800: 🔄\nфоновый уровень',
                xy=(_get_future_dates(historical_data['MohenTel8800']['last_date'], 4)[2], 200),
                fontsize=10, color='#2196F3', fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

    ax.set_ylabel('Количество')
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='x', rotation=45)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:,.0f}'))

    plt.tight_layout()
    plt.show()
    return fig


# ============================================================
# ГРАФИК 2: ИНТЕРНЕТ-КАНАЛЫ
# ============================================================

def plot_internet_channels():
    fig, ax = plt.subplots(figsize=(16, 7))
    fig.suptitle('ИНТЕРНЕТ-КАНАЛЫ: история и прогноз (2025-Q4 – 2026-Q3)',
                 fontsize=15, fontweight='bold')

    channels = ['InterResBezlicenzii', 'InterResPiramid']
    colors = ['#4CAF50', '#9C27B0']
    labels = ['Интернет-мошенничество', 'Финансовые пирамиды']

    for ch, color, label in zip(channels, colors, labels):
        hist = historical_data[ch]
        fc = forecasts_data[ch]

        # История
        ax.plot(hist['dates'], hist['y'], '-', color=color, linewidth=2.5, label=f'{label} (история)')

        # Прогноз
        future_dates = _get_future_dates(hist['last_date'], 4)
        all_dates = [hist['dates'][-1]] + list(future_dates)
        all_values = [hist['y'][-1]] + list(fc['forecast'])
        ax.plot(all_dates, all_values, '--o', color=color, linewidth=2.5, markersize=8,
                label=f'{label} (прогноз: {fc["model"]})')

    # Разделительная линия
    ax.axvline(x=historical_data['InterResBezlicenzii']['last_date'], color='black',
               linestyle=':', linewidth=2, alpha=0.7, label='Начало прогноза')

    # Аннотации
    ax.annotate('Интернет: 📈\nустойчивый рост (+16% за год)',
                xy=(_get_future_dates(historical_data['InterResBezlicenzii']['last_date'], 4)[-1], 3600),
                fontsize=10, color='#4CAF50', fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='honeydew', alpha=0.8))

    ax.annotate('Пирамиды: 🔄\nстабилизация после пика',
                xy=(_get_future_dates(historical_data['InterResPiramid']['last_date'], 4)[2], 2000),
                fontsize=10, color='#9C27B0', fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='lavender', alpha=0.8))

    ax.set_ylabel('Количество')
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='x', rotation=45)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:,.0f}'))

    plt.tight_layout()
    plt.show()
    return fig


# ============================================================
# ГРАФИК 3: ФИЗИЧЕСКИЕ ЛИЦА
# ============================================================

def plot_individuals():
    fig, axes = plt.subplots(2, 1, figsize=(16, 10))
    fig.suptitle('ФИЗИЧЕСКИЕ ЛИЦА: история и прогноз (2025-Q4 – 2026-Q3)',
                 fontsize=15, fontweight='bold')

    # График 1: Количество
    ax1 = axes[0]
    ch = 'DboFizLKolObs'
    hist = historical_data[ch]
    fc = forecasts_data[ch]

    ax1.plot(hist['dates'], hist['y'], 'b-', linewidth=2.5, label='История')
    future_dates = _get_future_dates(hist['last_date'], 4)
    all_dates = [hist['dates'][-1]] + list(future_dates)
    all_values = [hist['y'][-1]] + list(fc['forecast'])
    ax1.plot(all_dates, all_values, 'r--o', linewidth=2.5, markersize=8,
             label=f'Прогноз: {fc["model"]}')
    ax1.axvline(x=hist['last_date'], color='black', linestyle=':', linewidth=2, alpha=0.7)
    ax1.set_title(f'{fc["description"]}: {fc["model"]}', fontsize=12)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='x', rotation=45)
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:,.0f}'))

    ax1.annotate('⚠️ Прогноз = константа\n(последнее значение = выброс)',
                 xy=(future_dates[1], 100000), fontsize=10, color='red', fontweight='bold',
                 bbox=dict(boxstyle='round', facecolor='mistyrose', alpha=0.8))

    # График 2: Суммы
    ax2 = axes[1]
    ch = 'DboFizObTic'
    hist = historical_data[ch]
    fc = forecasts_data[ch]

    ax2.plot(hist['dates'], hist['y'], 'b-', linewidth=2.5, label='История')
    future_dates = _get_future_dates(hist['last_date'], 4)
    all_dates = [hist['dates'][-1]] + list(future_dates)
    all_values = [hist['y'][-1]] + list(fc['forecast'])
    ax2.plot(all_dates, all_values, 'g--o', linewidth=2.5, markersize=8,
             label=f'Прогноз: {fc["model"]}')
    ax2.axvline(x=hist['last_date'], color='black', linestyle=':', linewidth=2, alpha=0.7)
    ax2.set_title(f'{fc["description"]}: {fc["model"]} (MAPE 16.1% — лучшая модель)', fontsize=12)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis='x', rotation=45)
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:,.0f}'))

    ax2.annotate('✅ Плавный рост\nTheta отлично\nсправляется',
                 xy=(future_dates[2], 2850000), fontsize=10, color='green', fontweight='bold',
                 bbox=dict(boxstyle='round', facecolor='honeydew', alpha=0.8))

    plt.tight_layout()
    plt.show()
    return fig


# ============================================================
# ГРАФИК 4: АГРЕГАТЫ
# ============================================================

def plot_aggregates():
    fig, axes = plt.subplots(2, 1, figsize=(16, 10))
    fig.suptitle('АГРЕГИРОВАННЫЕ ПОКАЗАТЕЛИ: история и прогноз (2025-Q4 – 2026-Q3)',
                 fontsize=15, fontweight='bold')

    # График 1: Количество
    ax1 = axes[0]
    ch = 'ObhKartinaKolObs'
    hist = historical_data[ch]
    fc = forecasts_data[ch]

    ax1.plot(hist['dates'], hist['y'], 'b-', linewidth=2.5, label='История')
    future_dates = _get_future_dates(hist['last_date'], 4)
    all_dates = [hist['dates'][-1]] + list(future_dates)
    all_values = [hist['y'][-1]] + list(fc['forecast'])
    ax1.plot(all_dates, all_values, 'purple', linestyle='--', marker='o', linewidth=2.5, markersize=8,
             label=f'Прогноз: {fc["model"]}')
    ax1.axvline(x=hist['last_date'], color='black', linestyle=':', linewidth=2, alpha=0.7)
    ax1.set_title(f'{fc["description"]}: {fc["model"]} (MAPE 11.4% — лучший результат!)', fontsize=12)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='x', rotation=45)
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:,.0f}'))

    ax1.annotate('✅✅✅ Отличный прогноз\nETS — лучшая модель\n(MAPE 11.4%)',
                 xy=(future_dates[2], 332000), fontsize=10, color='purple', fontweight='bold',
                 bbox=dict(boxstyle='round', facecolor='lavender', alpha=0.8))

    # График 2: Суммы
    ax2 = axes[1]
    ch = 'ObhKartinaObTic'
    hist = historical_data[ch]
    fc = forecasts_data[ch]

    ax2.plot(hist['dates'], hist['y'], 'b-', linewidth=2.5, label='История')
    future_dates = _get_future_dates(hist['last_date'], 4)
    all_dates = [hist['dates'][-1]] + list(future_dates)
    all_values = [hist['y'][-1]] + list(fc['forecast'])
    ax2.plot(all_dates, all_values, 'orange', linestyle='--', marker='o', linewidth=2.5, markersize=8,
             label=f'Прогноз: {fc["model"]}')
    ax2.axvline(x=hist['last_date'], color='black', linestyle=':', linewidth=2, alpha=0.7)
    ax2.set_title(f'{fc["description"]}: {fc["model"]} (MAPE 18.5%)', fontsize=12)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis='x', rotation=45)
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:,.0f}'))

    ax2.annotate('✅ Умеренный рост\n+11% за год',
                 xy=(future_dates[2], 8600000), fontsize=10, color='orange', fontweight='bold',
                 bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout()
    plt.show()
    return fig


# ============================================================
# ГРАФИК 5: СВОДНАЯ ДИАГРАММА МИГРАЦИИ
# ============================================================

def plot_migration_summary():
    fig, ax = plt.subplots(figsize=(18, 8))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')
    fig.suptitle('ПРОГНОЗ ЭКОСИСТЕМЫ КИБЕРПРЕСТУПНОСТИ: 2025-Q4 – 2026-Q3',
                 fontsize=16, fontweight='bold', y=0.98)

    # Блоки
    blocks = [
        {'xy': (5, 55), 'width': 25, 'height': 35, 'color': '#FFCDD2', 'label': 'ТЕЛЕФОННЫЕ\nКАНАЛЫ\n(сжимаются)',
         'text_color': '#C62828'},
        {'xy': (35, 55), 'width': 25, 'height': 35, 'color': '#C8E6C9', 'label': 'ИНТЕРНЕТ-\nКАНАЛЫ\n(растут)',
         'text_color': '#2E7D32'},
        {'xy': (65, 55), 'width': 30, 'height': 35, 'color': '#E3F2FD', 'label': 'АГРЕГАТЫ\n(растут)',
         'text_color': '#1565C0'},
    ]

    for block in blocks:
        rect = FancyBboxPatch(block['xy'], block['width'], block['height'],
                              boxstyle="round,pad=1", facecolor=block['color'],
                              edgecolor='black', linewidth=2, alpha=0.8)
        ax.add_patch(rect)
        ax.text(block['xy'][0] + block['width'] / 2, block['xy'][1] + block['height'] / 2,
                block['label'], ha='center', va='center', fontsize=11, fontweight='bold',
                color=block['text_color'])

    # Стрелки миграции
    ax.annotate('', xy=(37, 72), xytext=(28, 72),
                arrowprops=dict(arrowstyle='->', color='red', lw=3, alpha=0.6))
    ax.text(32, 74, 'Миграция\n(лаг 3-4 кв.)', ha='center', fontsize=9, color='red', fontweight='bold')

    # Стрелка "НЕТ МИГРАЦИИ"
    ax.annotate('', xy=(67, 72), xytext=(58, 72),
                arrowprops=dict(arrowstyle='->', color='gray', lw=2, linestyle='dashed'))
    ax.text(62, 74, 'НЕТ\nмиграции', ha='center', fontsize=8, color='gray')

    # Прогнозы внутри блоков
    phone_forecasts = [
        'Городские: ☠️ 469 → 0',
        'Мобильные: ⚠️ 12K → 0',
        '8-800: 🔄 133 (const)',
    ]
    for i, text in enumerate(phone_forecasts):
        ax.text(7, 72 - i * 6, text, fontsize=9, color='#C62828', fontweight='bold')

    internet_forecasts = [
        'Интернет: 📈 3.3K → 3.8K',
        'Пирамиды: 🔄 1.9K (const)',
    ]
    for i, text in enumerate(internet_forecasts):
        ax.text(37, 72 - i * 6, text, fontsize=9, color='#2E7D32', fontweight='bold')

    aggregate_forecasts = [
        'Кол-во: 📈 327K → 339K',
        'Суммы: 📈 8.1M → 9.0M',
    ]
    for i, text in enumerate(aggregate_forecasts):
        ax.text(67, 72 - i * 6, text, fontsize=9, color='#1565C0', fontweight='bold')

    # Легенда внизу
    ax.text(50, 30, 'КЛЮЧЕВЫЕ ВЫВОДЫ', ha='center', fontsize=14, fontweight='bold', color='black')

    conclusions = [
        '1. Телефонное мошенничество сжимается: городские номера исчезают, мобильные падают.',
        '2. Интернет-мошенничество устойчиво растёт (+16% за год), пирамиды стабилизировались.',
        '3. Агрегированные показатели растут — киберпреступность не снижается, а мигрирует.',
        '4. Миграция подтверждена внутри телефонной подсистемы, но не между телефонной и интернет.',
        '5. Простые модели (Theta, ETS, Naïve) превосходят сложные (SARIMAX) на коротких рядах.',
    ]

    for i, text in enumerate(conclusions):
        ax.text(50, 25 - i * 5, text, ha='center', fontsize=9, color='black')

    plt.tight_layout()
    plt.show()
    return fig


# ============================================================
# ЗАПУСК
# ============================================================

if __name__ == "__main__":
    print("Построение графиков...")

    plot_phone_channels()
    plot_internet_channels()
    plot_individuals()
    plot_aggregates()
    plot_migration_summary()

    print("Готово!")