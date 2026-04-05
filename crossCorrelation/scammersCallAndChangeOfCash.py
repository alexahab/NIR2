from crossCorrelation.crossCorrelationSteps import *
from crossCorrelation.metricsCorrelation.metricsSteps import *
from dataFrame.doneDataFrame import *

# Реализация кросс-корреляции между "мошенники звонят" и "изменения количества наличных денек"

df_scammers_call_and_change_of_cash = data_frames_join(df_scammers_calling, df_change_amount_of_cash_in_circulation, how="inner")
# df_scammers_call_and_change_of_cash.info()
# print("Начало",df_scammers_call_and_change_of_cash[0:1])
# print("Конец", df_scammers_call_and_change_of_cash.tail())

# Убираем тренд и нормализуем данные
df_scammers_and_cash_norm = data_normalization(df_scammers_call_and_change_of_cash)

# Корреляция
corr_scammers_and_cash = cross_corr(df_scammers_and_cash_norm["number of requests"], df_scammers_and_cash_norm["cash billion rubles"], lag_max=12)
# Метрики
results = analyze_cross_correlation(corr_scammers_and_cash, df_scammers_and_cash_norm)
# График
plot_ccf(corr_scammers_and_cash,"мошенники звонят", "наличные", conf_level=0.15)