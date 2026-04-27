from cointegration.cointegrationSteps import *
from dataFrame.doneDataFrame import *
from dataFrame.preparationDataFrame import *

# Реализация коинтеграции между "мошенники звонят" и "изменения количества наличных денек"

df_scammers_call_and_change_of_cash = data_frames_join(df_scammers_calling, df_change_amount_of_cash_in_circulation)
adf_test(df_scammers_call_and_change_of_cash["number of requests"], "number of requests (level)")
adf_test(df_scammers_call_and_change_of_cash["cash billion rubles"], "cash billion rubles (level)")


result = engle_granger_cointegration(df_scammers_call_and_change_of_cash["number of requests"], df_scammers_call_and_change_of_cash["cash billion rubles"])
if result['is_cointegrated']:
    print(f"\nУравнение: Наличные = {result['cointegration_eq']['slope']:.4f} × Звонки + {result['cointegration_eq']['intercept']:.4f}")

plot_original_time_series(df_scammers_call_and_change_of_cash, "number of requests","Запросы 'мошенники звонят'", "cash billion rubles", "Изменения количества наличных денег")