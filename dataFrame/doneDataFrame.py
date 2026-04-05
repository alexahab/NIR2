from dataFrame.preparationDataFrame import *

# Подготовка данных для "мошенники звонят"
df_scammers_calling = create_dataframe_from_file("мошенники звонят.xlsx")
df_scammers_calling = df_scammers_calling.drop(columns='Доля от всех запросов, %')
df_scammers_calling = rename_columns(df_scammers_calling,"Период","date","Число запросов","number of requests")
df_scammers_calling = date_cutting(df_scammers_calling, "2018-01-01", "2026-01-01")

df_scammers_calling.info()
print("Начало",df_scammers_calling[0:1])
print("Конец", df_scammers_calling.tail())

print("-"*30)

# Подготовка данных для "изменения количества наличных денек"
df_change_amount_of_cash_in_circulation = create_dataframe_from_file("Изменение количества наличных денег в обращении.xlsx")
df_change_amount_of_cash_in_circulation = rename_columns(df_change_amount_of_cash_in_circulation,"Unnamed: 0","date","млрд руб.","cash billion rubles")
df_change_amount_of_cash_in_circulation = date_cutting(df_change_amount_of_cash_in_circulation, "2018-01-01", "2026-01-01")

df_change_amount_of_cash_in_circulation.info()
print("Начало",df_change_amount_of_cash_in_circulation[0:1])
print("Конец", df_change_amount_of_cash_in_circulation.tail())