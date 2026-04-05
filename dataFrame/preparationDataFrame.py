import pandas as pd
import os

def create_dataframe_from_file(file_name, folder_name="dataExcel"):
    # Получаем директорию, где находится текущий скрипт
    base_path = r"C:\Users\aleks\PycharmProjects\NIR2\dataFrame"

    file_path = os.path.join(base_path, folder_name, file_name)
    df = pd.read_excel(file_path)
    return df

def rename_columns(data_frame, name_first_columns, new_name_first_columns, name_second_columns, new_name_second_columns):
    df = data_frame.rename(columns={
        name_first_columns: new_name_first_columns,
        name_second_columns: new_name_second_columns
    })
    print(f"✅ Колонки переименованы")

    transformation_type_data(df)

    return df

def date_cutting(data_frame, start, end):
    mask = (data_frame['date'] >= start) & (data_frame['date'] <= end)
    df = data_frame.loc[mask]

    return df

def transformation_type_data(data_frame):
    if 'date' in data_frame.columns:
        data_frame['date'] = pd.to_datetime(data_frame['date'])
        print(f"✅ Колонка 'date' преобразована в datetime")

    return data_frame

def data_frames_join(df_first, df_second, how="inner"):
    result = pd.merge(df_first, df_second, on='date', how=how)
    result = result.sort_values('date').reset_index(drop=True)
    print(f"✅ Объединение прошло успешно")

    return result