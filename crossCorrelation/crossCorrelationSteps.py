import pandas as pd

def data_normalization(data_frame):
    df_diff = data_frame.diff().dropna()
    print(f"✅ Успешно убрали тренд")
    # TODO зацикливается в нормализации
    df_norm = (df_diff - df_diff.mean()) / df_diff.std()
    print(f"✅ Нормализация прошла успешно")

    return df_norm


def cross_corr(x, y, lag_max=12):
    lags = range(-lag_max, lag_max + 1)
    corr = []

    for lag in lags:
        if lag < 0:
            corr.append(x[:lag].corr(y[-lag:]))
        elif lag > 0:
            corr.append(x[lag:].corr(y[:-lag]))
        else:
            corr.append(x.corr(y))

    print(f"✅ Корреляция прошла успешно")
    return pd.DataFrame({"lag": lags, "correlation": corr})


