import pandas as pd

df = pd.read_csv("your_file.csv", parse_dates=["date"])
df = df.sort_values("date")

df = df.set_index("date")