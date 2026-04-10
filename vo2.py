import pandas as pd

df = pd.read_csv("./dielectricos_absolutamente_todo.csv")

vo2 = df[df["formula_pretty"] == "VO2"]
