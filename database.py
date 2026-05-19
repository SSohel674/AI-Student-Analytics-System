import pandas as pd

def load_data():
    df = pd.read_csv("student_data.csv")
    return df