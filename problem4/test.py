import pandas as pd

BASELINE_FILE = "data_participant.csv"
INPUT_FILE = "test.csv"

id_name = "ID"
label_name = "default-payment-next-month"

df1 = pd.read_csv(INPUT_FILE)
df2 = pd.read_csv(BASELINE_FILE)
df2.drop(columns=[label_name], inplace=True)

df1 = pd.DataFrame({id_name: df1[id_name], label_name: df1[label_name]})

output_df = df2.merge(df1, left_on=id_name, right_on=id_name)
output_file = "to_submit.csv"
output_df.to_csv(output_file, index=False)