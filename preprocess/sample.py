import pandas as pd
from sklearn.model_selection import train_test_split

# Load the dataset
# file_path = "/mnt/sdc/rjin02/MedVLMBench/data/camelyon17_v1.0/test_metadata.csv"
file_path = "/mnt/sdc/rjin02/MedVLMBench/data/camelyon17_v1.0/train_metadata.csv"
df = pd.read_csv(file_path)

# Perform stratified sampling based on the 'tumor' column
_, test_df = train_test_split(df, test_size=10000, stratify=df['tumor'], random_state=42)

test_df.to_csv("/mnt/sdc/rjin02/MedVLMBench/data/camelyon17_v1.0/sample_train_metadata.csv")