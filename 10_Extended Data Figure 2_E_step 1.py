import pandas as pd

# ===============================
# Input and output paths
# ===============================
input_path = "0_dataset.csv"
output_path = "0_dataset_binary.csv"

# ===============================
# Load dataset
# ===============================
df = pd.read_csv(input_path, encoding="utf-8")

# ===============================
# Binary transformation
# Original scale: 1–5
# 1,2 -> 0 ; 3,4,5 -> 1
# ===============================
df['Y_binary'] = df['Y'].apply(lambda x: 1 if x >= 3 else 0)

# ===============================
# Save new dataset
# ===============================
df.to_csv(output_path, index=False, encoding="utf-8")

# ===============================
# Console output
# ===============================
print("Binary transformation completed")
print("Output file:", output_path)

print("\nClass distribution:")
print(df['Y_binary'].value_counts())
