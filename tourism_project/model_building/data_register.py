import pandas as pd

# Define the local path where the raw dataset is stored
RAW_PATH = "tourism_project/data/tourism.csv"

# Load the raw dataset. If there are any rows that contain data anomalies like
# unescaped embedded quotes (e.g. "xxxx"), which breaks
# CSV tokenization, then on_bad_lines="skip" drops just those malformed rows
# instead of failing the whole pipeline; the count skipped is reported below.

# Pre-calculate the total number of lines in the CSV to compare with successfully loaded rows
rows_before = sum(1 for _ in open(RAW_PATH, encoding="utf-8", errors="replace")) - 1
df_tourism = pd.read_csv(RAW_PATH, on_bad_lines="skip")
rows_skipped = rows_before - df_tourism.shape[0]

# Report whether any malformed rows were skipped during the parsing phase
if rows_skipped:
    print(f"\nSkipped {rows_skipped} malformed row(s) while parsing {RAW_PATH}.")
else:
    print("\nAll rows read successfully, no malformed rows found while parsing")

# Define the schema of expected features and labels required by the downstream modeling tasks
expected_columns = [
    "CustomerID", "ProdTaken", "Age", "TypeofContact", "CityTier",
    "DurationOfPitch", "Occupation", "Gender", "NumberOfPersonVisiting",
    "NumberOfFollowups", "ProductPitched", "PreferredPropertyStar",
    "MaritalStatus", "NumberOfTrips", "Passport", "PitchSatisfactionScore",
    "OwnCar", "NumberOfChildrenVisiting", "Designation", "MonthlyIncome"
]

# Compare the dataset's columns against the expected list to detect missing features early
missing = [c for c in expected_columns if c not in df_tourism.columns]

if missing:
    raise ValueError(f"\nDataset is missing expected columns: {missing}")
else:
    print("\nDataset is not missing any of the expected columns")

# Output a summary registration report detailing shape, schema, and label balance
print("\nDataset registered successfully.")
print(f"\nRows : {df_tourism.shape[0]}, Columns : {df_tourism.shape[1]}")
print("\nColumns :", list(df_tourism.columns))
print("\nTarget column preview (ProdTaken) :\n")
print(df_tourism["ProdTaken"].describe())
