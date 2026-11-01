import os
import pandas as pd
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder

def clean_and_prepare_data():
    # Define path to the local raw CSV file
    raw_data_path = "tourism_project/data/tourism.csv"

    # Load raw dataset, ignoring malformed rows containing unescaped quote anomalies
    df = pd.read_csv(raw_data_path, on_bad_lines="skip")

    # Drop unique record keys and administrative index columns
    unnecessary_cols = ["Unnamed: 0", "CustomerID"]
    df = df.drop(columns=[col for col in unnecessary_cols if col in df.columns], errors="ignore")

    # Standardize 'Gender' variations cleanly using case-insensitive regex
    if "Gender" in df.columns:
        df["Gender"] = df["Gender"].replace(to_replace=r'(?i).*fe\s*male.*', value='Female', regex=True)
        df["Gender"] = df["Gender"].replace(to_replace=r'(?i).*male.*', value='Male', regex=True)

    # Standardize spelling and spacing variations for 'Occupation'
    if "Occupation" in df.columns:
        df["Occupation"] = df["Occupation"].replace(to_replace=r'(?i).*free\s*lancer.*', value='Free Lancer', regex=True)
        df["Occupation"] = df["Occupation"].replace(to_replace=r'(?i).*small\s*business.*', value='Small Business', regex=True)
        df["Occupation"] = df["Occupation"].replace(to_replace=r'(?i).*large\s*business.*', value='Large Business', regex=True)
        df["Occupation"] = df["Occupation"].replace(to_replace=r'(?i).*salaried.*', value='Salaried', regex=True)

    # Standardize 'MaritalStatus' classifications, consolidating single/unmarried values
    if "MaritalStatus" in df.columns:
        df["MaritalStatus"] = df["MaritalStatus"].replace(to_replace=r'(?i).*unmarried.*', value='Single', regex=True)
        df["MaritalStatus"] = df["MaritalStatus"].replace(to_replace=r'(?i).*single.*', value='Single', regex=True)
        df["MaritalStatus"] = df["MaritalStatus"].replace(to_replace=r'(?i).*married.*', value='Married', regex=True)
        df["MaritalStatus"] = df["MaritalStatus"].replace(to_replace=r'(?i).*divorced.*', value='Divorced', regex=True)

    # Standardize spelling and spacing variations for 'TypeofContact'
    if "TypeofContact" in df.columns:
        df["TypeofContact"] = df["TypeofContact"].replace(to_replace=r'(?i).*self\s*inquiry.*', value='Self Inquiry', regex=True)
        df["TypeofContact"] = df["TypeofContact"].replace(to_replace=r'(?i).*company\s*invited.*', value='Company Invited', regex=True)

    # Standardize complex options for 'ProductPitched'
    if "ProductPitched" in df.columns:
        df["ProductPitched"] = df["ProductPitched"].replace(to_replace=r'(?i).*super\s*deluxe.*', value='Super Deluxe', regex=True)
        df["ProductPitched"] = df["ProductPitched"].replace(to_replace=r'(?i).*deluxe.*', value='Deluxe', regex=True)
        df["ProductPitched"] = df["ProductPitched"].replace(to_replace=r'(?i).*basic.*', value='Basic', regex=True)
        df["ProductPitched"] = df["ProductPitched"].replace(to_replace=r'(?i).*standard.*', value='Standard', regex=True)
        df["ProductPitched"] = df["ProductPitched"].replace(to_replace=r'(?i).*king.*', value='King', regex=True)

    # Standardize hierarchical options for 'Designation'
    if "Designation" in df.columns:
        df["Designation"] = df["Designation"].replace(to_replace=r'(?i).*senior\s*manager.*', value='Senior Manager', regex=True)
        df["Designation"] = df["Designation"].replace(to_replace=r'(?i).*manager.*', value='Manager', regex=True)
        df["Designation"] = df["Designation"].replace(to_replace=r'(?i).*executive.*', value='Executive', regex=True)
        df["Designation"] = df["Designation"].replace(to_replace=r'(?i).*avp.*', value='AVP', regex=True)
        df["Designation"] = df["Designation"].replace(to_replace=r'(?i).*vp.*', value='VP', regex=True)

    # Fill missing values dynamically: numerical with median, categorical with mode
    for col in df.columns:
        if np.issubdtype(df[col].dtype, np.number):
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else "Unknown")

    # Drop any records missing the target label and force target type to integer
    df = df.dropna(subset=["ProdTaken"])
    df["ProdTaken"] = df["ProdTaken"].astype(int)

    # Separate predictive features from the target variable
    X = df.drop(columns=["ProdTaken"])
    y = df["ProdTaken"]

    # Partition the dataset into stratified 80% train and 20% test splits
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Create a Column Transformer to handle one-hot encoding for object/string categories
    categorical_features = X.select_dtypes(include=['object', 'string']).columns.tolist()
    preprocessor = make_column_transformer(
        (OneHotEncoder(handle_unknown="ignore", drop="first", sparse_output=False), categorical_features),
        remainder="passthrough"
    )

    # Fit column transformer on training features and apply transformation to train/test splits
    X_train_transformed = preprocessor.fit_transform(X_train_raw)
    X_test_transformed = preprocessor.transform(X_test_raw)

    # Format encoded results back to clean pandas DataFrames
    feature_names = preprocessor.get_feature_names_out()
    X_train = pd.DataFrame(X_train_transformed, columns=feature_names)
    X_test = pd.DataFrame(X_test_transformed, columns=feature_names)

    # Export the fitted preprocessing pipeline for deployment/production predictions
    os.makedirs("tourism_project/deployment", exist_ok=True)
    joblib.dump(preprocessor, "tourism_project/deployment/preprocessor.joblib")

    # Save training and testing CSV data splits locally to their designated subdirectory
    splits_dir = "tourism_project/data_splits"
    os.makedirs(splits_dir, exist_ok=True)
    X_train.to_csv(os.path.join(splits_dir, "Xtrain.csv"), index=False)
    X_test.to_csv(os.path.join(splits_dir, "Xtest.csv"), index=False)
    y_train.to_csv(os.path.join(splits_dir, "ytrain.csv"), index=False)
    y_test.to_csv(os.path.join(splits_dir, "ytest.csv"), index=False)

    print("\nSuccessfully completed : \n  - Advanced data cleaning \n  - Anomaly resolution \n  - ColumnTransformer encoding")
    print(f"\nTrain features shape : {X_train.shape} \nTest features shape : {X_test.shape}")

if __name__ == "__main__":
    clean_and_prepare_data()
