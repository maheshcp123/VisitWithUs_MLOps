import pandas as pd
import os

def validate():
    df = pd.read_csv('data/tourism_dataset.csv')
    expected_columns = ['CustomerID', 'ProdTaken', 'Age', 'TypeofContact', 'CityTier', 'Occupation', 'Gender', 'NumberOfPersonVisiting', 'PreferredPropertyStar', 'MaritalStatus', 'NumberOfTrips', 'Passport', 'OwnCar', 'NumberOfChildrenVisiting', 'Designation', 'MonthlyIncome', 'PitchSatisfactionScore', 'ProductPitched', 'NumberOfFollowups', 'DurationOfPitch']
    
    # Check columns
    missing = [col for col in expected_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    
    print("Data Summary:")
    print(df.info())
    print(f"Total Records: {len(df)}")

if __name__ == "__main__":
    validate()
