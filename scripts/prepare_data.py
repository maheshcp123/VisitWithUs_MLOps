import pandas as pd
from sklearn.model_selection import train_test_split

def prepare():
    df = pd.read_csv('data/tourism_dataset.csv')
    
    # Cleaning: Basic  imputation
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['MonthlyIncome'] = df['MonthlyIncome'].fillna(df['MonthlyIncome'].median())
    
    # Feature Selection: Dropping unnecessary columns
    df = df.drop(['CustomerID'], axis=1)
    
    # Split
    X = df.drop('ProdTaken', axis=1)
    y = df['ProdTaken']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Save locally for the workflow artifact
    X_train.to_csv('X_train.csv', index=False)
    X_test.to_csv('X_test.csv', index=False)
    y_train.to_csv('y_train.csv', index=False)
    y_test.to_csv('y_test.csv', index=False)

if __name__ == "__main__":
    prepare()
