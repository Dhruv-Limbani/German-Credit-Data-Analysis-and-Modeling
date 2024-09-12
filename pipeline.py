import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle

df = pd.read_csv('Credit-Data-Cleaned.csv')
y = df['class']
x = df.drop(['class','housing','employment'],axis=1)
numerical_columns = x.select_dtypes(include=['int64', 'float64']).columns
categorical_columns = [a for a in x.columns if a not in numerical_columns]

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_columns),  # Normalize numerical data
        ('cat', OneHotEncoder(), categorical_columns)  # One-hot encode categorical data
    ],
    remainder='drop'
)

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', SVC(C= 1, kernel='rbf'))
])


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

pipeline.fit(x_train, y_train)

y_pred = pipeline.predict(x_test)
decoded_pred = label_encoder.inverse_transform(y_pred)

print("Accuracy Score:",accuracy_score(y_test,y_pred))

with open('pipeline.pkl', 'wb') as f:
    pickle.dump(pipeline, f)

with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)