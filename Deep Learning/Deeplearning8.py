import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical

train_df = pd.read_csv("F:/bank-full.csv",sep=';')
test_df  = pd.read_csv("F:/bank.csv", sep=';')

target_column = "y"

X_train_raw = train_df.drop(target_column, axis=1)
y_train_raw = train_df[target_column]

X_test_raw = test_df.drop(target_column, axis=1)
y_test_raw = test_df[target_column]

combined_X = pd.concat([X_train_raw, X_test_raw], ignore_index=True)

combined_X_encoded = pd.get_dummies(combined_X, drop_first=True)

X_train = combined_X_encoded.iloc[:len(X_train_raw)]
X_test = combined_X_encoded.iloc[len(X_train_raw):]

encoder = LabelEncoder()
y_train = encoder.fit_transform(y_train_raw)
y_test  = encoder.transform(y_test_raw)

num_classes = len(np.unique(y_train))
y_train = to_categorical(y_train, num_classes)
y_test  = to_categorical(y_test, num_classes)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

model = Sequential([
    Dense(64, activation='tanh', input_shape=(X_train.shape[1],)),
    Dense(32, activation='tanh'),
    Dense(num_classes, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(
    X_train,
    y_train,
    epochs=50,
    batch_size=32,
    verbose=0
)

y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

print("Accuracy:", accuracy_score(y_true, y_pred_classes))
print(classification_report(y_true, y_pred_classes))
