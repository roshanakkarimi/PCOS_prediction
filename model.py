import numpy as np
import pandas as pd

#sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import f_classif, VarianceThreshold
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

#XGBoost
import xgboost as xgb

#TF
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from tensorflow.keras.regularizers import l2
##################################################################

##Feature handling (engineering and selection)
df = pd.read_csv('data/PCOS_Cleaned_Data.csv')

# Separate features and target
X = df.drop(["PCOS (Y/N)", "Patient File No."], axis=1)  # Drop non-predictive columns
y = df["PCOS (Y/N)"]

#train_test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)

# Normalize/Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#Remove Constant, Quasi Constant, and Duplicated Features
constant_filter = VarianceThreshold(threshold=0.01)
constant_filter.fit(X_train)
X_train_filter = constant_filter.transform(X_train)
X_test_filter = constant_filter.transform(X_test)

X_train_T = X_train_filter.T
X_test_T = X_test_filter.T

X_train_T = pd.DataFrame(X_train_T)
X_test_T = pd.DataFrame(X_test_T)

duplicated_features = X_train_T.duplicated()
features_to_keep = [not index for index in duplicated_features]
X_train_unique = X_train_T[features_to_keep].T
X_test_unique = X_test_T[features_to_keep].T

# F_test
scores = f_classif(X_train_unique, y_train)
p_values = pd.Series(scores[1])
p_values.index = X_train_unique.columns
p_values.sort_values(ascending=True, inplace=True)
#p_values.plot.bar(figsize=(16, 5))

#Select features with p_value < 0.05
p_values = p_values[p_values<0.05]
X_train_p = X_train_unique[p_values.index]
X_test_p = X_test_unique[p_values.index]


##Model training
def run_random_forest(X_train, X_test, y_train, y_test):
    clf = RandomForestClassifier(n_estimators=100, random_state=0)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print('RF Accuracy:',accuracy_score(y_test, y_pred))

def run_XGBoost(X_train, X_test, y_train, y_test):
    cls = xgb.XGBClassifier(n_estimators=100, random_state=0)
    cls.fit(X_train, y_train)
    y_pred = cls.predict(X_test)
    print('XGBoost Accuracy:',accuracy_score(y_test, y_pred))

def run_NN(X_train, X_test, y_train, y_test):
    X_val, X_train, y_val, y_train = train_test_split(
        X_train, y_train, test_size=0.5, stratify=y_train, random_state=42
    )

    # Build the Neural Network
    model = Sequential()
    model.add(Dense(32, input_dim=X_train.shape[1], activation='relu', kernel_regularizer=l2(0.01)))  # Input layer
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))  # Output layer

    # Compile the model
    model.compile(optimizer=Adam(learning_rate=0.05), loss='binary_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=32,
        verbose=1
    )

    # Evaluate the model on the test set
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Accuracy: {test_accuracy:.4f}")
