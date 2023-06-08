# CNN--LSTM
import pandas as pd
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, LSTM, Dense

# Step 1: Input the trained data as a CSV file from datasets.
train_data = pd.read_csv("IoT-23_train.csv")

# Step 2: Scaling and transforming all the data features for learning.
scaler = StandardScaler()
train_data_scaled = scaler.fit_transform(train_data)

# Step 3: Scaling and transforming all features of the testing data.
test_data = pd.read_csv("CTU-13_test.csv")
test_data_scaled = scaler.transform(test_data)

# Step 4: Using SMOTE for data balancing.
smote = SMOTE()
train_data_balanced, train_labels_balanced = smote.fit_resample(train_data_scaled, train_labels)

# Step 5: Definition of the CNN-LSTM model.
def create_model():
    model = Sequential()
    # The body of the model (layers)
    model.add(Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(input_shape)))
    model.add(LSTM(units=64))
    model.add(Dense(units=num_classes, activation='softmax'))
    return model

# Step 6: Return model
model = create_model()

# Step 8: Model ← CNN− LSTM()Model
model = create_model()

# Step 9: Fitting function
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_data_balanced, train_labels_balanced, epochs=num_epochs, batch_size=batch_size)

# Step 10: Valuation calculation
loss, accuracy = model.evaluate(test_data_scaled, test_labels)![image](https://github.com/adjenna/CNN--LSTM/assets/135993809/6c5f4974-ab5e-4e89-bae0-777a50411b57)
![C L](https://github.com/adjenna/CNN--LSTM/assets/135993809/e40369c9-b7f7-4466-abfc-07e285476343)
