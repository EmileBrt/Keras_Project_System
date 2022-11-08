import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Make numpy values easier to read.
np.set_printoptions(precision=3, suppress=True,threshold=np.inf)
pd.options.display.max_rows = 999

# Make numpy values easier to read.
np.set_printoptions(precision=3, suppress=True,threshold=np.inf)
pd.options.display.max_rows = 999

df1 = pd.read_csv("UNSW_NB15_testing-set.csv")
df2 = pd.read_csv("UNSW_NB15_training-set.csv")

df = pd.concat([df1,df2])

print(df1.shape)
print(df2.shape)
print(df.shape)

col2encode = ["proto","service","state"]
ohe_col = pd.get_dummies(df[col2encode],columns=col2encode)
df = df.drop(columns=col2encode)
df = df.drop(columns=["attack_cat","id"])
df_ohe = pd.concat([df,ohe_col],axis=1)
d_attack = {0:False,1:True}

Y = df_ohe["label"].map(d_attack)
df_ohe = df_ohe.drop(columns=["label"])

model = keras.Sequential(name="my_sequential")

model.add(keras.Input(shape=(196,)))
model.add(keras.layers.Dense(64,use_bias=True))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.ReLU())
model.add(keras.layers.Dense(64,use_bias=True))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.ReLU())
model.add(keras.layers.Dense(64,use_bias=True))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.ReLU())
model.add(keras.layers.Dense(1,use_bias=True))

X_train, X_test, y_train, y_test = train_test_split(df_ohe, Y, test_size=0.25, random_state=42)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss=keras.losses.BinaryCrossentropy(),
    metrics =['accuracy']
)

model.fit(X_train, y_train, batch_size=64, epochs=10,validation_data=(X_train,y_train))

print("Evaluate on test data")
results = model.evaluate(X_test, y_test)
print("test acc:", results)
model.save("model")

