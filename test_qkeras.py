from tensorflow import *
from qkeras import quantized_relu
from qkeras import QDense, QActivation
from qkeras import QBatchNormalization
from qkeras import quantized_bits

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Make numpy values easier to read.
np.set_printoptions(precision=3, suppress=True,threshold=np.inf)
pd.options.display.max_rows = 999

df = pd.read_csv("kddcup99_csv.csv")
col2encode = ["protocol_type","service","flag"]
ohe_col = pd.get_dummies(df[col2encode],columns=col2encode)
df = df.drop(columns=col2encode)
df_ohe = pd.concat([df,ohe_col],axis=1)

attack_values = pd.unique(df_ohe["label"].values.ravel())
attack_values = attack_values[1:] ## Remove "normal" as it is not an attack

### Create a dict to remap attack to True and normal to False
d_attack = {"normal":False}
for a in attack_values:
    d_attack.update({a:True})

label = df_ohe["label"].map(d_attack)
df_ohe = df_ohe.drop(columns="label",axis=1)

model = keras.Sequential(name="my_sequential")
model.add(keras.Input(shape=(118,)))
model.add(QDense(64,kernel_quantizer=quantized_bits(6,0,alpha=1)))
model.add(QBatchNormalization())
model.add(keras.layers.Dropout(0.5))
model.add(quantized_relu(6,0))
model.add(QDense(64,kernel_quantizer=quantized_bits(6,0,alpha=1)))
model.add(QBatchNormalization())
model.add(keras.layers.Dropout(0.5))
model.add(quantized_relu(6,0))
model.add(QDense(64,kernel_quantizer=quantized_bits(6,0,alpha=1)))
model.add(QBatchNormalization())
model.add(keras.layers.Dropout(0.5))
model.add(quantized_relu(6,0))
model.add(QDense(1,kernel_quantizer=quantized_bits(6,0,alpha=1)))

X_train, X_test, y_train, y_test = train_test_split(df_ohe, label, test_size=0.25, random_state=42)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss=keras.losses.BinaryCrossentropy(),
    metrics =['accuracy']
)

model.fit(X_train, y_train, batch_size=64, epochs=10,validation_data=(X_test,y_test))

print("Evaluate on test data")
results = model.evaluate(X_test, y_test)
print("test acc:", results)
#model.save("model")

