import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np
from sklearn.ensemble import RandomForestRegressor

def ANN_algo(predictArray):
    # print(">>>>>>>>>. running", predictArray, len(predictArray))
    # predictArray = np.array(predictArray)
    # predictArray.reshape(1, 46)

    series_obj = pd.Series(predictArray)
    arr = series_obj.values

    reshapedArray = arr.reshape(1,46)

    print('>>>>>>>>>>>>> ', reshapedArray.shape)
    # pred_df = pd.DataFrame(predictArray)
    try:
        df = pd.read_csv("E:/College Project/Final_Data.csv")
        df2 = df.copy()
        # df['Yield_'] = df['Yeild_Cat']
        df['Yield_'] = df['Yield']

        del df['Yield']
        del df['Yeild_Cat']

        X = df.iloc[:, 2:-1]
        Y = df.iloc[:, -1]

        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.25, random_state = 0)

        ann = tf.keras.models.Sequential()
        ann.add(tf.keras.layers.Dense(units=47, activation='sigmoid'))
        ann.add(tf.keras.layers.Dense(units=8, activation='sigmoid'))
        ann.add(tf.keras.layers.Dense(units=8, activation='sigmoid'))
        ann.add(tf.keras.layers.Dense(units=8, activation='sigmoid'))
        ann.add(tf.keras.layers.Dense(units=8, activation='sigmoid'))
        ann.add(tf.keras.layers.Dense(units=8, activation='sigmoid'))
        ann.add(tf.keras.layers.Dense(units=8, activation='sigmoid'))
        ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))
        ann.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics = ['accuracy'])
        ann.fit(X_train, y_train, batch_size = 5, epochs = 10)

        y_pred = ann.predict(X_test)
        pred = ann.predict(reshapedArray)

# ================================================================

        # df2['Yield_'] = df2['Yield']
        # del df2['Yield']
        # del df2['Yeild_Cat']

        # X2 = df.iloc[:, 2:-1]
        # Y2 = df.iloc[:, -1]

        # X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, Y2, test_size = 0.25, random_state = 0)

        model = RandomForestRegressor(n_estimators = 11)
        model.fit(X_train,y_train)
        rf_predict = model.predict(reshapedArray)
        print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> ',rf_predict)
        return pred

    except Exception as e:
        print(e)
