import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


dataset = pd.read_csv('mushrooms.csv')
dataset.head(10)

dataset.info()

dataset.shape


dataset.describe()


X = dataset[['cap-surface', "cap-shape", "cap-color", "odor"]]
y = dataset["class"]


X.isna().sum()


X.shape


from sklearn.preprocessing import LabelEncoder
LE = LabelEncoder()
X['cap-surface'] = LE.fit_transform(X['cap-surface'])
X['cap-shape'] = LE.fit_transform(X['cap-shape'])
X['cap-color'] = LE.fit_transform(X['cap-color'])
X['odor'] = LE.fit_transform(X['odor'])
y = LE.fit_transform(y)


X.tail()


from sklearn.preprocessing import StandardScaler
st = StandardScaler()
X_std = st.fit_transform(X)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_std, y, test_size=0.33, random_state=42)


from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()

lr.fit(X_train,y_train)

y_pred =lr.predict(X_test)

y_pred


from sklearn.metrics import accuracy_score


score = accuracy_score(y_pred, y_test)

score

from sklearn.metrics import confusion_matrix,classification_report


y_pred = np.resize(y_pred,(8124,))

y= np.resize(y,(8124,))

y.shape

report = classification_report(y,y_pred)

conf_matrix = confusion_matrix(y,y_pred)
TN,FP,FN,TP = confusion_matrix(y,y_pred).ravel()
print(TN,FP,FN,TP)


import seaborn as sb
sb.heatmap(conf_matrix,annot=True)
