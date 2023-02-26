import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# CSV読み込み
df = pd.read_csv('./Iris.csv')

# 特徴量だけ抽出 X['PetalLengthCm']みたいな形で取り出せる
X = df[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]

# 正解データだけ抽出
Y = df['Species']

# 8:2で学習と評価に分ける 
# x_train:学習用データ(8), x_test:評価用データ(2), y_train:学習用正解データ(8), y_test: 評価用正解データ(2)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

# 学ばせるモデルを作成
clf = RandomForestClassifier()

# 学べ！！
clf.fit(X_train, Y_train)

# 評価しろ！！
pred = clf.predict(X_test)

# 結果はどうなんだ！？？
result = accuracy_score(Y_test, pred)
print(result)
