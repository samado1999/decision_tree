import matplotlib.pyplot as plt
import pandas as pd
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.tree import export_text

soccer_bet = pd.read_csv(r'results.csv', delimiter=',')

soccer_bet.drop(soccer_bet[soccer_bet['tournament'] != 'Copa América'].index, inplace=True)

soccer_bet.drop(["date"], axis=1, inplace=True)
soccer_bet.drop(["away_team"], axis=1, inplace=True)
soccer_bet.drop(["tournament"], axis=1, inplace=True)
soccer_bet.drop(["city"], axis=1, inplace=True)
soccer_bet.drop(["country"], axis=1, inplace=True)

soccer_bet.loc[(soccer_bet.home_score > soccer_bet.away_score), ['home_win']] = 1
soccer_bet.loc[(soccer_bet.away_score > soccer_bet.home_score), ['away_win']] = 1
soccer_bet.loc[(soccer_bet.home_score == soccer_bet.away_score), ['draw']] = 1

soccer_bet['home_win'] = soccer_bet['home_win'].fillna(0)
soccer_bet['away_win'] = soccer_bet['away_win'].fillna(0)
soccer_bet['draw'] = soccer_bet['draw'].fillna(0)

soccer_bet['neutral'].replace({False: 0, True: 1}, inplace=True)

# soccer_bet.drop(["home_score"], axis=1, inplace=True)
# soccer_bet.drop(["away_score"], axis=1, inplace=True)
soccer_bet.drop(["neutral"], axis=1, inplace=True)

soccer_bet.to_csv(r'soccer_bet_clean_dataframe.csv', index=False, header=True, sep=',')

print(soccer_bet.head())
X = soccer_bet.drop(columns=['home_team'])
print(X)
Y = soccer_bet['home_team']

feature_names = X.columns
labels = Y.unique()

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.1)

model = tree.DecisionTreeClassifier(criterion="entropy")

model.fit(x_train, y_train)
model.predict(x_test)

# tree.plot_tree(model)
# facecolor='k'
plt.figure(figsize=(30, 10))

tree.plot_tree(model,
               feature_names=feature_names,
               class_names=labels,
               rounded=True,
               filled=True,
               fontsize=3)

image_format = 'pdf'
image_name = 'model.pdf'

plt.savefig(image_name, format=image_format, bbox_inches='tight', dpi=1200)
