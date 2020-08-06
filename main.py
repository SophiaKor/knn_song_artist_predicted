import pandas as pd
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import tree

if __name__ == "__main__":
    data = pd.read_csv(r'data/songs.csv')
    X = data.drop(['song', 'artist', 'lyrics'], axis=1)
    X = pd.get_dummies(X)
    y = data.artist

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    clf = KNeighborsClassifier()
    parametrs = {'n_neighbors': range(1, 7), 'weights': ['uniform', 'distance']}
    grid_search_cv_clf = GridSearchCV(clf, parametrs, cv=5)
    grid_search_cv_clf.fit(X_train, y_train)
    best_clf = grid_search_cv_clf.best_estimator_
    best_clf.fit(X_train, y_train)
    predicted = best_clf.predict(X_test)
    print(metrics.classification_report(y_test, predicted))
    print(metrics.confusion_matrix(y_test, predicted))

    X = data.drop(['song', 'artist', 'lyrics', 'genre'], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    clf2 = tree.DecisionTreeClassifier()
    clf2.fit(X_train, y_train)
    predicted2 = clf2.predict(X_test)
    print(metrics.classification_report(y_test, predicted2))
    print(metrics.confusion_matrix(y_test, predicted2))
