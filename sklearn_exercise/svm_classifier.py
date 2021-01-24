import sklearn
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.svm import SVC
from sklearn.datasets import fetch_olivetti_faces
import matplotlib.pyplot as plt
from sklearn import metrics

faces = fetch_olivetti_faces()

#print(faces.DESCR)
# 10개의 class, 총 400장의 image sample, 각 image는 64x64의 픽셀

data = faces.data
images = faces.images
labels = faces.target

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.25, random_state=42)

svc_clf = SVC(kernel='linear')

def k_folds_validation(clf, X, y, K) :
    cv = KFold(n_splits=K, shuffle=True, random_state=42)
    scores = cross_val_score(clf, X, y, cv = cv)
    return scores


def train_and_evaluate(clf, X_train, X_test, y_train, y_test) :
    print('학습 시작')
    clf.fit(X_train, y_train)
    print('학습 완료')

    print('Score for Training set : {}'.format(clf.score(X_train, y_train)))
    print('Score for Test set : {}'.format(clf.score(X_test, y_test)))

    y_pred = clf.predict(X_test)
    print('Classification Report')
    print(metrics.classification_report(y_test, y_pred))
    
    print('Confusion metrix')
    print(metrics.confusion_matrix(y_test, y_pred))

train_and_evaluate(svc_clf, X_train, X_test, y_train, y_test)
    