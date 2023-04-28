import argparse
import os

from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
import joblib
from sklearn import metrics
import numpy as np
import csv

DATA_PATH = '/Users/lizhiyuan/学习/深度学习与NLP/W7大作业/buaa_nlp_project3-main/segments2/'

label_dic = {'鹿鼎记': 0, '射雕英雄传': 1, '神雕侠侣': 2, '天龙八部': 3, '笑傲江湖': 4, '倚天屠龙记': 5, '连城诀': 6, '碧血剑': 7, '白马啸西风': 8, '飞狐外传': 9, '三十三剑客图': 10, '书剑恩仇录': 11, '侠客行': 12, '雪山飞狐': 13, '鸳鸯刀': 14, '越女剑': 15, }


def get_data():
    with open('theta20.csv') as f:
        segs = csv.reader(f)
        features = [seg for seg in segs]
    file_paths = os.listdir(DATA_PATH)
    labels = []
    for file in file_paths:
        name = ''
        for s in file:
            if '0' <= s <= '9':
                continue
            if s == '.':
                break
            name += s
        labels.append(label_dic[name])
    return features, labels


def train_svm():
    model = OneVsRestClassifier(svm.SVC(kernel='linear', probability=True))
    print('start training!')
    classifier = model.fit(x_train, y_train)
    print('finish training')
    joblib.dump(classifier, "classifier.pkl")
    y_predict = classifier.predict(x_train)
    accuracy = metrics.accuracy_score(y_predict, y_train)
    print('overall accuracy:{:.6f}'.format(accuracy))
    print('----------------------')
    single_accuracy = metrics.precision_score(y_train, y_predict, average=None)
    print('accuracy for each class:', single_accuracy)
    print('----------------------')
    avg_acc = np.mean(single_accuracy)
    print('average accuracy:{:.6f}'.format(avg_acc))


def test_svm():
    classifier = joblib.load('/Users/lizhiyuan/学习/深度学习与NLP/W7大作业/buaa_nlp_project3-main/code/classifier.pkl')
    y_predict = classifier.predict(x_test)
    accuracy = metrics.accuracy_score(y_predict, y_test)
    print('overall accuracy:{:.6f}'.format(accuracy))
    print('----------------------')
    single_accuracy = metrics.precision_score(y_test, y_predict, average=None)
    print('accuracy for each class:', single_accuracy)
    print('----------------------')
    avg_acc = np.mean(single_accuracy)
    print('average accuracy:{:.6f}'.format(avg_acc))


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='test', help='train test')
    opt = parser.parse_args()
    return opt


if __name__ == '__main__':
    params = parse_opt()
    x, y = get_data()
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.2, random_state=3)
    if params.mode == 'train':
        train_svm()
    else:
        test_svm()

