import concurrent.futures
from image_quality_assessment import Image
import csv
from os import listdir
from os.path import isfile, join
from numpy import mean
from numpy import std
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_dataset():
    # carga el dataset
    dataset = pd.read_csv("features_copy.csv")
    y = dataset["assessment"]
    X = dataset.drop(["name", "assessment"], axis=1)
    # sc = StandardScaler()
    # X = sc.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=41
    )
    # return X_train, X_test, y_train, y_test
    return X, y


def create_model(X=None, y=None):
    model = LDA()
    model.fit(X.values, y)
    cv = RepeatedStratifiedKFold(n_splits=2, n_repeats=2, random_state=1)
    scores = cross_val_score(model, X, y, scoring="accuracy", cv=cv, n_jobs=-1)
    mean_score = round(mean(scores), 4)
    standar_deviation = round(std(scores), 4)
    return model, [mean_score, standar_deviation]
    # print(f"Mean Accuracy:{mean(scores):.4f}({std(scores):.4f})")


def get_files(my_path=None):
    images_files = [f for f in listdir(my_path) if isfile(join(my_path, f))]
    return images_files


def write_csv(data=None):
    with open("features.csv", "a", encoding="UTF-8") as file:
        writer = csv.writer(file)
        writer.writerow(data)


def get_features(file_name=None):
    img = f"test/{file_name}"
    test = Image(img)
    features = [
        test.get_sharpness,
        test.get_luminance,
        test.get_average_information_entropy,
        test.get_colorfulness,
    ]

    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = [executor.submit(feature) for feature in features]

    return [r.result() for r in results]


def main():
    # X_train, X_test, y_train, y_test = load_dataset()
    X, y = load_dataset()
    model, scores = create_model(X, y)
    print(f"Model Score:{scores[0]} ({scores[1]})")

    path_list = get_files("test")
    for i in path_list:
        # print(get_features(i))
        a = get_features(i)
        b = model.predict_proba([a])
        c = model.predict([a])
        print(f"{a} -> {c}")
        break


if __name__ == "__main__":
    main()
