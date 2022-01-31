from feature_extraction import Image
import threading
import cv2, time
import urllib.request
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import train_test_split
import pandas as pd
from numpy import mean
from numpy import std
import numpy as np


def load_dataset():
    # carga el dataset
    dataset = pd.read_csv("features_copy.csv")
    y = dataset["assessment"]
    X = dataset.drop(["name", "assessment"], axis=1)
    # X_train, X_test, y_train, y_test = train_test_split(
    #     X, y, test_size=0.1, random_state=41
    # )
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


CAP_STATUS = True


def video_ip_cam(url="http://192.168.1.5:6969/shot.jpg"):
    while True:

        global frame
        frameResp = urllib.request.urlopen(url)
        frameNp = np.array(bytearray(frameResp.read()), dtype=np.uint8)
        frame = cv2.imdecode(frameNp, -1)
        cv2.imshow("IWebcam", frame)
        # cv2.imwrite("test/test.JPG", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


def video_cam():
    global frame
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    while True:
        _, frame = cap.read()
        cv2.imshow("Name", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()


def image_processing():
    time.sleep(1)
    X, y = load_dataset()
    model, scores = create_model(X, y)
    while True:
        cv2.imwrite("test/test.JPG", frame)
        image_cam = get_features("test.JPG")
        a = model.predict([image_cam])
        print(a)


def get_features(file_name=None):
    img = f"test/{file_name}"
    test = Image(img)
    features = [
        test.get_sharpness(),
        test.get_luminance(),
        test.get_average_information_entropy(),
        test.get_colorfulness(),
    ]
    return features


def main():

    t1 = threading.Thread(target=video_ip_cam)
    t2 = threading.Thread(target=image_processing)

    t1.daemon = True
    t2.daemon = True

    t1.start()
    t2.start()

    t1.join()
    t2.join()


if __name__ == "__main__":
    main()
