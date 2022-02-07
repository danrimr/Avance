"""Docs"""
from camera_widget import CameraWidget
from feature_extraction import Image
from classification import Classifier


def main():
    """Docs"""
    model = Classifier.from_csv("features_3.csv")
    model.make_classifier()
    print(model.get_model_score())
    # rtsp_server = "http://192.168.1.5:8080/shot.jpg"
    camera = CameraWidget(1)
    camera.start_stream()

    while True:
        image = Image(camera.dst)
        features = image.get_features()
        camera.save_frame()
        print(features)
        print(model.get_prediction([features]))


if __name__ == "__main__":
    main()
