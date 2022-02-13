"""Docs"""

from camera_widget import CameraWidget
from feature_extraction import Image
from classification import Classifier
from light_control import LightControl


def main():
    """Docs"""
    ligh_control = LightControl(port="COM3")
    model = Classifier.from_csv("features_3.csv")
    model.make_classifier()
    print(model.get_model_score())
    camera = CameraWidget(1)
    camera.start_stream()

    while True:
        image = Image(camera.dst)
        features = image.get_features()
        camera.save_frame()
        predicted = model.get_prediction([features])
        print(predicted)
        pred_float = float(predicted[0])
        ligh_control.send_data(pred_float)


if __name__ == "__main__":
    main()
