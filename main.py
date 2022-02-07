"""Docs"""
from camera_widget import CameraWidget
from feature_extraction import Image
from prediction_model import GetModel


def main():
    """Docs"""
    model = GetModel.from_csv("features_2.csv")
    print(model.get_scores())
    # rtsp_server = "http://192.168.1.5:8080/shot.jpg"
    camera = CameraWidget(1)
    camera.start_stream()

    while True:
        image = Image(camera.dst)
        features = image.get_features()
        camera.save_frame()
        # camera.set_text(model.get_prediction([features])[0])
        # print(features)
        print(model.get_prediction([features]))


if __name__ == "__main__":
    main()
