from ultralytics import YOLO

def study():
  dataset_path = "data.yaml"
  model = YOLO("yolo11n.pt")
  model.train(data=dataset_path, epochs=150, batch=16, imgsz=640)


def testCase():
  model = YOLO("runs/detect/train/weights/best.pt")
  result = model("DataSet/train/images/1_png.rf.4e51bf3538fac001a33a66aedc7b7da0.jpg", conf=0.1)
  result[0].show()