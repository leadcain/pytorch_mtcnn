from detlib.detector.vision import easy_vis
try:
    from detlib.detector.detect import DetectFace
except:
    from detlib.detector.detect_onnx import DetectFace
