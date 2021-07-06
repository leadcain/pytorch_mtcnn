import onnx

onnx_pmodel = onnx.load("pnet.onnx")
onnx_rmodel = onnx.load("rnet.onnx")
onnx_omodel = onnx.load("onet.onnx")

print("pnet:", onnx.checker.check_model(onnx_pmodel))
print("rnet:", onnx.checker.check_model(onnx_rmodel))
print("onet:", onnx.checker.check_model(onnx_omodel))
