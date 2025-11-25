import onnx
model = onnx.load('models/PPOCR_v5/det.onnx')
ops = set()
for node in model.graph.node:
  ops.add(node.op_type)
print(ops)