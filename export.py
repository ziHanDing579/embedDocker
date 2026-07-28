from sentence_transformers import SentenceTransformer
from sentence_transformers.backend import export_dynamic_quantized_onnx_model

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", backend="onnx")
model.save_pretrained("model/")
export_dynamic_quantized_onnx_model(model, "arm64", "model/")