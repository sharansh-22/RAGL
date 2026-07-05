from sentence_transformers import CrossEncoder
model = CrossEncoder("cross-encoder/nli-deberta-v3-small", device="cpu")
print("id2label:", model.config.id2label)
