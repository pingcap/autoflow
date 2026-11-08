"""Download only reviewed model artifacts at image-build time."""

from huggingface_hub import snapshot_download
from model_manifest import EMBEDDING_MODELS, RERANKER_MODELS

for model, revision in (EMBEDDING_MODELS | RERANKER_MODELS).items():
    artifacts = ["*.json", "*.txt", "*.model", "tokenizer.*"]
    # The reviewed bge-m3 revision publishes a PyTorch state dict, not safetensors.
    # Runtime uses pinned patched PyTorch/Transformers with weights-only loading.
    artifacts += (
        ["pytorch_model.bin"]
        if model in EMBEDDING_MODELS
        else ["model.safetensors", "model-*.safetensors"]
    )
    snapshot_download(model, revision=revision, allow_patterns=artifacts, max_workers=4)
