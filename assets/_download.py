from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="TianxingChen/RoboTwin2.0",
    repo_type="dataset",
    local_dir=".",
    allow_patterns=["background_texture.zip", "embodiments.zip", "objects.zip"],
    resume_download=True,
)
