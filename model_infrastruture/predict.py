"""Predict X3 gene expression from a trained checkpoint and write the Kaggle submission.

Usage (from the project root):
    python -m model_infrastruture.predict
    python -m model_infrastruture.predict --checkpoint checkpoints/model.pt --output submission.csv
"""
import argparse

import numpy as np
import pandas as pd
import torch

from model_infrastruture.data_prep import ALL_GENES, make_loader
from model_infrastruture.train import CHECKPOINT_PATH
from models.model import MODELS


def load_model(path: str):
    # weights_only=True is fine: the checkpoint holds only tensors, dicts and lists
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    # checkpoints from before model_name was saved were all ConvMaxPoolFeedForward
    model = MODELS[ckpt.get("model_name", "ConvMaxPoolFeedForward")](**ckpt["model_config"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    # checkpoints from before normalization existed were trained on un-normalized inputs
    return model, ckpt["tracks"], ckpt.get("normalize", False)


@torch.no_grad()
def predict(model: torch.nn.Module, loader) -> np.ndarray:
    # Predictions are in log1p(gex) space; the competition is scored with Spearman,
    # so only their order matters and no back-transform is needed.
    return torch.cat([model(x).squeeze(-1) for (x,) in loader]).numpy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=CHECKPOINT_PATH)
    parser.add_argument("--output", default="submission.csv")
    args = parser.parse_args()

    model, tracks, normalize = load_model(args.checkpoint)

    # make_loader keeps the row order of ALL_GENES, so gene names line up with predictions
    test_genes = ALL_GENES[(ALL_GENES["cell"] == "X3") & (ALL_GENES["split"] == "test")]
    loader = make_loader(split="test", tracks=tracks, cells=("X3",), shuffle=False, normalize=normalize)
    pred = predict(model, loader)

    submission = pd.DataFrame(
        {"gene_name": test_genes["gene_name"].to_numpy(), "gex_predicted": pred.astype(float)}
    )
    assert len(submission) == len(pred), "One prediction per test gene required"
    assert submission["gene_name"].is_unique, "Duplicate gene names"
    assert np.isfinite(submission["gex_predicted"]).all(), "Predictions contain NaN or inf"

    submission.to_csv(args.output, index=False)  # index=False is essential
    print(f"Saved {len(submission)} predictions to {args.output}")


if __name__ == "__main__":
    main()