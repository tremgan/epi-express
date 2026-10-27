import numpy as np
import pandas as pd
import torch

from model_infrastruture.data_prep import ALL_GENES, N_BINS, N_CHANNELS, make_loader
from model_infrastruture.train import MODEL_NAME
from models.model import make_model


@torch.no_grad()
def predict(model, loader):
    model.eval()
    return torch.cat([model(x).squeeze(-1) for (x,) in loader]).numpy()


if __name__ == "__main__":
    model = make_model(MODEL_NAME, N_BINS, N_CHANNELS)
    model.load_state_dict(torch.load(f"model_{MODEL_NAME}.pt"))

    # X3 is an unseen cell line: its tracks are z-scored with its own statistics
    test_loader = make_loader(split="test", cells=("X3",), shuffle=False)
    pred = predict(model, test_loader)

    # same genes, same order as make_loader used (= order of X3_test_info.tsv)
    test_genes = ALL_GENES[(ALL_GENES["cell"] == "X3") & (ALL_GENES["split"] == "test")]
    submission = pd.DataFrame(
        {
            "gene_name": test_genes["gene_name"].to_numpy(),
            "gex_predicted": pred.astype(float),
        }
    )

    assert len(submission) == len(test_genes), "One prediction per test gene required"
    assert submission["gene_name"].is_unique, "Duplicate gene names"
    assert np.isfinite(submission["gex_predicted"]).all(), "Predictions contain NaN or inf"

    submission.to_csv("submission.csv", index=False)  # index=False is essential
    print(f"Saved submission.csv with {len(submission)} rows")
