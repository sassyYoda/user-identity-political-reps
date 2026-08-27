import pytest
from huggingface_hub import HfApi

from polreps.config import (
    MODEL_NAME,
    MODEL_REVISION,
    REPLICATION_MODEL_NAME,
    REPLICATION_MODEL_REVISION,
    hf_token,
)


@pytest.mark.slow
@pytest.mark.parametrize(
    "model,revision",
    [
        (MODEL_NAME, MODEL_REVISION),
        (REPLICATION_MODEL_NAME, REPLICATION_MODEL_REVISION),
    ],
)
def test_token_sees_gated_gemma_at_pinned_revision(model, revision):
    token = hf_token()
    assert token, "HF_TOKEN missing from .env"

    # model_info raises GatedRepoError without gate access, so a successful
    # call at the pinned revision is itself the access proof
    info = HfApi(token=token).model_info(model, revision=revision)

    assert info.sha == revision
