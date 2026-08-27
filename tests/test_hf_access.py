import pytest
from huggingface_hub import HfApi

from polreps.config import MODEL_NAME, MODEL_REVISION, hf_token


@pytest.mark.slow
def test_token_sees_gated_gemma_at_pinned_revision():
    token = hf_token()
    assert token, "HF_TOKEN missing from .env"

    # model_info raises GatedRepoError without gate access, so a successful
    # call at the pinned revision is itself the access proof
    info = HfApi(token=token).model_info(MODEL_NAME, revision=MODEL_REVISION)

    assert info.sha == MODEL_REVISION
