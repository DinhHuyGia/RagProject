import pytest

from grounded.shared import get_base_url


def test_get_base_url_requires_azure_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("AZURE_OPENAI_ENDPOINT", raising=False)

    with pytest.raises(
        RuntimeError,
        match="AZURE_OPENAI_ENDPOINT environment variable is required",
    ):
        get_base_url()


def test_get_base_url_normalizes_azure_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "AZURE_OPENAI_ENDPOINT",
        " https://example-resource.openai.azure.com/ ",
    )

    assert (
        get_base_url()
        == "https://example-resource.openai.azure.com/openai/v1/"
    )
