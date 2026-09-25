import os
import runpy
from pathlib import Path

import pytest

from grounded import shared
from grounded.shared import get_base_url


@pytest.mark.parametrize("existing", [None, "https://process.example.com"])
def test_dotenv_loads_from_project_root_and_preserves_environment(
    monkeypatch: pytest.MonkeyPatch,
    workspace_temp_path: Path,
    existing: str | None,
) -> None:
    package = workspace_temp_path / "grounded"
    package.mkdir()
    module = package / "shared.py"
    module.write_text(
        Path(shared.__file__).read_text(encoding="utf-8"), encoding="utf-8"
    )
    (workspace_temp_path / ".env").write_text(
        "AZURE_OPENAI_ENDPOINT=https://dotenv.example.com\n", encoding="utf-8"
    )
    monkeypatch.delenv("AZURE_OPENAI_ENDPOINT", raising=False)
    if existing is not None:
        monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", existing)
    with monkeypatch.context() as context:
        context.chdir(package)
        runpy.run_path(str(module))

    assert os.environ["AZURE_OPENAI_ENDPOINT"] == (
        existing or "https://dotenv.example.com"
    )


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
