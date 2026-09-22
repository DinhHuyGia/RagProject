"""Temporary compatibility helpers for importing Ragas 0.4.3."""

import sys
from importlib import import_module
from types import ModuleType


def install_vertexai_import_shim() -> None:
    """Provide the removed legacy Vertex AI module expected by Ragas.

    Ragas 0.4.3 imports ChatVertexAI unconditionally even when the
    evaluator uses OpenAI. langchain-community 0.4 removed that module.
    The placeholder is safe here because this project does not use Vertex AI.
    """
    module_name = "langchain_community.chat_models.vertexai"

    try:
        import_module(module_name)
        return
    except ModuleNotFoundError as error:
        if error.name != module_name:
            raise

    class ChatVertexAI:
        pass

    module = ModuleType(module_name)
    module.ChatVertexAI = ChatVertexAI
    sys.modules[module_name] = module
