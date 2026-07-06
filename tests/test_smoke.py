"""Smoke tests that don't require GPU or network."""
import importlib


def test_package_imports():
    assert importlib.import_module("trinity").__version__


def test_fireworks_client_module_imports():
    # Import must not require the API key (only instantiation does).
    importlib.import_module("trinity.llm.fireworks_client")
