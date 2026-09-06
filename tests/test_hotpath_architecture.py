"""Architectural guardrails for the visibility inference dependency graph."""

from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HOT_PATH = tuple(
    path
    for package in ("forward", "likelihood", "infer")
    for path in (ROOT / "src/kinuv" / package).rglob("*.py")
)


def test_hot_path_has_no_mass_decomposition_or_cosmology():
    forbidden_text = (
        "dark_matter",
        "dark matter",
        "nfw",
        "burkert",
        "halo_mass",
        "baryonic_mass",
        "mass_decomposition",
        "astropy.cosmology",
    )
    for path in HOT_PATH:
        source = path.read_text(encoding="utf-8").lower()
        for token in forbidden_text:
            assert token not in source, f"{path.relative_to(ROOT)} contains {token!r}"


def test_hot_path_does_not_import_postprocessing():
    forbidden_identifiers = ("nfw", "burkert", "halo", "baryon", "dark_matter", "mass_decomposition")
    for path in HOT_PATH:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.append(node.module)
        assert not any(name.startswith("kinuv.postprocess") for name in imports), path
        identifiers = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                identifiers.append(node.name.lower())
            elif isinstance(node, ast.Name):
                identifiers.append(node.id.lower())
        for name in identifiers:
            assert not any(token in name for token in forbidden_identifiers), (
                path.relative_to(ROOT),
                name,
            )


def test_likelihood_is_visibility_chi2_only():
    source = (ROOT / "src/kinuv/likelihood/chi2.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    functions = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}
    assert {"chi2", "chi2_zero", "delta_chi2"} <= functions
    assert "potential" not in source.lower()
