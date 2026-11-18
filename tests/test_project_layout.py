"""Check notebook generation and execution paths without training real models."""
import ast
import importlib.util
from pathlib import Path

import nbformat
import pytest

ROOT = Path(__file__).resolve().parents[1]


def load_builder(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'scripts' / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('kind', ['retailhero', 'criteo'])
def test_baseline_builder_uses_experiment_paths_and_finds_root(tmp_path, monkeypatch, kind):
    module = load_builder('build_notebooks')
    monkeypatch.setattr(module, 'ROOT', tmp_path)
    rel = ('experiments/01_retailhero/study.ipynb' if kind == 'retailhero'
           else 'experiments/02_criteo/archive/baseline.ipynb')
    path = tmp_path / rel
    path.parent.mkdir(parents=True)
    current = tmp_path / 'experiments/02_criteo/study.ipynb'
    current.parent.mkdir(parents=True, exist_ok=True)
    current.write_text('current-study-must-not-be-overwritten')
    module.build(kind)
    assert current.read_text() == 'current-study-must-not-be-overwritten'
    verify_generated_notebook(path, tmp_path, monkeypatch)


def test_current_criteo_builder_uses_experiment_paths_and_finds_root(tmp_path, monkeypatch):
    module = load_builder('build_criteo_study')
    monkeypatch.setattr(module, 'ROOT', tmp_path)
    path = tmp_path / 'experiments/02_criteo/study.ipynb'
    path.parent.mkdir(parents=True)
    module.build()
    verify_generated_notebook(path, tmp_path, monkeypatch)
    source = '\n'.join(cell.source for cell in nbformat.read(path, as_version=4).cells)
    assert 'shared.criteo.models' in source
    assert 'experiments/02_criteo/results' in source


def verify_generated_notebook(path, root, monkeypatch):
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    code = [cell.source for cell in notebook.cells if cell.cell_type == 'code']
    for source in code:
        ast.parse(source)
    marker = root / 'shared/causal_utils.py'
    marker.parent.mkdir(exist_ok=True)
    marker.touch()
    monkeypatch.chdir(path.parent)
    root_assignments = [node for source in code for node in ast.parse(source).body
                        if isinstance(node, ast.Assign)
                        and any(isinstance(target, ast.Name) and target.id == 'ROOT'
                                for target in node.targets)]
    assert len(root_assignments) == 1
    namespace = {'Path': Path}
    exec(compile(ast.Module(body=root_assignments, type_ignores=[]), '<root-discovery>', 'exec'), namespace)
    assert namespace['ROOT'] == root


def test_current_criteo_provenance_paths_exist(tmp_path, monkeypatch):
    module = load_builder('build_criteo_study')
    monkeypatch.setattr(module, 'ROOT', tmp_path)
    path = tmp_path / 'experiments/02_criteo/study.ipynb'
    path.parent.mkdir(parents=True)
    module.build()
    notebook = nbformat.read(path, as_version=4)
    assignments = [node for cell in notebook.cells if cell.cell_type == 'code'
                   for node in ast.parse(cell.source).body
                   if isinstance(node, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id == 'source_hashes' for t in node.targets)]
    assert len(assignments) == 1
    paths = ast.literal_eval(assignments[0].value.generators[0].iter)
    for rel in paths:
        assert (ROOT / rel).is_file(), rel
