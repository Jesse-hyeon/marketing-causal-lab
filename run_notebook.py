"""Execute the prototype from its project directory using this Python environment."""

from pathlib import Path
import os
import tempfile

import nbformat
from nbclient import NotebookClient
from jupyter_client.kernelspec import KernelSpecManager


def main():
    root = Path(__file__).resolve().parent
    target = root / "02. prototype.ipynb"
    notebook = nbformat.read(target, as_version=4)
    kernel_manager = KernelSpecManager()
    kernel_name = "causal-study"
    kernel_spec = kernel_manager.get_kernel_spec(kernel_name)
    expected_python = root / ".venv" / "bin" / "python"
    if Path(kernel_spec.argv[0]).absolute() != expected_python:
        raise RuntimeError(
            f"{kernel_name} must use {expected_python}, got {kernel_spec.argv[0]}"
        )
    client = NotebookClient(
        notebook,
        timeout=600,
        kernel_name=kernel_name,
        resources={"metadata": {"path": str(root)}},
        allow_errors=False,
    )
    # Use the project-specific registered kernel, never the generic python3 alias.
    print("Kernel:", kernel_spec.resource_dir)
    print("Kernel Python:", kernel_spec.argv[0])
    client.execute()
    nbformat.validate(notebook)
    descriptor, temporary = tempfile.mkstemp(prefix=".prototype-", suffix=".ipynb", dir=root)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            nbformat.write(notebook, stream)
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    print(f"Executed {sum(c.cell_type == 'code' for c in notebook.cells)} code cells: {target.name}")


if __name__ == "__main__":
    main()
