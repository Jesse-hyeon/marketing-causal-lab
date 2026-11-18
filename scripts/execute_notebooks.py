"""Execute notebooks sequentially and preserve the last executed state on failure."""
import os
from pathlib import Path
import sys
import nbformat
from nbclient import NotebookClient
ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
for name in (sys.argv[1:] or ['experiments/01_retailhero/study.ipynb','experiments/02_criteo/study.ipynb']):
    path=ROOT/name
    notebook=nbformat.read(path,as_version=4)
    def on_cell_start(cell,cell_index,**kwargs):
        if cell.cell_type=='code':
            print(f'{name} cell {cell_index}: {cell.source.splitlines()[0][:80]}',flush=True)
    try:
        NotebookClient(notebook,timeout=1200,kernel_name='causal-study',resources={'metadata':{'path':str(ROOT)}},on_cell_start=on_cell_start).execute()
    finally:
        nbformat.write(notebook,path)
    print('COMPLETE',name,flush=True)
