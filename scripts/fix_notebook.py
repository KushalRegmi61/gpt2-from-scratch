# fix_notebooks.py
import nbformat
import os
from pathlib import Path

ROOT = Path('./notebooks')  # change as needed
ACTION = 'add_state'  # 'add_state' or 'remove_widgets'

def fix_nb(path: Path):
    nb = nbformat.read(path.as_posix(), as_version=nbformat.NO_CONVERT)
    meta = nb.get('metadata', {})
    if 'widgets' in meta:
        widgets = meta['widgets']
        if ACTION == 'remove_widgets':
            print(f"Removing widgets metadata from {path}")
            del meta['widgets']
        else:
            # ensure application/vnd... key has a state
            for k, v in list(widgets.items()):
                if isinstance(v, dict) and 'state' not in v:
                    print(f"Adding empty state for {k} in {path}")
                    v['state'] = {}
                    widgets[k] = v
            meta['widgets'] = widgets
        nb['metadata'] = meta
        nbformat.write(nb, path.as_posix())
    else:
        print(f"No widgets metadata in {path}")

for p in ROOT.rglob('*.ipynb'):
    fix_nb(p)
