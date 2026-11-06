import argparse
import contextlib
import io
import os
import time
import traceback
from pathlib import Path

import nbformat
from IPython.display import display


def run(path):
    path = path.resolve()
    os.chdir(path.parent)
    namespace = {'display': display, '__name__': '__main__'}
    notebook = nbformat.read(path, as_version=4)
    execution_count = 0
    for index, cell in enumerate(notebook.cells):
        if cell.cell_type != 'code':
            continue
        execution_count += 1
        cell.execution_count = execution_count
        stream = io.StringIO()
        started = time.time()
        print(f'{path.name} cell {index} started', flush=True)
        try:
            with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
                exec(compile(cell.source, f'{path.name}:cell{index}', 'exec'), namespace)
        except Exception:
            cell.outputs = [nbformat.v4.new_output('stream', name='stdout', text=stream.getvalue())]
            cell.outputs.append(nbformat.v4.new_output(
                'error', ename='ExecutionError', evalue='Cell execution failed',
                traceback=traceback.format_exc().splitlines(),
            ))
            nbformat.write(notebook, path)
            raise
        cell.outputs = [nbformat.v4.new_output('stream', name='stdout', text=stream.getvalue())]
        nbformat.write(notebook, path)
        print(stream.getvalue()[-7000:], flush=True)
        print(f'cell {index} finished in {time.time() - started:.1f}s', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('notebook', type=Path)
    run(parser.parse_args().notebook)
