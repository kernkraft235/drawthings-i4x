"""Use the official model importer for architecture detection and tensor remapping."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

from .converter import convert


def importer_path(executable=None):
    resolved = shutil.which(str(executable) if executable else 'draw-things-cli')
    if not resolved:
        raise ValueError('Safetensors import requires draw-things-cli on PATH or --draw-things-cli PATH. Native .ckpt conversion needs no importer.')
    return resolved


def run_import(source, stage, executable, *, dry_run=False, quiet=False):
    source = Path(source).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    models = stage / 'models'
    models.mkdir()
    command = [executable, 'models', 'import', str(source), '--models-dir', str(models),
               '--offline', '--no-download-missing']
    if dry_run:
        command.append('--dry-run')
    stream = sys.stdout if dry_run else sys.stderr
    log_path = stage / 'import.log'
    with log_path.open('w') as log:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, bufsize=1)
        try:
            for line in process.stdout:
                log.write(line)
                log.flush()
                if not quiet:
                    print(line, end='', file=stream, flush=True)
            status = process.wait()
        except BaseException:
            process.terminate()
            process.wait()
            raise
        finally:
            process.stdout.close()
    if status:
        raise ValueError(f'Official Draw Things importer exited with status {status}. Details: {log_path}')
    return models


def inspect_safetensors(source, executable=None):
    executable = importer_path(executable)
    with tempfile.TemporaryDirectory(prefix='dt-quant-inspect-') as temporary:
        run_import(source, Path(temporary), executable, dry_run=True)


def convert_safetensors(source, destination, executable=None, **options):
    executable = importer_path(executable)
    source, destination = Path(source).resolve(), Path(destination).resolve()
    specification_path = destination.with_suffix('.model.json')
    for path in (destination, destination.with_suffix('.validation.json'), specification_path):
        if path.exists():
            raise FileExistsError(path)
    stage = Path(str(destination) + '.import.work')
    stage.mkdir(parents=True, exist_ok=False)
    models = run_import(source, stage, executable, quiet=options.get('quiet', False))
    custom_path = models / 'custom.json'
    if not custom_path.is_file():
        raise ValueError(f'Importer produced no custom.json specification; see {stage / "import.log"}')
    specifications = json.loads(custom_path.read_text())
    if not isinstance(specifications, list) or len(specifications) != 1:
        raise ValueError(f'Expected one imported model specification in {custom_path}')
    specification = specifications[0]
    if not isinstance(specification, dict):
        raise ValueError(f'Invalid model specification in {custom_path}')
    name = specification.get('file')
    if not isinstance(name, str) or Path(name).name != name:
        raise ValueError('Importer returned an invalid primary checkpoint name')
    imported = models / name
    if not imported.is_file():
        raise ValueError(f'Importer produced no primary checkpoint: {imported}')
    companions = sorted(path for path in models.iterdir()
                        if path.is_file() and path.name not in (name, name + '-tensordata')
                        and (path.suffix == '.ckpt' or path.name.endswith('.ckpt-tensordata')))
    for path in companions:
        if (destination.parent / path.name).exists():
            raise FileExistsError(destination.parent / path.name)
    result = convert(imported, destination, **options)
    result.update(input=str(source), importer=executable, imported_model_version=specification.get('version'))
    for path in companions:
        os.link(path, destination.parent / path.name)
    specification['file'] = destination.name
    with specification_path.open('x') as file:
        json.dump(specification, file, indent=2)
    result['model_specification'] = specification_path.name
    result['imported_companions'] = [path.name for path in companions]
    destination.with_suffix('.validation.json').write_text(json.dumps(result, indent=2))
    shutil.rmtree(stage)
    # The reference converter retains its work directory; this import's intermediate is disposable.
    shutil.rmtree(Path(str(destination) + '.work'))
    return result
