"""The downstream dataset feeder: published paths and identities, without tensors.

Creation models and initialization regimes describe the data, not the student.
Some exports were selected using validation examples; test examples were not used.
RAMBO generates operating points. BAS, US and DSS select real training examples.
"""
import argparse
import json
from functools import lru_cache
from pathlib import Path

import yaml

# in `experiments`
CATALOG = Path(__file__).resolve().parents[1] / "experiments/configs/datasets.yaml"

# methods
# - distillation/condensation: DD, DCGM (w/ grad matching), HMDC
# - operating-point generation: RAMBO; selection: BAS
# - gen. selection: US, DSS
ACRONYMS = dict(DD='DD',
                DCGM='DCGM',
                HMDC='HMDC',
                BAS='BAS', 
                RAMBO='RAMBO',
                uniform='US',
                demand_stratified='DSS',
                input_target_herding='ITH', physics_herding='PIH', input_kcenter='KC',
                demand_quantiles='DQS', learned_demand='LDM', analytical_labels='ALC')


@lru_cache(maxsize=4)
def _catalog(path, signature):
    return yaml.safe_load(Path(path).read_text())['datasets']

def find_datasets(*,
                  method=None,
                  model=None,
                  size=None,
                  case=None,
                  target_mode=None,
                  initialization=None,
                  generation=None,
                  initialization_name=None,
                  dataset_id=None,
                  catalog=CATALOG):
    """Return matching ready records, including IDs, regimes and artifact_path.

    Method accepts an acronym or YAML method name. Model is the exact name/pair
    (e.g. 'GIN+HEAT'), 'none' for model-independent data, or None for all models.
    BAS uses its selector model ('MLP'). Excluded/partial datasets are never read.
    """
    if method is not None:
        aliases = {name.upper(): acronym for name, acronym in ACRONYMS.items()}
        method = aliases.get(method.upper(), method.upper())
        if method not in ACRONYMS.values():
            raise ValueError(f'Unknown method {method!r}; choose {", ".join(ACRONYMS.values())}')
    path = Path(catalog).expanduser().resolve()
    stat = path.stat()
    records, seen = [], set()
    for row in _catalog(str(path), (stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)):
        if row['dataset_id'] in seen:
            raise ValueError(f"Duplicate catalog dataset_id: {row['dataset_id']}")
        seen.add(row['dataset_id'])
        acronym = ACRONYMS[row['method']]
        reference = '+'.join(row['models']) or 'none'
        filters = ((method, acronym), (model, reference), (size, row['count']),
                   (case, row['case']), (target_mode, row['target_mode']),
                   (initialization, row['initialization']),
                   (generation, row['generation']),
                   (initialization_name, row['initialization_name']),
                   (dataset_id, row['dataset_id']))
        if all(wanted is None or wanted == value for wanted, value in filters):
            records.append(dict(row, acronym=acronym, reference_model=reference))
    return records


def dataset_record(dataset_id, *, catalog=CATALOG):
    """Resolve exactly one published dataset. Partial/excluded entries cannot resolve."""
    records = find_datasets(dataset_id=dataset_id, catalog=catalog)
    if not records:
        raise ValueError(f'No ready dataset {dataset_id!r} in {catalog}')
    return records[0]


def dataset_paths(**filters):
    """Return {acronym: {reference_model: {size: [pt_paths]}}}; retain all matches.

    Filters are those of find_datasets. Use that function for case/run metadata
    and artifact_path, which the shared datasets.load_dataset reader accepts.
    """
    groups = {}
    for row in find_datasets(**filters):
        models = groups.setdefault(row['acronym'], {})
        sizes = models.setdefault(row['reference_model'], {})
        sizes.setdefault(row['count'], []).append(row['data_path'])
    return groups

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--catalog', type=Path, default=CATALOG)
    parser.add_argument('--method', choices=tuple(ACRONYMS.values()), help='Distillation/condensation or selection technique')
    parser.add_argument('--model', help="Creator model/pair, MLP, or none; not the downstream student")
    parser.add_argument('--size', type=int, help='Published sample count; derived M400 mixtures have their own manifests')
    parser.add_argument('--case', choices=(14, 30, 57, 118, 500), type=int, help='Case size; supported 14, 30,..., 500')
    parser.add_argument('--target-mode', choices=('learned', 'solver', 'frozen', 'original_opfdata'))
    parser.add_argument('--initialization', choices=('random', 'fixed', 'pretrained'), help='`random` or `fixed` seed (from scratch) or `pretrained`.')
    parser.add_argument('--records', action='store_true', help='Include full dataset metadata')
    parser.add_argument('--generation')
    parser.add_argument('--initialization-name')
    parser.add_argument('--dataset-id')
    
    # parse args
    args = vars(parser.parse_args())
    
    # idetnfiy dset
    lookup = find_datasets if args.pop('records') else dataset_paths
    
    # check if meta JSON exist
    try:
        print(json.dumps(lookup(**args), indent=2))
    except ValueError as error:
        parser.error(str(error))

# entry point
if __name__ == '__main__':
    main()
