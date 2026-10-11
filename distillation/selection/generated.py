"""Generated training data has its own IDs; it is never an OPFData subset."""
import json
from pathlib import Path

import torch
from torch.utils.data import Dataset

from distillation.training.T1_single_topo.common import file_hash


class GeneratedDataset(Dataset):
    """Load generated training graphs after validating their identities and files."""
    def __init__(self, manifest):
        self.path = Path(manifest)
        self.manifest = json.loads(self.path.read_text())
        
        # only the supported manifest format accepted
        if self.manifest['format'] != 'lumina-generated-v1' or self.manifest['role'] != 'training':
            raise ValueError('Expected a generated training dataset')
        self.records = self.manifest['records']
        
        # keep generated IDs unique and separate from original OPFData IDs
        ids = [r['id'] for r in self.records]
        if len(set(ids)) != len(ids) or any(not i.startswith('generated:') for i in ids):
            raise ValueError('Generated IDs must be unique and distinct from original sample IDs')
        for record in self.records:
            path = self.path.parent / record['file']
            
            # resolve paths to ensure each file sits directly
            if path.resolve().parent != (self.path.parent / 'samples').resolve():
                raise ValueError('Generated sample path must be in samples/')
            if file_hash(path) != record['sha256']:
                raise ValueError(f'Generated sample changed: {path}')

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        """Load a graph on CPU, validate its identity, and return a clone"""
        record = self.records[index]
        value = torch.load(self.path.parent / record['file'], map_location='cpu', weights_only=False)
        if value['id'] != record['id'] or 'sample_id' in value['graph']:
            raise ValueError('Generated sample identity mismatch')
        return value['graph'].clone()
