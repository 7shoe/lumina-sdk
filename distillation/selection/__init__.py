"""Reusable real-data selection infrastructure; no acquisition objective."""

from .pool import (TrainingPool,
                   input_view,
                   ids_hash,
                   selected_dataset)
from .buckets import (Buckets,
                      allocate)

__all__ = ['TrainingPool',
           'input_view',
           'ids_hash',
           'selected_dataset',
           'Buckets',
           'allocate']
