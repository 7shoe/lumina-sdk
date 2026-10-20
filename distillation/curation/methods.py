"""The five pilot methods and one explicit physics-feature herding sibling."""
import numpy as np

from distillation.experiments.speculative_followup.data import kcenter
from distillation.experiments.speculative_followup_two.scores import select
from distillation.experiments.speculative_followup_three.curation import balanced, features, targets
from distillation.experiments.speculative_followup_three.novel import ridge_features, teaching_labels


def choose(method, source, config):
    n, seed = config['count'], config['data_seed']
    strata, width = config['scoring']['strata'], config['rff_features']
    if method == 'KC':
        return kcenter(source['features'], n)
    if method in ('DQS', 'LDM', 'ALC'):
        return select(np.zeros(len(source['ids'])), n, source['demand'], seed=seed,
                      strata=n if method == 'DQS' else strata)
    if method not in ('ITH', 'PIH'):
        raise ValueError(f'Unknown curation method: {method}')
    blocks = [features(source['features'], seed, width),
              features(targets(source['graphs']).double().numpy(), seed+1, width)]
    if method == 'PIH':
        from .physics import descriptor
        blocks.append(features(np.stack([descriptor(g) for g in source['graphs']]), seed+2, width))
    # Equal kernel-block weights; physics does not receive extra weight for dimension.
    embedded = np.concatenate(blocks, axis=1)/np.sqrt(len(blocks))
    return balanced(source, embedded, n, strata, 'herding')


def correct_labels(config, source, ids, predictions, excluded):
    if excluded not in config['committee'] or set(predictions) != set(config['committee'])-{excluded}:
        raise ValueError('ALC requires exactly the other seven teacher architectures')
    phi = ridge_features(source, predictions, config['data_seed'], config['novel']['feature_width'])
    graphs, info = teaching_labels(config, source, ids, phi)
    return graphs, dict(info, target_mode='learned', correction='anchored ridge label solve',
                       excluded_student=excluded, teacher_models=list(predictions))

