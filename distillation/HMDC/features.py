"""Graph spatial/semantic alignment for all eight unchanged Lumina backbones."""
import torch
from torch import nn

# canonical (AC-)OPF graph definition
NODE_TYPES = ('bus',
              'generator',
              'load',
              'shunt')

def node_graph_ids(batch):
    return torch.cat([batch[k].batch for k in NODE_TYPES])

def pooled(nodes, ids, count):
    totals = nodes.new_zeros((count, *nodes.shape[1:])).index_add(0, ids, nodes)
    sizes = torch.bincount(ids, minlength=count).clamp_min(1)
    return totals / sizes.reshape(-1, *([1] * (nodes.ndim - 1)))


def forward_features(model, batch):
    """Read each post-ReLU message-passing layer without altering predictions.

    All current benchmark backbones apply ReLU after each convolution. Dropout
    is disabled. Features use canonical node-type order, including empty stores;
    both models therefore refer to the same physical nodes, without interpolation.
    """
    if model.training:
        raise ValueError('HMDC feature capture requires eval mode (no dropout)')
    features, handles = [], []

    def capture(_, __, output):
        if isinstance(output, dict):
            values = [output[k].relu() for k in NODE_TYPES if batch[k].num_nodes]
        else:
            order = model.backbone.node_types if model.kind == 'HEAT' else NODE_TYPES
            parts = dict(zip(order, output.split([batch[k].num_nodes for k in order])))
            values = [parts[k].relu() for k in NODE_TYPES if batch[k].num_nodes]
        features.append(torch.cat(values))

    try:
        for layer in model.backbone.convs:
            handles.append(layer.register_forward_hook(capture))
        predictions = model(batch)
    finally:
        for handle in handles:
            handle.remove()
    if not features:
        raise ValueError('At least one message-passing layer is required for HMDC')
    return predictions, features


class SpatialSemanticAlignment(nn.Module):
    """HMDC SSD: affine channel alignment and learned softmax layer alignment.

    Node embeddings replace image locations. A graph mean is the semantic token.
    Ragged graphs are reduced independently so larger grids do not dominate.
    Layer softmax orientation follows the pinned authors' implementation.
    """
    def __init__(self, widths_a, widths_b):
        super().__init__()
        if not widths_a or not widths_b or min(*widths_a, *widths_b) <= 0:
            raise ValueError('Nonempty positive layer widths required')
        width = min(*widths_a, *widths_b)
        self.projections = nn.ModuleList([
            nn.ModuleList([nn.Linear(c, width) for c in widths])
            for widths in (widths_a, widths_b)])
        self.layer_logits = nn.Parameter(torch.zeros(len(widths_a), len(widths_b)))

    def align(self, features_a, features_b):
        values = []
        for layers, features in zip(self.projections, (features_a, features_b)):
            if len(layers) != len(features):
                raise ValueError('Feature depth changed after alignment initialization')
            values.append(torch.stack([layer(f) for layer, f in zip(layers, features)], dim=-1))
        a, b = values  # nodes x channels x layers
        if a.shape[0] != b.shape[0]:
            raise ValueError('Models must describe the same ordered nodes')
        if a.shape[-1] < b.shape[-1]:
            b = b @ self.layer_logits.T.softmax(dim=1)
        else:
            a = a @ self.layer_logits.softmax(dim=1)
        return a, b

    def forward(self, features_a, features_b, ids, graph_count):
        a, b = self.align(features_a, features_b)
        spatial = (a - b).square().mean(dim=(1, 2))
        semantic = (pooled(a, ids, graph_count) - pooled(b, ids, graph_count)).square().mean(dim=(1, 2))
        totals = spatial.new_zeros(graph_count).index_add(0, ids, spatial)
        sizes = torch.bincount(ids, minlength=graph_count)
        
        # One semantic token plus each actual node; padding contributes nothing.
        return ((totals + semantic) / (sizes + 1)).mean()

