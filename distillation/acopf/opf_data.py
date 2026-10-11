import os
import socket
from pathlib import Path

from torch_geometric.data import download_url, extract_tar
from torch_geometric.datasets import OPFDataset


def get_data_path():
    hostname = socket.gethostname()
    host = hostname.lower()
    if host.startswith(("aurora-", "x")):
        return Path("/lus/flare/projects/FRAME-IDP/siebenschuh/acopf_datasets/raw/OPFData"), True
    if "macbook" in host:
        return Path.home() / "data" / "datasets" / "OPFData", False
    raise RuntimeError(
        f"Unsupported host {hostname!r}: expected Aurora (aurora-* or x*) or MacBook."
    )


class SharedArchiveOPFDataset(OPFDataset):
    @property
    def raw_dir(self):
        return str(Path(self.root) / self._release)

    def download(self):
        for name in self.raw_file_names:
            download_url(f"{self.url}/{self._release}/{name}", self.raw_dir)

    def process(self):
        for name in self.raw_file_names:
            extract_tar(str(Path(self.raw_dir) / name), self.raw_dir)
        super().process()


def load_dataset(
    case_name="pglib_opf_case118_ieee",
    split="train",
    num_groups=None,
    topological_perturbations=False,
):
    """
    Return OPFData (up to certain network size)
    """
    
    # allowed (lumina-bench sizes)
    allowed_cases = (
        "pglib_opf_case118",
        "pglib_opf_case14",
        "pglib_opf_case2000",
        "pglib_opf_case30",
        "pglib_opf_case4661",
        "pglib_opf_case500",
        "pglib_opf_case57",
    )

    if not isinstance(case_name, str) or not any(
        case_name == name or case_name.startswith(name + "_")
        for name in allowed_cases
    ):
        raise ValueError(
            f"Unsupported case_name {case_name!r}. "
            f"Allowed cases: {', '.join(allowed_cases)} (with optional suffixes)."
        )

    opf_path, use_shared_archives = get_data_path()

    if num_groups is None:
        num_groups = int(os.environ.get("OPF_NUM_GROUPS", "5"))

    dataset_type = SharedArchiveOPFDataset if use_shared_archives else OPFDataset
    
    
    return dataset_type(
        root=str(opf_path),
        case_name=case_name,
        split=split,
        num_groups=num_groups,
        topological_perturbations=topological_perturbations,
    )
    

if __name__ == "__main__":
    opf_path, _ = get_data_path()
    print(f"Host: {socket.gethostname()}\nOPFData root: {opf_path}")
    ds = load_dataset()
    data = ds[int(os.environ.get("OPF_SAMPLE_INDEX", "10000"))]
    print(data)
