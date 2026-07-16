from torch_geometric.data import Dataset


class GraphDataset(Dataset):
    def __init__(self, data_list: list) -> None:
        super().__init__()
        self.data_list = data_list

    def len(self) -> int:
        return len(self.data_list)

    def get(self, idx: int) -> object:
        return self.data_list[idx]
