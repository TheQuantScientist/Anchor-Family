import importlib

from torch.utils.data import Dataset, DataLoader, default_collate
from utils.ExpConfigs import ExpConfigs
from data.data_provider.perturbations import apply_history_controls

def data_provider(configs: ExpConfigs, flag: str, shuffle_flag: bool = None, drop_last: bool = None) -> tuple[Dataset, DataLoader]:
    '''
    - flag: "train", "val", "test", "test_all"
    - shuffle_flag: In rare cases, it can be manually overwrite.
    - drop_last: In rare cases, it can be manually overwrite.
    '''
    # backward compatibility
    assert not (shuffle_flag or drop_last), "Please use --train_val_loader_shuffle 0/1 and --train_val_loader_drop_last 0/1 to set shuffle_flag and drop_last for train/val dataloader instead."
    # dynamically import the desired dataset class
    dataset_module = importlib.import_module(f"data.data_provider.datasets.{configs.dataset_name}")
    Data = dataset_module.Data

    # try to load custom collate_fn for the dataset, if present
    try:
        collate_fn = getattr(dataset_module, configs.collate_fn)
    except:
        collate_fn = None

    if flag in ["test", "test_all"]:
        shuffle_flag = False
        drop_last = False
        batch_size = configs.batch_size
    else:
        shuffle_flag = configs.train_val_loader_shuffle or True
        drop_last = configs.train_val_loader_drop_last or True
        batch_size = configs.batch_size

    data_set: Dataset = Data(
        configs=configs,
        flag=flag,
        # DEBUG: temporal change
        # **configs._asdict()
    )

    if configs.history_perturbation != "original" or float(configs.history_keep_fraction) < 1.0:
        base_collate_fn = collate_fn or default_collate

        def controlled_collate_fn(batch):
            return apply_history_controls(base_collate_fn(batch), configs=configs, flag=flag)

        collate_fn = controlled_collate_fn

    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=configs.num_workers,
        drop_last=drop_last,
        collate_fn=collate_fn
    )
    return data_set, data_loader
