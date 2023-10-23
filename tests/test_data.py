from sklearn.model_selection import train_test_split
from gpnpytorchtools import data
import numpy as np
import pandas as pd


def test_pandas_datamodule():
    test_array = np.random.rand(10, 10)
    str_col = np.array([f"str{i}" for i in range(10)])
    test_array = np.concatenate((test_array, str_col[:, None]), axis=1)
    df = pd.DataFrame(test_array)
    df.to_csv("test.csv")
    dm = data.PandasDatamodule(csv_path="test.csv")
    dm.prepare_data()
    dm.setup()
    train = dm.train_dataloader()
    val = dm.val_dataloader()
    test = dm.test_dataloader()

    dm = data.PandasDatamodule(csv_path="test.csv", unsupervised=True)
    dm.prepare_data()
    dm.setup()

    dm = data.PandasDatamodule(csv_path="test.csv", classification=True)
    dm.prepare_data()
    dm.setup()

    try:
        next(iter(dm.val_dataloader()))
    except RuntimeError as e:
        assert False
