import os
from typing import Any, Dict, List, Optional, Union

import lightning.pytorch as pl
import numpy as np
import pandas as pd
import torch
from sklearn import preprocessing
from torch.utils.data import DataLoader, random_split

# TODO: add stratified split
# TODO: add crossvalidation splits by index


class PandasDatamodule(pl.LightningDataModule):
    """Pandas datamodule.

    Args:
        csv_path (str): path to csv file that will be used to create the datamodule.
            Requires that all columns included in the file will be used for model training.
        batch_size (int, optional): batch size.
            Defaults to 1.
        num_workers (int, optional): number of workers.
            Defaults to 0.
        val_ratio (float, optional): validation ratio. Train Ratio will be calculated
            using the val and test ratio to prevent unused data. Defaults to 0.1.
        test_ratio (float, optional): test ratio. Train Ratio will be calculated using
            the val and test ratio to prevent unused data.
            Defaults to 0.1.
        pin_memory (bool, optional): pin memory for gpu memory optimizations.
            Defaults to False.
        classification (bool, optional): classification flag to change the datatype for
            the labels yielded by the dataloader. Must also set unsupervised to True.
            Defaults to False.
        unsupervised (bool, optional): unsupervised flag that uses all columns for input
            if set True. If False, all columns except the last are used for the input and
            the last one is used as the label.
            Defaults to False.
    """

    def __init__(
        self,
        csv_path,
        batch_size=1,
        num_workers=0,
        val_ratio=0.1,
        test_ratio=0.1,
        pin_memory=False,
        classification=False,
        unsupervised=False,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.csv_path = csv_path
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.input_ratio = (
            1 - val_ratio - test_ratio
        )  # unused, declared for easy access
        self.input_len = None
        self.val_ratio = val_ratio
        self.val_len = None
        self.test_ratio = test_ratio
        self.test_len = None
        self.classification = classification
        self.unsupervised = unsupervised

        self.input_cols: Union[List[str], None] = None
        self.label_cols: Union[List[str], None] = None

        self.pin_memory = pin_memory

    def prepare_data(self):
        """Download data if needed. Lightning ensures that `self.prepare_data()` is called only
        within a single process on CPU, so you can safely add your downloading logic within. In
        case of multi-node training, the execution of this hook depends upon
        `self.prepare_data_per_node()`.

        Do not use it to assign state (self.x = y).
        """
        pass

    def setup(self, stage=None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by Lightning before `trainer.fit()`, `trainer.validate()`, `trainer.test()`, and
        `trainer.predict()`, so be careful not to execute things like random split twice! Also, it is called after
        `self.prepare_data()` and there is a barrier in between which ensures that all the processes proceed to
        `self.setup()` once the data is prepared and available for use.

        :param stage: The stage to setup. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`. Defaults to ``None``.
        """
        assert os.path.isfile(self.csv_path)
        self.dataframe = pd.read_csv(self.csv_path)
        self.columns = self.dataframe.columns

        self.handle_cols()

        self.create_dataset()

        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            self.dataset,
            [
                self.input_len,  # type: ignore
                self.val_len,  # type: ignore
                self.test_len,  # type: ignore
            ],
        )

    def create_dataset(self):
        """Creates a PyTorch dataset from the Pandas dataframe stored in `self.dataframe`. If
        `self.unsupervised` is False, the dataset will include both input and label data.
        Otherwise, only input data will be included.

        Returns:
            A PyTorch TensorDataset object containing the input and label data (if applicable).
        """
        # No na values when creating the input
        """self.dataframe = self.dataframe.dropna(subset=[self.input_cols])

        if not self.unsupervised:
            self.dataframe = self.dataframe.dropna(subset=[self.label_cols])"""

        pd_input_data = self.dataframe[self.input_cols]
        pd_input_data = pd_input_data.astype(np.float32)  # type: ignore

        input_values = pd_input_data.values
        self.input_data = torch.tensor(input_values)

        self.val_len = int(len(self.input_data) * self.val_ratio)
        self.test_len = int(len(self.input_data) * self.test_ratio)
        self.input_len = len(self.input_data) - self.val_len - self.test_len

        if not self.unsupervised:
            pd_label_data = self.dataframe[self.label_cols]

            if self.classification:
                pd_label_data = pd_label_data.astype(np.int32)  # type: ignore
            else:
                pd_label_data = pd_label_data.astype(np.float32)  # type: ignore

            label_values = pd_label_data.values
            self.label_data = torch.tensor(label_values)

            self.dataset = torch.utils.data.TensorDataset(
                self.input_data, self.label_data
            )
        else:
            self.dataset = torch.utils.data.TensorDataset(self.input_data)

    def handle_cols(self):  # noqa: F811
        """Encodes categorical columns in the input dataframe using OrdinalEncoder. Determines
        input and label columns based on whether the datamodule is unsupervised or not. Returns the
        list of categorical columns, input columns, and label columns.

        :return: Tuple containing the list of categorical columns, input columns, and label
            columns.
        :rtype: Tuple[List[str], List[str], Union[str, None]]
        """

        self.col_encoder = preprocessing.OrdinalEncoder()

        if self.unsupervised:
            self.input_cols = self.dataframe.columns.values  # type: ignore
            self.label_cols = None  # type: ignore
        else:
            self.input_cols = self.dataframe.columns[:-1].values  # type: ignore
            self.label_cols = self.dataframe.columns[-1]  # type: ignore

        cat_cols = []
        for col in self.input_cols:
            if self.dataframe[col].dtype == "object":
                cat_cols.append(col)

        if self.label_cols:
            if self.dataframe[self.label_cols].dtype == "object":
                cat_cols.append(self.label_cols)

        if cat_cols:
            cat_encoded = self.col_encoder.fit_transform(
                self.dataframe[cat_cols]
            )
            self.dataframe[cat_cols] = pd.DataFrame(
                cat_encoded, columns=cat_cols
            )

        return cat_cols, self.input_cols, self.label_cols

    def train_dataloader(self):
        """Create and return the train dataloader.

        :return: The train dataloader.
        """
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        """Create and return the val dataloader.

        :return: The val dataloader.
        """
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        """Create and return the test dataloader.

        :return: The test dataloader.
        """
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
