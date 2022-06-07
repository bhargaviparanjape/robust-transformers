""" Dataset wrapping an Arrow Table. Extended to also contain group information"""
from datasets.arrow_dataset import Dataset

from typing import (
    TYPE_CHECKING,
    Any,
    BinaryIO,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
)

from datasets.formatting import format_table, get_format_type_from_alias, get_formatter, query_table
from datasets.info import DatasetInfo
from datasets.splits import NamedSplit, Split, SplitInfo
from datasets.table import Table
from datasets.utils import logging


if TYPE_CHECKING:
    from datasets.dataset_dict import DatasetDict

logger = logging.get_logger(__name__)

class GroupRobustDataset(Dataset):
    def __init__(
        self,
        arrow_table: Table,
        info: Optional[DatasetInfo] = None,
        split: Optional[NamedSplit] = None,
        indices_table: Optional[Table] = None,
        fingerprint: Optional[str] = None,
    ):
        super().__init__(arrow_table, info, split, indices_table, fingerprint)

    def update_group_information():
        pass
