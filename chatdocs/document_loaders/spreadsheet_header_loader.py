from typing import Optional

import pandas as pd
from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader


class SpreadsheetHeaderLoader(BaseLoader):
    def __init__(
        self,
        file_path: str,
        encoding: Optional[str] = None,
        csv_args: Optional[dict] = {
            "low_memory": False,
            "nrows": 1000,
        },
    ):
        """
        Args:
            file_path: The path to the CSV file.
            encoding: The encoding of the CSV file. Optional. Defaults to None.
            csv_args: A dictionary of arguments to pass to the pd.read_csv.
              Optional. Defaults to {
                "low_memory": False,
                "nrows": 1000,
            },
        """
        self.file_path = file_path
        self.encoding = encoding
        self.csv_args = csv_args or {}

    def convert_column_to_datetime(self, column):
        if (not pd.api.types.is_object_dtype(column.dtype) and
            not pd.api.types.is_string_dtype(column.dtype)):
            return None

        try:
            result = pd.to_datetime(column, infer_datetime_format=True, errors="raise")
            return result
        except (pd.errors.ParserError, TypeError, ValueError):
            return None

    def load(self) -> list[Document]:
        """Load data into document objects."""

        docs = []
        csv = pd.read_csv(self.file_path, encoding=self.encoding, **self.csv_args)
        csv = csv.convert_dtypes()

        for cname in csv.columns:
            converted = self.convert_column_to_datetime(csv[cname])
            if converted is not None:
                csv[cname] = converted

        for cname, ctype in zip(csv.columns, csv.dtypes):
            metadata = {"source": self.file_path, "dtype": str(ctype)}
            content = cname
            docs.append(Document(page_content=content, metadata=metadata))

        return docs
