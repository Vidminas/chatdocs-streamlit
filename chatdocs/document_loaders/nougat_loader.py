from functools import partial
from typing import Optional, Iterator
import re

import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from langchain.docstore.document import Document
from langchain.document_loaders.pdf import BasePDFLoader

from chatdocs.logger import logger


class NougatPDFLoader(BasePDFLoader):
    """Load `PDF` files using Nougat (https://facebookresearch.github.io/nougat/)."""

    def __init__(
        self,
        file_path: str,
        *,
        num_workers: Optional[int] = 0,
        headers: Optional[dict] = None,
    ) -> None:
        """Initialize with file path."""
        super().__init__(file_path, headers=headers)
        self.num_workers = num_workers

        try:
            from nougat import NougatModel
            from nougat.utils.checkpoint import get_checkpoint
            from nougat.utils.device import move_to_device
        except ImportError:
            raise ImportError(
                "`nougat` package not found, please install it with "
                "`pip install nougat-ocr`"
            )
        checkpoint = get_checkpoint("nougat", download=True)
        self.model = NougatModel.from_pretrained(checkpoint)

        if torch.cuda.is_available():
            self.batch_size = int(
                torch.cuda.get_device_properties(0).total_memory
                / 1024
                / 1024
                / 1000
                * 0.3
            )
            if self.batch_size == 0:
                self.batch_size = 1
                logger.warning("GPU VRAM is too small. Computing on CPU.")
        elif torch.backends.mps.is_available():
            self.batch_size = 4
        else:
            self.batch_size = 1
            logger.warning("No GPU found. Conversion on CPU is very slow.")

        self.model = move_to_device(self.model)
        self.model.eval()

    def load(self) -> list[Document]:
        """Eagerly load the content."""
        return list(self.lazy_load())

    def lazy_load(
        self,
    ) -> Iterator[Document]:
        """Lazily load documents."""
        import pypdf
        from nougat.utils.dataset import LazyDataset
        from nougat.postprocessing import markdown_compatible

        try:
            dataset = LazyDataset(
                pdf=self.file_path,
                prepare=partial(self.model.encoder.prepare_input, random_padding=False),
            )
        except pypdf.errors.PdfStreamError:
            logger.info(f"Could not load file {str(self.file_path)}.")
            return

        dataloader = DataLoader(
            dataset,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=LazyDataset.ignore_none_collate,
        )

        for page_num, (sample, is_last_page) in enumerate(
            tqdm(
                dataloader,
                desc=f"Processing file {dataset.name} with {dataset.size} pages",
                ncols=80,
                position=0,
                leave=True,
            )
        ):
            model_output = self.model.inference(
                image_tensors=sample, early_stopping=True
            )
            # check if model output is faulty
            for i, output in enumerate(model_output["predictions"]):
                if output.strip() == "[MISSING_PAGE_POST]":
                    # uncaught repetitions -- most likely empty page
                    logger.warning(f"[MISSING_PAGE_EMPTY:{page_num}]")
                elif model_output["repeats"][i] is not None:
                    if model_output["repeats"][i] > 0:
                        # If we end up here, it means the output is most likely not complete and was truncated.
                        logger.warning(f"Skipping page {page_num} due to repetitions.")
                    else:
                        # If we end up here, it means the document page is too different from the training domain.
                        # This can happen e.g. for cover pages.
                        logger.warning(f"[MISSING_PAGE_EMPTY:{i+1}]")
                else:
                    output = markdown_compatible(output)
                    output = re.sub(r"\n{3,}", "\n\n", output).strip()
                    metadata = {"source": self.file_path, "page": page_num}
                    yield Document(page_content=output, metadata=metadata)
