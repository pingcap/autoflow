from autoflow.data_types import DataType
from autoflow.loaders import Loader


def get_loader_for_datatype(datatype: DataType) -> Loader:
    if datatype == DataType.MARKDOWN:
        from autoflow.loaders.markdown import MarkdownLoader

        return MarkdownLoader()
    elif datatype == DataType.PDF:
        from autoflow.loaders.pdf import PDFLoader

        return PDFLoader()
    elif datatype == DataType.HTML:
        from autoflow.loaders.webpage import WebpageLoader

        return WebpageLoader()
    else:
        raise ValueError(f"Unsupported loader for data type: {datatype}")
