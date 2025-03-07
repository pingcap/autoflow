class EmbeddingColumnMismatchError(ValueError):
    """
    Exception raised when the existing embedding column does not match the expected dimension.

    Attributes:
        existing_col (str): The definition of the existing embedding column.
        expected_col (str): The definition of the expected embedding column.
    """

    def __init__(self, existing_col, expected_col):
        self.existing_col = existing_col
        self.expected_col = expected_col
        super().__init__(
            f"The existing embedding column ({existing_col}) does not match the expected dimension ({expected_col})."
        )
