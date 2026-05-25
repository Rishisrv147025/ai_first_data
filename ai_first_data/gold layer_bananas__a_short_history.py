import sys

# gold_bananas.py
# ==============================================================================
# ABORT: The provided Source-to-Target Mapping Document is not a valid mapping
# document. It is a narrative article about the history of bananas. There are no
# source columns, target columns, data types, or transformation rules defined.
# No valid execution plan can be generated from this input.
# ==============================================================================

raise ValueError(
    "No valid Source-to-Target Mapping was supplied. "
    "The provided document is a narrative history of bananas and contains no "
    "column definitions, table definitions, or transformation logic. "
    "Cannot generate an executable Gold layer script without a valid mapping."
)