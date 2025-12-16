"""Project-wide runtime tweaks to keep third-party warnings quiet."""

# Ensure Starlette uses the maintained multipart parser implementation.
import python_multipart  # type: ignore  # noqa: F401
