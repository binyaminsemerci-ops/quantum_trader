"""Global pytest configuration hooks."""

# Ensure Starlette wires the maintained multipart parser before FastAPI loads.
import python_multipart  # type: ignore  # noqa: F401

import warnings

warnings.filterwarnings(
	"ignore",
	category=PendingDeprecationWarning,
	module="starlette.formparsers",
)
