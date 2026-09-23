"""Errors raised by the pipeline.

These are ordinary exceptions rather than ``SystemExit``, which is what the
command-line versions of this code used to raise. That distinction matters once
the same functions run inside the GUI: ``SystemExit`` derives from
``BaseException``, so Qt's ``except Exception`` handlers do not catch it, and an
exception escaping a slot aborts the process. A mistyped key in a profile file
would have closed the application.

The command-line wrappers in ``tools/`` catch these and convert them back into a
clean ``SystemExit`` with the same message, so the CLI behaviour is unchanged.
"""


class PipelineError(Exception):
    """Base class, so callers can catch everything this package raises."""


class PreprocessingError(PipelineError):
    """A frame-enhancement profile could not be read or made sense of."""


class SamplingError(PipelineError):
    """Frames could not be sampled from the given videos."""


class DatasetError(PipelineError):
    """A COCO dataset could not be built from the given annotations."""
