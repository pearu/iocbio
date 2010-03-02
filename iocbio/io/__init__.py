"""Microscope data I/O tools.

.. currentmodule:: iocbio.io

The :mod:`iocbio.io` provides the following I/O related modules to
read, write, and hold microscope data:

.. autosummary::

  image_stack
  pathinfo
  io

The front-end class for I/O tasks is
`iocbio.io.image_stack.ImageStack`, see `iocbio.io.image_stack` for
more information.

"""
from .image_stack import ImageStack
from .io import RowFile, load_image_stack, save_image_stack
