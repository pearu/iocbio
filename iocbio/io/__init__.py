"""Microscope data I/O tools

Overview
========

.. currentmodule:: iocbio.io

The :mod:`iocbio.io` provides the following I/O related tools to
read, write, and hold microscope data:

.. autosummary::

  io.load_image_stack
  io.save_image_stack
  image_stack.ImageStack
  io.RowFile
  cacher.Cacher

The front-end class for I/O tasks is
`iocbio.io.image_stack.ImageStack`, see `iocbio.io.image_stack` for
more information.
"""

__autodoc__ = ['image_stack', 'pathinfo', 'io', 
               'RowFile', 'ImageStack', 'load_image_stack','save_image_stack',
               'cacher', 'Cacher']

from .image_stack import ImageStack
from .io import RowFile, load_image_stack, save_image_stack
from .cacher import Cacher
import pathinfo

