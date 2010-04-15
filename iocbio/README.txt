
How to add new subpackages to iocbio?
=====================================

To add a subpackage, say ``foo``, to ``iocbio`` package,
insert the following line::

  config.add_subpackage('foo')

to ``iocbio/setup.py`` file and copy the directory of the ``foo``
package to ``iocbio/`` directory.

After installing ``iocbio``, the subpackage ``foo`` will be
accessible as follows::

  >>> import iocbio.foo

For generating documentation from package sources, add subpackage
information to the following files::

  iocbio/__init__.py

Subpackage Python scripts must be in ``foo/scripts/`` directory and
optparse argument definitions should be in ``foo/script_options.py``.
Installation of scripts will add ``iocbio.`` prefix to scripts names.
