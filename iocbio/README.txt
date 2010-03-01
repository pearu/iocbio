
How to add subpackages to iocbio?
=================================

To add a subpackage, say ``foo``, to ``iocbio`` package,
insert the following line::

  config.add_subpackage('foo')

to ``iocbio/setup.py`` file and copy the directory of the ``foo``
package to ``iocbio/`` directory.

After installing ``iocbio``, the subpackage ``foo`` will be
accessible as follows::

  >>> import iocbio.foo
