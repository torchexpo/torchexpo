Get Started with TorchExpo
##########################

Prerequisites
=============

Python
------

It is recommended that you use Python 3.5 or greater, which can be installed from `Python website <https://www.python.org/downloads/>`_.


Package Manager
---------------

To install the TorchExpo, you will need to use one of the supported package managers: `pip <https://pip.pypa.io/en/stable/>`_.
pip is the recommended package manager as it will provide you all of the dependencies in one, sandboxed install, including Python.

pip
~~~

*Python 3*

If you installed Python via the Python website, ``pip`` was installed with it. If you installed Python 3.x, then you will be using the command ``pip3``.

Anaconda
~~~~~~~~

**Coming Soon**


Installation
============

pip
---

To install TorchExpo via pip, use following command:

.. code-block:: bash

   # Python 3.x
   pip3 install torchexpo


Anaconda
--------

To install TorchExpo via Anaconda, use following command:

**Coming Soon**

Building from Source
====================

For the majority of TorchExpo users, installing from a pre-built binary via a package manager will provide the best experience. However, there are times when you may want to install the nightly TorchExpo code, whether for testing or actual development. To install the latest TorchExpo code, you will need to build TorchExpo from source.

.. code-block:: bash

   git clone https://github.com/torchexpo/torchexpo.git
   cd torchexpo
   python3 setup.py install