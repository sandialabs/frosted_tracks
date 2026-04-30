Installing Frosted Tracks
=========================

If you're reading this message, congratulations, you're an early adopter!  The code is still hot from being uploaded and we're working on the PyPI and conda-forge recipes in real time.

From PyPI
---------

Once this code is officially released, we'll upload wheels to `PyPI <https://pypi.org>`_ so that you can install with ``pip install frosted_tracks``.

From conda-forge
----------------

We're going to contribute a recipe to `conda-forge <https://conda-forge.org>`_ so that you can install with ``conda install -c conda-forge frosted_tracks``.

From source
-----------

First, get a copy of the source.  Our Github repository is at https://github.com/sandialabs/frosted_tracks and you can either clone or download the repository or download the release package for the latest version.

Second, if you haven't done so already, create a Python virtual environment to hold the ``frosted_tracks`` installation and its dependencies.  This helps avoid version conflicts between different packages.

Third, go to the directory containing the package (whether you've cloned it or unpacked it from a download) and run ``python -m pip install .``.  This will build and install ``frosted-tracks`` and its dependencies.

**Developer Mode**: If you want to work on the ``frosted-tracks`` code to add features or fix bugs, you can install it with ``python -m pip install -e .``.  The "-e" argument tells Python to use the source tree you downloaded instead of copying the library into its package collection.