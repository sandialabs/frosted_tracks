Installing Frosted Tracks
=========================

If you're reading this message, congratulations, you're an early adopter!  

From PyPI
---------

You can install Frosted Tracks and its dependencies with ``pip install frosted-tracks``.

From source
-----------

First, get a copy of the source from our `Github repository`_.  You can either clone the repository   or downloaqd the source code archive for the latest release.

Second, if you haven't done so already, create a Python virtual environment to hold the ``frosted_tracks`` installation and its dependencies.  This helps avoid version conflicts between different packages.

Third, go to the directory containing the package (whether you've cloned it or unpacked it from a download) and run ``python -m pip install .``.  This will build and install ``frosted-tracks`` and its dependencies.

**Developer Mode**: If you want to work on the Frosted Tracks code to add features or fix bugs, you can install it with ``python -m pip install -e .``.  The "-e" argument tells Python to use the source tree you downloaded instead of copying the library into its package collection.  We welcome comments and pull requests for our repository!

.. _Github repository: https://github.com/sandialabs/frosted_tracks

