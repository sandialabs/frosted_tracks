# Frosted Tracks

Analysis of trajectory behavior using TICC and DBSCAN

## Repository Structure

- `src/`: Source code

Best practice: prototype code in a notebook, then move it into
src/frosted_tracks with proper docstrings and test cases when it's
ready to share.  Open a pull request to our GitHub repository if
you'd like to integrate your work into the main trunk!

## Python Environment

We recommend that you use Anaconda (https://www.anaconda.com) for your
Python environment.  If you do, there's an `environment.yml` file in this
repository that you can use to set up your dependencies as follows:

```bash
conda env create -f environment.yml
```

## License

See the file LICENSE in the root directory of the repository for
details.  We release this work under a 3-clause BSD license.


## Changes

Version 1.0: Initial release.  Not distributed to PyPI.

Version 1.1: Experimental cluster predictor disconnected.  It was causing 
    build errors when we tried to construct wheels.  You must now supply
    the desired number of clusters when you call cluster_trajectories.
