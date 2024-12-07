# Frosted Tracks Tests: Care and Feeding

We use Pytest to manage our test cases.  Here's what you need to know.

For the most part, our tests follow this pattern:

1. Load some predefined ("golden") input data.
2. Execute some function on that data to generate an output.
3. Compare that output with a known-good version that we saved earlier.

The parts you need to keep track of involve how to generate, wrangle, save, and load the inputs and outputs.

## Your Friend the Oracle

We have a helper class `frosted_tracks.test.infrastructure.ground_truth.GroundTruthOracle` to keep track of artifacts we generate.  It maintains a separate store for each module.  Within each store, it keeps track of objects with user-specified strings for names.

For "module", think "file containing test cases".  `frosted_tracks.test.ticc.test_ticc_initial_labels` is a module.

An oracle has two methods you'll need:

- `put(data, key)`: Save an object with a certain key
- `get(key)`: Load a previously saved object

For the most part you don't instantiate oracles yourself.  Keep reading.

### Generating Golden Inputs

Sometimes we need to freeze a test's input to avoid platform- or OS-dependent
variation.  We do this by computing an input once and storing it in an oracle.
All of this is done in `frosted_tracks.test.golden_test_inputs`.

If a test input is simple enough to be a fixture, make it a fixture.  Golden
inputs are meant for things like sets of features whose values can be affected
ever-so-slightly by different versions of Tracktable and underlying libraries --
not enough to be meaningful, but enough to throw off the least significant bits
of floating-point computations and make it difficult to assess whether test
results differ because of problems in the code or because of numerical fiddliness.

If you really do need a golden input, add code to generate it in the
`generate_ground_truth` function in `golden_test_inputs.py`.


### Generating Ground Truth

When you write a file containing a set of tests, you will also write a function named `generate_ground_truth(oracle)`.  The name of the function has to be exactly `generate_ground_truth` and it has to take exactly one argument.  The name of that argument doesn't matter, but it's clearest to call it `oracle`.

Inside your `generate_ground_truth()` function you will build all of the ground-truth
outputs that you will use to evaluate your tests for success.  As you build each one,
store it in the oracle.  You don't have to configure the oracle or tell it what module
to use -- that's handled for you.


### Running the Ground Truth Generators

There is a script `generate_test_ground_truth.py` that will walk through all of `frosted_tracks.test` to find functions named `generate_ground_truth()`.  It collects the functions and
their containing modules, then runs each ground truth generator with an oracle that
points to a store for its containing module.

To run this, make sure the Frosted Tracks package is installed with `pip install -e` or on your `PYTHONPATH` and then run...

`python -m frosted_tracks.test.generate_test_ground_truth`

