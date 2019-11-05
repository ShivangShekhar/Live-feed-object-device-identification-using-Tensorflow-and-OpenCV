# TensorFlow Official Models

The TensorFlow official models are a collection of example models that use
TensorFlow's high-level APIs. They are intended to be well-maintained, tested,
and kept up to date with the latest TensorFlow API. They should also be
reasonably optimized for fast performance while still being easy to read.

These models are used as end-to-end tests, ensuring that the models run with the
same speed and performance with each new TensorFlow build.

## Tensorflow releases

The master branch of the models are **in development**, and they target the
[nightly binaries](https://github.com/tensorflow/tensorflow#installation) built
from the
[master branch of TensorFlow](https://github.com/tensorflow/tensorflow/tree/master).
We aim to keep them backwards compatible with the latest release when possible
(currently TensorFlow 1.5), but we cannot always guarantee compatibility.

**Stable versions** of the official models targeting releases of TensorFlow are
available as tagged branches or
[downloadable releases](https://github.com/tensorflow/models/releases). Model
repository version numbers match the target TensorFlow release, such that
[branch r1.4.0](https://github.com/tensorflow/models/tree/r1.4.0) and
[release v1.4.0](https://github.com/tensorflow/models/releases/tag/v1.4.0) are
compatible with
[TensorFlow v1.4.0](https://github.com/tensorflow/tensorflow/releases/tag/v1.4.0).

If you are on a version of TensorFlow earlier than 1.4, please
[update your installation](https://www.tensorflow.org/install/).

## Requirements

Please follow the below steps before running models in this repo:

1.  TensorFlow
    [nightly binaries](https://github.com/tensorflow/tensorflow#installation)

2.  Add the top-level ***/models*** folder to the Python path with the command:
    `export PYTHONPATH="$PYTHONPATH:/path/to/models"`

    Using Colab: `import os os.environ['PYTHONPATH'] += ":/path/to/models"`

3.  Install dependencies: `pip3 install --user -r official/requirements.txt` or
    `pip install --user -r official/requirements.txt`

To make Official Models easier to use, we are planning to create a pip
installable Official Models package. This is being tracked in
[#917](https://github.com/tensorflow/models/issues/917).

## Available models

**NOTE:** Please make sure to follow the steps in the
[Requirements](#requirements) section.

*   [bert](bert): A powerful pre-trained language representation model: BERT,
    which stands for Bidirectional Encoder Representations from Transformers.
*   [mnist](mnist): A basic model to classify digits from the MNIST dataset.
*   [resnet](vision/image_classification): A deep residual network that can be
    used to classify both CIFAR-10 and ImageNet's dataset of 1000 classes.
*   [transformer](transformer): A transformer model to translate the WMT English
    to German dataset.
*   [ncf](recommendation): Neural Collaborative Filtering model for
    recommendation tasks.

Models that will not update to TensorFlow 2.x stay inside R1 directory:

*   [boosted_trees](r1/boosted_trees): A Gradient Boosted Trees model to
    classify higgs boson process from HIGGS Data Set.
*   [wide_deep](r1/wide_deep): A model that combines a wide model and deep
    network to classify census income data.

## More models to come!

We are in the progress to revamp official model garden with TensorFlow 2.0 and
Keras. In the near future, we will bring:

*   State-of-the-art language understanding models: XLNet, GPT2, and more
    members in Transformer family.
*   Start-of-the-art image classification models: EfficientNet, MnasNet and
    variants.
*   A set of excellent objection detection models.

If you would like to make any fixes or improvements to the models, please
[submit a pull request](https://github.com/tensorflow/models/compare).

## New Models

The team is actively working to add new models to the repository. Every model
should follow the following guidelines, to uphold the our objectives of
readable, usable, and maintainable code.

**General guidelines** * Code should be well documented and tested. * Runnable
from a blank environment with relative ease. * Trainable on: single GPU/CPU
(baseline), multiple GPUs, TPU * Compatible with Python 2 and 3 (using
[six](https://pythonhosted.org/six/) when necessary) * Conform to
[Google Python Style Guide](https://github.com/google/styleguide/blob/gh-pages/pyguide.md)

**Implementation guidelines**

These guidelines exist so the model implementations are consistent for better
readability and maintainability.

*   Use [common utility functions](utils)
*   Export SavedModel at the end of training.
*   Consistent flags and flag-parsing library
    ([read more here](utils/flags/guidelines.md))
*   Produce benchmarks and logs ([read more here](utils/logs/guidelines.md))
