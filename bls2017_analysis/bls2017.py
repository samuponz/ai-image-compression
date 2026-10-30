# Copyright 2018 Google LLC. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Basic nonlinear transform coder for RGB images.

This is a close approximation of the image compression model published in:
J. Ballé, V. Laparra, E.P. Simoncelli (2017):
"End-to-end Optimized Image Compression"

Int. Conf. on Learning Representations (ICLR), 2017
https://arxiv.org/abs/1611.01704

With patches from Victor Xing <victor.t.xing@gmail.com>

This is meant as 'educational' code - you can use this to get started with your
own experiments. To reproduce the exact results from the paper, tuning of hyper-
parameters may be necessary. To compress images with published models, see
`tfci.py`.

This script requires TFC v2 (`pip install tensorflow-compression==2.*`).
"""

import argparse
import glob
import sys
from absl import app
from absl.flags import argparse_flags
import tensorflow as tf
import tensorflow_compression as tfc
import tensorflow_datasets as tfds
# --- SAMU
# import tensorrt
from matplotlib import pyplot as plt
import tensor_list_to_tfrecord
# --- 

'''
 FILE STRUCTURE:
  - ADDED FUNTIONS
  - I/O FUNCTIONS
  - ANALYSIS & SYNTHESIS CLASSES
  - MODEL CLASS
  - UTILITIES
  - USER METHODS IMPLEMENTATION
  - PARSE ARGS USER METHODS
  - MAIN

Classes are used to implement the model and the analysis/synthesis. All the other stuff is
implemented with funtions.
'''

# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------
# ADDED FUNCTIONS

def compute_and_save_intermediate_features(x, model, num_filters=128):

  # analysis_transform = AnalysisTransform(num_filters)
  
  x0 = tf.cast(x, tf.float32)
  
  # analysis
  x0 = tf.expand_dims(x0, 0) # add batch dimension
  x0 = model.analysis_transform.layers[0](x0) # normalization (x / 255.)
  x1 = model.analysis_transform.layers[1](x0) # first conv + downsample 4 + GDN
  x2 = model.analysis_transform.layers[2](x1) # second conv + downsample 2 + GDN
  y = model.analysis_transform.layers[3](x2) # third conv + downsample 2 + GDN
  
  # Preserve spatial shapes of both image and latents.
  x_shape = tf.shape(x)[0:-1]
  y_shape = tf.shape(y)[0:-1]
  
  # ~~~ encoding and decoding with entropy model would be done here ~~~

  # synthesis
  # synthesis_transform = SynthesisTransform(num_filters)
  
  x2_hat = model.synthesis_transform.layers[0](y) # first IGDN + upsample 2 + upconv
  x1_hat = model.synthesis_transform.layers[1](x2_hat) # second IGDN + upsample 2 + upconv
  x0_hat = model.synthesis_transform.layers[2](x1_hat) # third IGDN + upsample 4 + upconv
  x_hat = model.synthesis_transform.layers[3](x0_hat) # denormalization (x * 255.)
  
  # Remove batch dimension, and crop away any extraneous padding.
  x_hat = x_hat[0, :x_shape[0], :x_shape[1], :]
  # Then cast back to 8-bit integer.
  x_hat = tf.saturate_cast(tf.round(x_hat), tf.uint8)

  tensor_list_to_tfrecord.write_tensor_list([x, x1, x2, y, x2_hat, x1_hat, x_hat], "intermediate_features.data")

# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------
# I/O FUNCTIONS

# load image from filename
def read_png(filename):
  """Loads a PNG image file."""
  string = tf.io.read_file(filename)
  return tf.image.decode_image(string, channels=3)

# save image with given filename
def write_png(filename, image):
  """Saves an image to a PNG file."""
  string = tf.image.encode_png(image)
  tf.io.write_file(filename, string)


# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------
# ANALYSIS AND SYNTHESIS


# analysis transform model
class AnalysisTransform(tf.keras.Sequential):
  """The analysis transform."""

  # constructor building the network
  def __init__(self, num_filters):

    # super init of keras.Sequential and names the network "analysis"
    super().__init__(name="analysis")
    
    # custom layer implementitf.keras.mixed_precision.Policyng intensity normalization as a lambda expression
    self.add(tf.keras.layers.Lambda(lambda x: x / 255.))

    # first 2D convolution layer (receptive field: 9x9), stride: 4
    # GENERALIZED DIVISIVE NORMALIZATION is used as an activation:
    #   y[i] = x[i] / (beta[i] + sum_j(gamma[j, i] * |x[j]|^alpha))^epsilon
    #   This implementation never sums across spatial dimensions.
    #   It's a form of local contrast normalization that aims to enhance the features of an image.
    #   GDN is particularly useful in scenarios where the local statistics of an image vary, 
    #   as it helps to normalize the responses and enhance the features.
    self.add(tfc.SignalConv2D(
        num_filters, (9, 9), name="layer_0", corr=True, strides_down=4,
        padding="same_zeros", use_bias=True,
        activation=tfc.GDN(name="gdn_0")))
    
    # 2D convolution layer (receptive field: 5x5), stride: 2
    # GENERALIZED DIVISIVE NORMALIZATION is used as an activation:
    self.add(tfc.SignalConv2D(
        num_filters, (5, 5), name="layer_1", corr=True, strides_down=2,
        padding="same_zeros", use_bias=True,
        activation=tfc.GDN(name="gdn_1")))
    
    # 2D convolution layer (receptive field: 5x5), stride: 2
    # GENERALIZED DIVISIVE NORMALIZATION is used as an activation:
    self.add(tfc.SignalConv2D(
        num_filters, (5, 5), name="layer_2", corr=True, strides_down=2,
        padding="same_zeros", use_bias=False,
        activation=None))

# synthesis transform model
class SynthesisTransform(tf.keras.Sequential):
  """The synthesis transform."""

  def __init__(self, num_filters):
    super().__init__(name="synthesis")
    self.add(tfc.SignalConv2D(
        num_filters, (5, 5), name="layer_0", corr=False, strides_up=2,
        padding="same_zeros", use_bias=True,
        activation=tfc.GDN(name="igdn_0", inverse=True)))
    self.add(tfc.SignalConv2D(
        num_filters, (5, 5), name="layer_1", corr=False, strides_up=2,
        padding="same_zeros", use_bias=True,
        activation=tfc.GDN(name="igdn_1", inverse=True)))
    self.add(tfc.SignalConv2D(
        3, (9, 9), name="layer_2", corr=False, strides_up=4,
        padding="same_zeros", use_bias=True,
        activation=None))
    self.add(tf.keras.layers.Lambda(lambda x: x * 255.))


# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------
# MODEL CLASS


class BLS2017Model(tf.keras.Model):
  """Main model class."""

  # Constructor: instantiate the components:
  # - lambda: trade-off parameter
  # - analysis and synthesis transforms
  # - prior: distribution of the latent variables (represented with piecewise polynomials)
  def __init__(self, lmbda, num_filters):
    super().__init__()
    self.lmbda = lmbda
    self.analysis_transform = AnalysisTransform(num_filters)
    self.synthesis_transform = SynthesisTransform(num_filters)
    self.prior = tfc.NoisyDeepFactorized(batch_shape=(num_filters,))
    
    # This is where the weights are created based on the input shape, 
    # which is a 4D tensor with shape (batch_size, height, width, channels).
    self.build((None, None, None, 3))

  # call should define the forward-pass of your model. i.e. how is the input transformed to the output.
  def call(self, x, training):
    """Computes rate and distortion losses."""
    """This method computes rate and distortion losses for an input x.
    - It defines an entropy model using the prior distribution and a specified coding rank.
    - It applies the analysis transform to x, then uses the entropy model to encode the result, obtaining y_hat and bits.
    - It uses the synthesis transform to generate a reconstructed image x_hat.
    - It calculates bits per pixel (bpp) and mean squared error (mse) between x and x_hat.
    - Finally, it computes the loss using a Lagrangian formulation with the trade-off parameter."""

    # Define entropy model: 
    # See arguments at https://www.tensorflow.org/api_docs/python/tfc/entropy_models/ContinuousBatchedEntropyModel
    entropy_model = tfc.ContinuousBatchedEntropyModel(
        self.prior, coding_rank=3, compression=False)
    
    x = tf.cast(x, self.compute_dtype)  # TODO(jonycgn): Why is this necessary?
    y = self.analysis_transform(x)
    y_hat, bits = entropy_model(y, training=training)
    x_hat = self.synthesis_transform(y_hat)
    # Total number of bits divided by total number of pixels.
    num_pixels = tf.cast(tf.reduce_prod(tf.shape(x)[:-1]), bits.dtype)
    bpp = tf.reduce_sum(bits) / num_pixels
    # Mean squared error across pixels.
    mse = tf.reduce_mean(tf.math.squared_difference(x, x_hat))
    mse = tf.cast(mse, bpp.dtype)
    # The rate-distortion Lagrangian.
    loss = bpp + self.lmbda * mse
    return loss, bpp, mse

  # this is run inside the model.fit(), it an override!
  # train_step is called by fit. Basically, fit loops over the dataset and provide each batch to train_step 
  # (and then handles metrics, bookkeeping, etc., of course).
  # call should define the forward-pass of your model. i.e. how is the input transformed to the output.
  # train_step defines the logic of a training step, usually with gradient descent. It will often make 
  # use of call since the training step tends to include a forward pass of the model to compute gradients.

  def train_step(self, x):
    """This method defines a single training step.
    - It uses a gradient tape to compute gradients of the loss with respect to the trainable variables.
    - It applies these gradients using the optimizer.
    - It updates metrics (self.loss, self.bpp, self.mse) to keep track of training progress.
    - It returns a dictionary containing the current values of these metrics."""
    with tf.GradientTape() as tape:
      loss, bpp, mse = self(x, training=True)
    variables = self.trainable_variables
    # compute gradients of the loss with respect to the trainable variables
    gradients = tape.gradient(loss, variables)
    self.optimizer.apply_gradients(zip(gradients, variables))
    self.loss.update_state(loss)
    self.bpp.update_state(bpp)
    self.mse.update_state(mse)
    # return the metrics names and results for this training step
    return {m.name: m.result() for m in [self.loss, self.bpp, self.mse]}

  def test_step(self, x):
    loss, bpp, mse = self(x, training=False)
    self.loss.update_state(loss)
    self.bpp.update_state(bpp)
    self.mse.update_state(mse)
    return {m.name: m.result() for m in [self.loss, self.bpp, self.mse]}
  
  # ???
  def predict_step(self, x):
    raise NotImplementedError("Prediction API is not supported.")

  # I don't fully understand but the optimizer (Adam) is one of the args
  def compile(self, **kwargs):
    super().compile(
        loss=None,
        metrics=None,
        loss_weights=None,
        weighted_metrics=None,
        **kwargs,
    )
    self.loss = tf.keras.metrics.Mean(name="loss")
    self.bpp = tf.keras.metrics.Mean(name="bpp")
    self.mse = tf.keras.metrics.Mean(name="mse")

  # Used to actually fit the model! Uses Keras.Model.fit() method
  def fit(self, *args, **kwargs):
    retval = super().fit(*args, **kwargs)
    # After training, fix range coding tables.
    self.entropy_model = tfc.ContinuousBatchedEntropyModel(
        self.prior, coding_rank=3, compression=True)
    return retval

  # @tf.function is a decorator provided by TensorFlow that allows you to create a graph of the computation
  # defined by a Python function. This can lead to significant speed improvements, especially when dealing 
  # with large computational graphs. 
  # The input_signature argument specifies the shape and data type of the input that the function expects.
  # In this case, it's expecting a 3D tensor with shape (None, None, 3) and data type tf.uint8.
  # The input is clearly the (H, W, 3) shape image to compress.
  @tf.function(input_signature=[
      tf.TensorSpec(shape=(None, None, 3), dtype=tf.uint8),
  ])
  def compress(self, x):
    """Compresses an image."""
    # Add batch dimension and cast to float.
    x = tf.expand_dims(x, 0)
    x = tf.cast(x, dtype=self.compute_dtype)
    y = self.analysis_transform(x)
    # Preserve spatial shapes of both image and latents.
    x_shape = tf.shape(x)[1:-1]
    y_shape = tf.shape(y)[1:-1]
    return self.entropy_model.compress(y), x_shape, y_shape

  # In this case, the input is made of a string and 2 integers
  @tf.function(input_signature=[
      tf.TensorSpec(shape=(1,), dtype=tf.string),
      tf.TensorSpec(shape=(2,), dtype=tf.int32),
      tf.TensorSpec(shape=(2,), dtype=tf.int32),
  ])
  def decompress(self, string, x_shape, y_shape):
    """Decompresses an image."""
    y_hat = self.entropy_model.decompress(string, y_shape)
    x_hat = self.synthesis_transform(y_hat)
    # Remove batch dimension, and crop away any extraneous padding.
    x_hat = x_hat[0, :x_shape[0], :x_shape[1], :]
    # Then cast back to 8-bit integer.
    return tf.saturate_cast(tf.round(x_hat), tf.uint8)


# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------
# UTILITIES


def check_image_size(image, patchsize):
  shape = tf.shape(image)
  return shape[0] >= patchsize and shape[1] >= patchsize and shape[-1] == 3


def crop_image(image, patchsize):
  image = tf.image.random_crop(image, (patchsize, patchsize, 3))
  return tf.cast(image, tf.keras.mixed_precision.global_policy().compute_dtype)


def get_dataset(name, split, args):
  """Creates input data pipeline from a TF Datasets dataset."""
  with tf.device("/cpu:0"):
    dataset = tfds.load(name, split=split, shuffle_files=True)
    if split == "train":
      dataset = dataset.repeat()
    dataset = dataset.filter(
        lambda x: check_image_size(x["image"], args.patchsize))
    dataset = dataset.map(
        lambda x: crop_image(x["image"], args.patchsize))
    dataset = dataset.batch(args.batchsize, drop_remainder=True)
  return dataset


def get_custom_dataset(split, args):
  """Creates input data pipeline from custom PNG images."""
  with tf.device("/cpu:0"):
    files = glob.glob(args.train_glob)
    if not files:
      raise RuntimeError(f"No training images found with glob "
                         f"'{args.train_glob}'.")
    dataset = tf.data.Dataset.from_tensor_slices(files)
    dataset = dataset.shuffle(len(files), reshuffle_each_iteration=True)
    if split == "train":
      dataset = dataset.repeat()
    dataset = dataset.map(
        lambda x: crop_image(read_png(x), args.patchsize),
        num_parallel_calls=args.preprocess_threads)
    dataset = dataset.batch(args.batchsize, drop_remainder=True)
  return dataset


# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------
# USER METHODS IMPLEMENTATIONS


# SAMU
def check_gpu(args):
  print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
  tf.config.list_physical_devices()


def train(args):
  """Instantiates and trains the model."""
  if args.precision_policy:
    tf.keras.mixed_precision.set_global_policy(args.precision_policy)
  if args.check_numerics:
    tf.debugging.enable_check_numerics()

  # instantiation of the model using lmbda and num_filters
  model = BLS2017Model(args.lmbda, args.num_filters)

  # compile is used to initialize the optimizer!
  model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),)

  # train_glob=True means the user wants to use custom data, if different from True
  # then CLIC dataset is used
  if args.train_glob: 
    train_dataset = get_custom_dataset("train", args)
    validation_dataset = get_custom_dataset("validation", args)
  else:
    train_dataset = get_dataset("clic", "train", args)
    validation_dataset = get_dataset("clic", "validation", args)
  validation_dataset = validation_dataset.take(args.max_validation_steps)

  # model fit performes the training, here we pass a lot of arguments
  model.fit(
      train_dataset.prefetch(8),
      epochs=args.epochs,
      steps_per_epoch=args.steps_per_epoch,
      validation_data=validation_dataset.cache(),
      validation_freq=1,

      callbacks=[
          # This is a callback that terminates the training process if it encounters a NaN (Not a Number) value during training. 
          # NaN values can occur if there are numerical stability issues or if the model's weights are not being updated properly.
          tf.keras.callbacks.TerminateOnNaN(),
          
          # This callback is used for logging training information in a format that can be visualized with TensorBoard, 
          # which is a web-based tool provided by TensorFlow for monitoring and debugging machine learning models.
          tf.keras.callbacks.TensorBoard(
              # log_dir=args.train_path: Specifies the directory where the TensorBoard logs will be stored. args.train_path likely contains the path to the directory where the training logs should be saved.
              log_dir=args.train_path,
              # histogram_freq=1: This indicates how often to compute and log histograms for the activations and weights. In this case, it's set to every epoch (1).
              histogram_freq=1, 
              # update_freq="epoch": This specifies how often the TensorBoard logs should be updated. In this case, it's set to once per epoch.
              update_freq="epoch"),

          tf.keras.callbacks.BackupAndRestore(args.train_path),
      ],
      verbose=int(args.verbose),
  )
  model.save(args.model_path)


def compress(args):
  """Compresses an image."""
  # Load model and use it to compress the image.
  model = tf.keras.models.load_model(args.model_path)
  x = read_png(args.input_file)
  compute_and_save_intermediate_features(x, model)
  tensors = model.compress(x)

  # Write a binary file with the shape information and the compressed string.
  packed = tfc.PackedTensors()
  packed.pack(tensors)
  with open(args.output_file, "wb") as f:
    f.write(packed.string)

  # If requested, decompress the image and measure performance.
  if args.verbose:
    x_hat = model.decompress(*tensors)

    # Cast to float in order to compute metrics.
    x = tf.cast(x, tf.float32)
    x_hat = tf.cast(x_hat, tf.float32)
    mse = tf.reduce_mean(tf.math.squared_difference(x, x_hat))
    psnr = tf.squeeze(tf.image.psnr(x, x_hat, 255))
    msssim = tf.squeeze(tf.image.ssim_multiscale(x, x_hat, 255))
    msssim_db = -10. * tf.math.log(1 - msssim) / tf.math.log(10.)

    # The actual bits per pixel including entropy coding overhead.
    num_pixels = tf.reduce_prod(tf.shape(x)[:-1])
    bpp = len(packed.string) * 8 / num_pixels

    return mse, psnr, msssim, msssim_db, bpp
    # print(f"Mean squared error: {mse:0.4f}")
    # print(f"PSNR (dB): {psnr:0.2f}")
    # print(f"Multiscale SSIM: {msssim:0.4f}")
    # print(f"Multiscale SSIM (dB): {msssim_db:0.2f}")
    # print(f"Bits per pixel: {bpp:0.4f}")
  
  return None


def decompress(args):
  """Decompresses an image."""
  # Load the model and determine the dtypes of tensors required to decompress.
  model = tf.keras.models.load_model(args.model_path)
  dtypes = [t.dtype for t in model.decompress.input_signature]

  # Read the shape information and compressed string from the binary file,
  # and decompress the image using the model.
  with open(args.input_file, "rb") as f:
    packed = tfc.PackedTensors(f.read())
  tensors = packed.unpack(dtypes)
  x_hat = model.decompress(*tensors)

  # Write reconstructed image out as a PNG file.
  write_png(args.output_file, x_hat)


# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------
# PARSE ARGS USER METHODS

def parse_args(argv):
  """Parses command line arguments."""
  parser = argparse_flags.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)

  # High-level options.
  parser.add_argument(
      "--verbose", "-V", action="store_true",
      help="Report progress and metrics when training or compressing.")
  parser.add_argument(
      "--model_path", default="bls2017",
      help="Path where to save/load the trained model.")
  subparsers = parser.add_subparsers(
      title="commands", dest="command",
      help="What to do: 'train' loads training data and trains (or continues "
           "to train) a new model. 'compress' reads an image file (lossless "
           "PNG format) and writes a compressed binary file. 'decompress' "
           "reads a binary file and reconstructs the image (in PNG format). "
           "input and output filenames need to be provided for the latter "
           "two options. Invoke '<command> -h' for more information.")

  # --- SAMU
  check_gpu_cmd = subparsers.add_parser(
      "check_gpu",
      formatter_class=argparse.ArgumentDefaultsHelpFormatter,
      description="Checks available GPUs.")
  # ---

  # 'train' subcommand.
  train_cmd = subparsers.add_parser(
      "train",
      formatter_class=argparse.ArgumentDefaultsHelpFormatter,
      description="Trains (or continues to train) a new model. Note that this "
                  "model trains on a continuous stream of patches drawn from "
                  "the training image dataset. An epoch is always defined as "
                  "the same number of batches given by --steps_per_epoch. "
                  "The purpose of validation is mostly to evaluate the "
                  "rate-distortion performance of the model using actual "
                  "quantization rather than the differentiable proxy loss. "
                  "Note that when using custom training images, the validation "
                  "set is simply a random sampling of patches from the "
                  "training set.")
  train_cmd.add_argument(
      "--lambda", type=float, default=0.01, dest="lmbda",
      help="Lambda for rate-distortion tradeoff.")
  train_cmd.add_argument(
      "--train_glob", type=str, default=None,
      help="Glob pattern identifying custom training data. This pattern must "
           "expand to a list of RGB images in PNG format. If unspecified, the "
           "CLIC dataset from TensorFlow Datasets is used.")
  train_cmd.add_argument(
      "--num_filters", type=int, default=128,
      help="Number of filters per layer.")
  train_cmd.add_argument(
      "--train_path", default="/tmp/train_bls2017",
      help="Path where to log training metrics for TensorBoard and back up "
           "intermediate model checkpoints.")
  train_cmd.add_argument(
      "--batchsize", type=int, default=8,
      help="Batch size for training and validation.")
  train_cmd.add_argument(
      "--patchsize", type=int, default=256,
      help="Size of image patches for training and validation.")
  train_cmd.add_argument(
      "--epochs", type=int, default=1000,
      help="Train up to this number of epochs. (One epoch is here defined as "
           "the number of steps given by --steps_per_epoch, not iterations "
           "over the full training dataset.)")
  train_cmd.add_argument(
      "--steps_per_epoch", type=int, default=1000,
      help="Perform validation and produce logs after this many batches.")
  train_cmd.add_argument(
      "--max_validation_steps", type=int, default=16,
      help="Maximum number of batches to use for validation. If -1, use one "
           "patch from each image in the training set.")
  train_cmd.add_argument(
      "--preprocess_threads", type=int, default=16,
      help="Number of CPU threads to use for parallel decoding of training "
           "images.")
  train_cmd.add_argument(
      "--precision_policy", type=str, default=None,
      help="Policy for `tf.keras.mixed_precision` training.")
  train_cmd.add_argument(
      "--check_numerics", action="store_true",
      help="Enable TF support for catching NaN and Inf in tensors.")

  # 'compress' subcommand.
  compress_cmd = subparsers.add_parser(
      "compress",
      formatter_class=argparse.ArgumentDefaultsHelpFormatter,
      description="Reads a PNG file, compresses it, and writes a TFCI file.")

  # 'decompress' subcommand.
  decompress_cmd = subparsers.add_parser(
      "decompress",
      formatter_class=argparse.ArgumentDefaultsHelpFormatter,
      description="Reads a TFCI file, reconstructs the image, and writes back "
                  "a PNG file.")

  # Arguments for both 'compress' and 'decompress'.
  for cmd, ext in ((compress_cmd, ".tfci"), (decompress_cmd, ".png")):
    cmd.add_argument(
        "input_file",
        help="Input filename.")
    cmd.add_argument(
        "output_file", nargs="?",
        help=f"Output filename (optional). If not provided, appends '{ext}' to "
             f"the input filename.")

  # Parse arguments.
  args = parser.parse_args(argv[1:])
  if args.command is None:
    parser.print_usage()
    sys.exit(2)
  return args


# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------
# MAIN

def main(args):
  
  # --- SAMU
  import tensorrt
  physical_devices = tf.config.list_physical_devices('GPU')
  print('Set memory limit: done')
  tf.config.experimental.set_memory_growth(physical_devices[0], True)
  # --- 

  # Invoke subcommand.
  if args.command == "check_gpu":
    check_gpu(args)
  if args.command == "train":
    train(args)
  elif args.command == "compress":
    if not args.output_file:
      args.output_file = args.input_file + ".tfci"
    compress(args)
  elif args.command == "decompress":
    if not args.output_file:
      args.output_file = args.input_file + ".png"
    decompress(args)


if __name__ == "__main__":
  app.run(main, flags_parser=parse_args)
