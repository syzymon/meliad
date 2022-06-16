# Copyright 2022 Google.
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

r"""Main program to train htransformer models.

"""
import os
import sys
from typing import Sequence

import gin
from absl import app
from absl import flags
from clu import platform
import jax
from transformers import set_seed

from transformer import launcher
import tensorflow.compat.v2 as tf


FLAGS = flags.FLAGS


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  launcher.parse_gin_configuration()

  # Hide any GPUs from TensorFlow. Otherwise TF might reserve memory and make
  # it unavailable to JAX.
  tf.config.experimental.set_visible_devices([], "GPU")

  set_seed(2137)
  # Set global seed for datasets.
  # tf.random.set_seed(1234)

  # Add a note so that we can tell which task is which JAX host.
  # (Depending on the platform task 0 is not guaranteed to be host 0)
  platform.work_unit().set_task_status(f"process_index: {jax.process_index()}, "
                                       f"process_count: {jax.process_count()}")
  platform.work_unit().create_artifact(platform.ArtifactType.DIRECTORY,
                                       FLAGS.workdir, "workdir")

  launcher.run_training_loop(testing=False)


if __name__ == "__main__":
  search_path = os.path.dirname(os.path.dirname(__file__))
  gin.add_config_file_search_path(search_path + "/transformer/configs")
  app.run(main)
