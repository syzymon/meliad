import os
import time
from typing import Sequence

import gin
import jax
from absl import app
from absl import flags
from transformers import set_seed

import metric_logging
from jobs.core import Job
from transformer import launcher
import tensorflow.compat.v2 as tf
from clu import platform

FLAGS = flags.FLAGS
flags.DEFINE_bool("mrunner", False,
                  "Add mrunner spec to gin-config overrides and Neptune to loggers.")
flags.DEFINE_string("config", "", "Gin config file to use.")
flags.DEFINE_string("config_file", "", "config file to use.")


def main(argv: Sequence[str]) -> None:
    if len(argv) > 1:
        raise app.UsageError("Too many command-line arguments.")

    launcher.parse_gin_configuration()
    metric_logging.log_property("gin_config", gin.config_str())
    metric_logging.log_property("gin_parameters", gin.config._CONFIG)

    # Hide any GPUs from TensorFlow. Otherwise TF might reserve memory and make
    # it unavailable to JAX.
    tf.config.experimental.set_visible_devices([], "GPU")

    set_seed(2137)
    # Set global seed for datasets.
    # tf.random.set_seed(1234)

    # Add a note so that we can tell which task is which JAX host.
    # (Depending on the platform task 0 is not guaranteed to be host 0)
    platform.work_unit().set_task_status(
        f"process_index: {jax.process_index()}, "
        f"process_count: {jax.process_count()}"
    )
    platform.work_unit().create_artifact(
        platform.ArtifactType.DIRECTORY, FLAGS.workdir, "workdir"
    )

    launcher.run_training_loop(testing=False)


class TrainMemorizingJob(Job):
    def execute(self):
        print(f"Hello Job at {time.time()}")
        metric_logging.log_scalar("time", 0, time.time())

        search_path = os.path.dirname(os.path.dirname(__file__))
        print('ABS PATH: ', search_path)
        gin.add_config_file_search_path(search_path + "/transformer/configs")

        app.run(main)
