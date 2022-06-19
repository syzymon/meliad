import subprocess
import time

import metric_logging
from jobs.core import Job


class SetupIsabelleJob(Job):
    def execute(self):
        print(f"Hello Job at {time.time()}")

        subprocess.call("bash scripts/init_isabelle.sh", shell=True)

        metric_logging.log_scalar("time", 0, time.time())
