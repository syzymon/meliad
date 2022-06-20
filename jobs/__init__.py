import gin

from jobs import hello_job
from jobs.train_memorizing_job import TrainMemorizingJob


def configure_job(goal_generator_class):
    return gin.external_configurable(goal_generator_class, module="jobs")


JobHello = configure_job(hello_job.HelloJob)
JobTrainMemorizing = configure_job(TrainMemorizingJob)
