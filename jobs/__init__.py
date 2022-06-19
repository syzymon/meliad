import gin

from jobs import hello_job


def configure_job(goal_generator_class):
    return gin.external_configurable(goal_generator_class, module="jobs")


JobHello = configure_job(hello_job.HelloJob)
