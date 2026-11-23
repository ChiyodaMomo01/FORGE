"""Spawn target: importing the learner transport does not import physics."""


def serve(*args):
    from .oracle import serve as numerical_service

    return numerical_service(*args)
