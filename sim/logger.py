import numpy as np

class SystemLogger:
    def __init__(self):
        # self._xs = np.empty((0, 4))
        self._xs = []
        self._us = []


class ControllerLogger:
    def __init__(self):
        self._xtrajs = []
        self._utrajs = []


class LocalPlannerLogger:
    def __init__(self):
        self._trajs = []


class GlobalPlannerLogger:
    def __init__(self):
        self._paths = []