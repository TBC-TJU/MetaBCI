# -*- coding: utf-8 -*-
#
# Authors: Jie Mei <chmeijie@gmail.com>
# Date: 2023-10-3
# License: MIT License
"""
Movement intention paradigms.

"""
from .base import BaseParadigm


class MovementIntention(BaseParadigm):
    """Basic movement intention paradigm."""

    def is_valid(self, dataset):
        ret = True
        if dataset.paradigm != "movement_intention":
            ret = False
        return ret
