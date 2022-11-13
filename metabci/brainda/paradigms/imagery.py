# -*- coding: utf-8 -*-
#
# Authors: Swolf <swolfforever@gmail.com>
# Date: 2020/6/01
# License: MIT License
"""
Motor Imagery Paradigm.

"""
from .base import BaseParadigm


class MotorImagery(BaseParadigm):
    """Basic motor imagery paradigm.

    """
    def is_valid(self, dataset):
        ret = True
        if dataset.paradigm != 'imagery':
            ret = False   
        return ret