# -*- coding: utf-8 -*-
#
# Authors: Swolf <swolfforever@gmail.com>
# Date: 2020/6/01
# License: MIT License
"""
P300 Paradigm.
"""
from .base import BaseParadigm


class P300(BaseParadigm):
    def is_valid(self, dataset):
        ret = True
        if dataset.paradigm != 'p300':
            ret = False             
        return ret