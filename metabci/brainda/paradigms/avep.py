# -*- coding: utf-8 -*-
#
# Authors: Jie Mei <chmeijie@gmail.com>
# Date: 2023/1/11
# License: MIT License
"""
aVEP Paradigm.
"""
from .base import BaseTimeEncodingParadigm


class aVEP(BaseTimeEncodingParadigm):
    def is_valid(self, dataset):
        ret = True
        if dataset.paradigm != "aVEP":
            ret = False
        return ret
