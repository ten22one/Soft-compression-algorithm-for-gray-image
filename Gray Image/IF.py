"""
Copyright (c) 2020-2021 WistLab.Co.Ltd
This file is a part of soft compression algorithm
"""
import math

def entropy(pd):
    Entropy = 0
    for p in pd:
        if p == 0 or p == 1:
            pass
        else:
            Entropy = Entropy - p * math.log2(p)
    return Entropy
