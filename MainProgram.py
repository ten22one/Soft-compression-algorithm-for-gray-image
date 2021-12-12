"""
Copyright (c) 2020-2021 WistLab.Co.Ltd
This file is a part of soft compression algorithm
"""

from PreProcess import PreProcess
from ShapeFinding import ShapeFinding
from CodeProcessing import CodeProcessing
from Encode import Encode
from Decode import Decode

# Choose dataset: drive PH2
dataset = 'drive'
# Preprocess the images
PreProcess(dataset)
# Search the set of shapes
mode = ShapeFinding(dataset)
# Process the codebooks
CodeProcessing()
# Encoder
Encode(dataset, mode)
# Decoder
Decode()
