"""
Copyright (c) 2020-2021 WistLab.Co.Ltd
This file is a part of soft compression algorithm
"""

import os
import numpy as np
import ast
import PreProcess
import math


# Define the node
class Node:
    def __init__(self, freq):
        self.left = None
        self.right = None
        self.father = None
        self.freq = freq

    def isLeft(self):
        return self.father.left == self  # Judge whether father left is itself


# Create lead nodes
def createNodes(freqs):
    return [Node(freq) for freq in freqs]  # define leaf node


# Create Huffman tree
def createHuffmanTree(nodes):
    num = 1  # number
    queue = nodes[:]
    while len(queue) > 1:
        queue.sort(key=lambda item: item.freq)  # sort by frequency
        node_left = queue.pop(0)  # Delete the node with the smallest frequency and define it as the left node
        node_right = queue.pop(0)  # Delete the node with the second smallest frequency and define it as the left node
        node_father = Node(node_left.freq + node_right.freq)  # Merge the left and right nodes into a new parent node
        num = num + 1
        node_father.left = node_left  # The left node of the parent node is the node with the smallest frequency
        node_father.right = node_right  # The right node of the parent node is the node with the second smallest frequency
        node_left.father = node_father  # The parent node of the left node is the parent node just defined
        node_right.father = node_father  # The parent node of the right node is the parent node just defined
        queue.append(node_father)  # Add a parent node to the new queue
    queue[0].father = None  # The root node has no parent
    return queue[0]  # Return to the root node


# Huffman coding
def huffmanEncoding(nodes, root):
    codes = [''] * len(nodes)  # generate the codebook
    for i in range(len(nodes)):
        node_tmp = nodes[i]
        while node_tmp != root:  # Determine whether it is the root node
            if node_tmp.isLeft():  # Determine whether it is a child node of its parent node
                codes[i] = '0' + codes[i]
            else:
                codes[i] = '1' + codes[i]
            node_tmp = node_tmp.father
    return codes  # return the codebook


def get_keys(d, value):
    """
    Get all keys in the dictionary that are equal to a certain value
    :param d: dictionary
    :param value:value
    :return:key
    """
    return [k for k, v in d.items() if v == value]


def codebook_delete(book):
    """
    Delete repetitive and useless codewords
    :param book: codebook
    :return: new codebook
    """
    book = dict(sorted(book.items(), key=lambda item: item[1], reverse=True))  # sort
    book_one = {}
    num = 0
    for key in list(book.keys()):
        kernel = np.array(key, np.float32)
        if kernel.size == 1:
            book_one[key] = book[key]
        else:
            # retain 3000 shapes
            if num <= 3000:
                book_one[key] = book[key]
                num = num + 1
    book_return = dict(book_one.items())  # codebook
    return book_return


def CodeProcessing():
    input_dir = 'frequency'  # input folder
    output_dir = 'codebook'  # output folder
    PreProcess.dir_check(output_dir, empty_flag=True)  # empty the output folder
    print("*" * 150, '\n')
    for frequency in os.listdir(input_dir):
        frequency_name = os.path.join(input_dir, frequency)  # file name
        with open(frequency_name, 'r') as f:  # read the file
            codebook = f.read()  # read
            codebook = ast.literal_eval(codebook)  # convert into dictionary
        print('Reading %s，' % frequency_name)
        # Reassign the weight
        frequency_class = frequency[0:frequency.rfind('_')]
        if frequency_class == 'frequency_detail':
            # sort by frequency
            codebook = dict(sorted(codebook.items(), key=lambda item: item[1], reverse=True))
            # convert into list
            chars = list(codebook.keys())
            freqs = list(codebook.values())
            nodes = createNodes(freqs)  # Create all leaf nodes
            root = createHuffmanTree(nodes)  # Find the root node and determine the relationship between the nodes
            codes = huffmanEncoding(nodes, root)  # Find codebook
            codebook = dict(zip(chars, codes))  # Generate the codebook
        elif frequency_class == 'frequency_rough':
            # sort by frequency
            codebook = dict(sorted(codebook.items(), key=lambda item: item[1], reverse=True))
            value_list = list(codebook.values())
            last_value = value_list[len(value_list) - 1]
            mean_value = np.mean(list(codebook.values()))
            for i in list(codebook.keys()):
                codebook[i] = math.ceil(codebook[i] / mean_value)
                codebook[i] = format(codebook[i], 'b')
        elif frequency_class == 'frequency_shape':
            codebook = codebook_delete(codebook)  # Eliminate repetitive and useless codewords
            for key in codebook.keys():
                mid_value = codebook[key] * np.sum(key != 0)
                codebook[key] = mid_value
            codebook = codebook_delete(codebook)  # Eliminate repetitive and useless codewords
            # sort
            codebook = dict(sorted(codebook.items(), key=lambda item: np.sum(np.array(item[0]) != 0), reverse=True))
            # convert into list
            chars = list(codebook.keys())
            freqs = list(codebook.values())
            nodes = createNodes(freqs)  # Create all leaf nodes
            root = createHuffmanTree(nodes)  # Find the root node and determine the relationship between the nodes
            codes = huffmanEncoding(nodes, root)  # Find codebook
            codebook = dict(zip(chars, codes))  # Generate the codebook
        output_name = os.path.join(output_dir, 'codebook') + frequency[frequency.find('_'):]  # the name of codebook
        with open(output_name, 'w') as f:  # write
            f.write(str(codebook))
        print('The codebook has been saved as %s' % output_name)
