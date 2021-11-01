"""
Copyright (c) 2020-2021 WistLab.Co.Ltd
This file is a part of soft compression algorithm
"""

import os
import numpy as np
import ast
import PreProcess
import cv2


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
    book_return = dict(book.items())
    for key in list(book.keys()):
        key_same_total = get_keys(book, book[key])  # Get codewords with equal frequency
        key = np.array(key, np.float32)  # Convert into matrix
        for key_same in key_same_total:
            kernel = np.array(key_same, np.float32)
            if kernel.size == 1:
                continue
            if kernel.shape == key.shape and (kernel == key).all():  # The two keys are the same
                continue
            if kernel.shape[0] > key.shape[0] or kernel.shape[1] > key.shape[1]:
                continue
            dst = cv2.filter2D(key, -1, kernel, anchor=(0, 0), borderType=cv2.BORDER_CONSTANT)  # convolution
            same_location = np.argwhere(dst == np.sum(np.power(kernel, 2)))  # similar nodes
            for sl in same_location:
                # Determine whether the position can be coded in order to prevent [1 1] from being similar to [1 0]
                if key[sl[0]: sl[0] + kernel.shape[0], sl[1]: sl[1] + kernel.shape[1]].shape == kernel.shape \
                        and (key[sl[0]: sl[0] + kernel.shape[0], sl[1]: sl[1] + kernel.shape[1]] == kernel).all():
                    try:
                        del book_return[key_same]  # delete
                    except KeyError:
                        pass
    return book_return


def CodeProcessing(input_dir, output_dir):
    PreProcess.dir_check(output_dir, empty_flag=True)
    for frequency in os.listdir(input_dir):
        frequency_name = os.path.join(input_dir, frequency)  # file name
        with open(frequency_name, 'r') as f:  # read the file
            codebook = f.read()
            codebook = ast.literal_eval(codebook)  # convert into dictionary
        # Reassign the weight
        if frequency == 'frequency_detail.txt':
            # sort by frequency
            codebook = dict(sorted(codebook.items(), key=lambda item: item[1], reverse=True))
        elif frequency == 'frequency_rough.txt':
            # sort by frequency
            codebook = dict(sorted(codebook.items(), key=lambda item: item[1], reverse=True))
        elif frequency == 'frequency_shape.txt':
            initial_shape_kind = len(codebook)  # Number of code words before deletion
            codebook = codebook_delete(codebook)  # Delete repetitive and useless code words
            final_shape_kind = len(codebook)  # Number of code words after deletion
            for key in codebook.keys():
                mid_value = codebook[key]
                codebook[key] = mid_value
            # sort
            codebook = dict(sorted(codebook.items(), key=lambda item: np.sum(np.array(item[0]) != 0), reverse=True))
        # convert into list
        chars = list(codebook.keys())
        freqs = list(codebook.values())
        nodes = createNodes(freqs)  # Create all leaf nodes
        root = createHuffmanTree(nodes)  # Find the root node and determine the relationship between the nodes
        codes = huffmanEncoding(nodes, root)  # Find codebook
        codebook = dict(zip(chars, codes))  # Generate the latest codebook
        output_name = os.path.join(output_dir, 'codebook') + frequency[frequency.rfind('_'):]  # the name of codebook
        with open(output_name, 'w') as f:  # Write file
            f.write(str(codebook))
        print('The codebook has been saved as %s' % output_name)
