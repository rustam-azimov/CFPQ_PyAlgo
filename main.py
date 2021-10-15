from src.utils.useful_paths import LOCAL_CFPQ_DATA
from cfpq_data import cfg_from_txt
from src.graph.graph import Graph

from src.problems.SinglePath.SinglePath import SinglePathProblem
from src.problems.SinglePath.algo.matrix_single_path_index import MatrixSingleAlgo

from src.problems.utils import ResultAlgo

import matplotlib.pyplot as plt
import numpy as np

import os

import sys
sys.setrecursionlimit(1000000)

first_letters = ['a', 'b', 'f', 'l', 'm']
second_letters = ['d', 'c', 'i', 'j', 'k']
third_letters = ['e', 'x', 'y', 'z']

STEP = 5
MAX_VALUE = 12
START = 5

def get_row(start, letter, end):
    return str(start) + ' ' + letter + ' ' + str(end)

def append_dimension(k, start, end, fletter, sletter, tletter):
    edges = []
    second_cycle_length = k/2
    first_cycle_length = second_cycle_length + 1
    current_vertex = end + 1
    middle = end + 1
    #first word in cycle
    edges.append(get_row(middle, fletter, start))
    current_vertex += 1
    edges.append(get_row(end, sletter, current_vertex))
    #starting second word
    start = current_vertex
    edges.append(get_row(current_vertex, fletter, middle))
    #sausages
    l = 1
    while l < first_cycle_length - 1:
        edges.append(get_row(current_vertex, sletter, current_vertex + 1))
        edges.append(get_row(current_vertex+1, fletter, current_vertex))
        current_vertex += 1
        l += 1
    edges.append(get_row(current_vertex, sletter, middle))
    #second cycle
    current_vertex += 1
    edges.append(get_row(middle, tletter, current_vertex))
    l = 1
    while l < second_cycle_length - 1:
        edges.append(get_row(current_vertex, tletter, current_vertex + 1))
        current_vertex += 1
        l += 1
    edges.append(get_row(current_vertex, tletter, middle))
    end = current_vertex
    return start, end, middle, edges

def gen_butterfly(k, fletter, sletter):
    second_cycle_length = k/2
    first_cycle_length = second_cycle_length + 1
    middle = int((k/2 + 1)/2)
    start = middle + 1 if middle + 1 < first_cycle_length else 0
    end = k - 1
    edges = []
    current_vertex = 0
    while current_vertex < first_cycle_length - 1:
        edges.append(str(current_vertex) + ' ' + fletter + ' ' + str(current_vertex + 1))
        current_vertex += 1
    edges.append(str(current_vertex) + ' ' + fletter + ' ' + "0")
    current_vertex += 1
    edges.append(str(middle) + ' ' + sletter + ' ' + str(current_vertex))
    while current_vertex < k - 1:
        edges.append(str(current_vertex) + ' ' + sletter + ' ' + str(current_vertex + 1))
        current_vertex += 1
    edges.append(str(current_vertex) + ' ' + sletter + ' ' + str(middle))
    return start, end, middle, edges


def gen_graph(dimension, k):
    base = 2*k
    d = 1
    edges = []
    while d <= dimension:
        first_letter_index = dimension - d
        second_letter_index = dimension - d
        third_letter_index = dimension - d
        if d == 1:
            start, end, middle, new_edges = gen_butterfly(base, first_letters[first_letter_index], second_letters[second_letter_index])
        else:
            start, end, middle, new_edges = append_dimension(base, start, end, first_letters[first_letter_index], second_letters[second_letter_index],
                                                             third_letters[third_letter_index])
        edges.extend(new_edges)
        d = d + 1

    return middle, edges


def run(dimension):
    results = []
    cases = [i for i in range(START, MAX_VALUE, STEP)]
    print(cases)
    i = 0
    for c in cases:
        results.append(run_one(dimension, c))
        i += 1
    visual(cases, results, dimension)
    print(results)

def run_one(dimension, p): # p = number of edges/2
    test_data_path = LOCAL_CFPQ_DATA
    dir = test_data_path.joinpath('Graphs/dim_' + str(dimension))
    if not os.path.exists(dir):
        os.makedirs(dir)
    grammar_filename = test_data_path.joinpath('Grammars/d' + str(dimension) + ".cfg")
    graph_filename = dir.joinpath('graph_' + str(p) + ".txt")
    middle, edges = gen_graph(dimension, p)
    with open(graph_filename, 'w') as f:
        # f.write('\n'.join(edges))
        for e in edges:
            f.write(e + '\n')
    singlepath_algo: SinglePathProblem = MatrixSingleAlgo()
    graph = Graph.from_txt(graph_filename)
    grammar = cfg_from_txt(grammar_filename)
    singlepath_algo.prepare(graph, grammar)
    result: ResultAlgo = singlepath_algo.solve()
    print("solving paths:", "number of edges =", p*dimension*2,"middle point=", middle)
    paths = singlepath_algo.getPath(middle, middle, "S")
    print("Success")
    print(p, paths)
    return paths

def visual(x, y, dimension):
    xpoints = np.array([2*dimension*i for i in x])
    ypoints = np.array(y)

    plt.plot(xpoints, ypoints, '-o', label = "results")

    function_ypoints = np.array([2*pow(i, dimension)*pow(i + 1, dimension) for i in x])
    plt.plot(xpoints, function_ypoints, label='y = p^' + str(dimension) + "*q^" + str(dimension))
    for i, txt in enumerate(ypoints):
        plt.annotate(txt, (xpoints[i], ypoints[i]))

    plt.title("Dimension = " + str(dimension))
    plt.xlabel("Number of Graph Vertices")
    plt.ylabel("Length of the Path")

    plt.legend()
    plt.show()


run_one(1, 105)