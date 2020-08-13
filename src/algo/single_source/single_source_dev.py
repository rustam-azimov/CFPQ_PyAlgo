from collections import Iterable
from typing import Tuple

from pygraphblas import Matrix, BOOL

from src.algo.single_source.single_source import SingleSourceSolver, SingleSourceIndex, update_sources, update_sources_opt
from src.grammar.cnf_grammar import CnfGrammar
from src.graph.label_graph import LabelGraph


class Response:
    def __init__(self, iters):
        self.iters = iters


class SingleSourceAlgoBruteDev(SingleSourceSolver):
    def __init__(self, graph: LabelGraph, grammar: CnfGrammar):
        super().__init__(graph, grammar)

    def solve(self, sources_vertices: Iterable):
        # Creating new index per solve call
        index = SingleSourceIndex(self.graph, self.grammar)

        # Initialize simple rules
        index.init_simple_rules()

        # Initialize sources and nonterms nnz
        # nnz: (l, r1, r2) in complex rules -> (nnz(l), nnz(r1), nnz(r2))
        nnz = {}
        for l, r1, r2 in self.grammar.complex_rules:
            nnz[(l, r1, r2)] = (0, 0, 0)

        # Initialize source matrices masks
        for v in sources_vertices:
            index.sources[index.grammar.start_nonterm][v, v] = True

        # Create temporary matrix
        tmp = Matrix.sparse(BOOL, index.graph.matrices_size, index.graph.matrices_size)

        # Algo's body
        iter_count = 0
        changed = True
        while changed:
            iter_count += 1
            changed = False

            # Number of instances before operation
            # old_nnz_nonterms = {nonterm: index.nonterms[nonterm].nvals for nonterm in index.grammar.nonterms}
            # old_nnz_sources = {nonterm: index.sources[nonterm].nvals for nonterm in index.grammar.nonterms}

            # Iterate through all complex rules
            for l, r1, r2 in index.grammar.complex_rules:
                new_nnz = index.sources[l].nvals, index.nonterms[r1].nvals, index.nonterms[r2].nvals
                if nnz[(l, r1, r2)] != new_nnz:
                    # 1) r1_src += {(j, j) : (i, j) \in l_src}
                    update_sources(index.sources[l], index.sources[r1])

                    # 2) tmp = l_src * r1
                    tmp = index.sources[l] @ index.nonterms[r1]

                    # 3) r2_src += {(j, j) : (i, j) \in tmp}
                    update_sources(tmp, index.sources[r2])

                    # 4) l += tmp * r2
                    index.nonterms[l] += tmp @ index.nonterms[r2]

                    # update nnz
                    nnz[(l, r1, r2)] = index.sources[l].nvals, index.nonterms[r1].nvals, index.nonterms[r2].nvals
                    changed = True

        return index.nonterms[index.grammar.start_nonterm], Response(iter_count)


class SingleSourceAlgoSmartDev(SingleSourceSolver):
    def __init__(self, graph: LabelGraph, grammar: CnfGrammar):
        super().__init__(graph, grammar)
        self.index = SingleSourceIndex(graph, grammar)
        self.index.init_simple_rules()

    def solve(self, sources_vertices: Iterable) -> Tuple[Matrix, Response]:
        # Initialize source matrices masks
        for v in sources_vertices:
            self.index.sources[self.index.grammar.start_nonterm][v, v] = True

        # Initialize sources and nonterms nnz
        # nnz: (l, r1, r2) in complex rules -> (nnz(l), nnz(r1), nnz(r2))
        nnz = {}
        for l, r1, r2 in self.grammar.complex_rules:
            nnz[(l, r1, r2)] = (0, 0, 0)

        # Create temporary matrix
        tmp = Matrix.sparse(BOOL, self.index.graph.matrices_size, self.index.graph.matrices_size)

        # Algo's body
        changed = True
        iter_count = 0
        while changed:
            iter_count += 1
            changed = False

            # Iterate through all complex rules
            for l, r1, r2 in self.index.grammar.complex_rules:
                new_nnz = self.index.sources[l].nvals, self.index.nonterms[r1].nvals, self.index.nonterms[r2].nvals
                if nnz[(l, r1, r2)] != new_nnz:
                    # l -> r1 r2 ==> l += (l_src * r1) * r2 =>

                    # 1) r1_src += {(j, j) : (i, j) \in l_src}
                    update_sources(self.index.sources[l], self.index.sources[r1])

                    # 2) tmp = l_src * r1
                    tmp = self.index.sources[l] @ self.index.nonterms[r1]

                    # 3) r2_src += {(j, j) : (i, j) \in tmp}
                    update_sources(tmp, self.index.sources[r2])

                    # 4) l += tmp * r2
                    self.index.nonterms[l] += tmp @ self.index.nonterms[r2]

                    # update nnz
                    nnz[(l, r1, r2)] = self.index.sources[l].nvals, self.index.nonterms[r1].nvals, self.index.nonterms[r2].nvals
                    changed = True

        return self.index.nonterms[self.index.grammar.start_nonterm], Response(iter_count)


class SingleSourceAlgoOptDev(SingleSourceSolver):
    def __init__(self, graph: LabelGraph, grammar: CnfGrammar):
        super().__init__(graph, grammar)
        self.index = SingleSourceIndex(graph, grammar)
        self.index.init_simple_rules()

    def solve(self, sources_vertices: Iterable) -> Tuple[Matrix, Response]:
        new_sources = LabelGraph(self.graph.matrices_size)

        # Initialize sources and nonterms nnz
        # nnz: (l, r1, r2) in complex rules -> (nnz(new[l]), nnz(index[r1]), nnz(index[r2]))
        nnz = {}
        for l, r1, r2 in self.grammar.complex_rules:
            nnz[(l, r1, r2)] = (0, 0, 0)

        # Initialize source matrices masks
        for i in sources_vertices:
            if (i, i) not in self.index.sources[self.grammar.start_nonterm]:
                new_sources[self.grammar.start_nonterm][i, i] = True

        # Create temporary matrix
        tmp = Matrix.sparse(BOOL, self.graph.matrices_size, self.graph.matrices_size)

        # Algo's body
        changed = True
        iter_count = 0
        while changed:
            iter_count += 1
            changed = False

            # Iterate through all complex rules
            for l, r1, r2 in self.index.grammar.complex_rules:
                # l -> r1 r2 ==> index[l] += (new[l_src] * index[r1]) * index[r2]

                new_nnz = new_sources[l].nvals, self.index.nonterms[r1].nvals, self.index.nonterms[r2].nvals
                if nnz[(l, r1, r2)] != new_nnz:
                    # 1) new[r1_src] += {(j, j) : (j, j) in new[l_src] and not in index[r1_src]}
                    for i, _, _ in new_sources[l]:
                        if (i, i) not in self.index.sources[r1]:
                            new_sources[r1][i, i] = True

                    # 2) tmp = new[l_src] * index[r1]
                    tmp = new_sources[l] @ self.index.nonterms[r1]

                    # 3) new[r2_src] += {(j, j) : (i, j) in tmp and not in index[r2_src]}
                    update_sources_opt(tmp, self.index.sources[r2], new_sources[r2])

                    # 4) index[l] += tmp * index[r2]
                    self.index.nonterms[l] += tmp @ self.index.nonterms[r2]

                    # update nnz
                    nnz[(l, r1, r2)] = new_sources[l].nvals, self.index.nonterms[r1].nvals, self.index.nonterms[r2].nvals
                    changed = True

        return self.index.nonterms[self.index.grammar.start_nonterm], Response(iter_count)
