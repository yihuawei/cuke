from cset.ast import *
from cset.ast2ir import *

def triangle_counting():
    num_node = 10
    num_edges = 20
    max_degree = 20
    rowptr = Tensor('rowptr', (num_node+1,), dtype='int')
    colidx = Tensor('colidx', (num_edges,), dtype='int')

    edge_list =  Set(Tensor('edge_list', (num_edges, 2), dtype='int'))

    def inner_triangle_counting(edge):
        v0_nb =   Set(colidx[rowptr[edge[0]]:rowptr[edge[0]+1]])
        v1_nb =   Set(colidx[rowptr[edge[1]]:rowptr[edge[1]+1]])
        k = v0_nb.filter(BinarySearch(v1_nb))
        return k.num_elem()
    res = edge_list.apply(inner_triangle_counting).sum()
    code = codegen.cpu.print_cpp(res._gen_ir())
    print(code)