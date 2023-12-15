import os
from itertools import permutations 
import numpy as np
import torch

import run
import helpers
import codegen

from cset.ast import *
# from cset.ast2ir import *
# from cset.opt.codemotion import *
# from cset.opt.fusion import *
from cset.opt.parallelize import *


def symmetry_breaking(pmtx):
    num_node = len(pmtx)
    perms = list(permutations(list(range(num_node))))

    valid_perms = []
    for perm in perms:
        mapped_adj = [None for x in range(num_node)]
        for i in range(num_node):
            tp = set()
            for j in range(num_node):
                if pmtx[i][j]==0:
                    continue
                tp.add(perm[j])
            mapped_adj[perm[i]] = tp

        valid = True
        for i in range(num_node):
            equal = True
            count = 0
            for j in range(num_node):
                if pmtx[i][j]==1:
                    count+=1
                    if j not in mapped_adj[i]:
                        equal=False
            
            if not equal or count!=len(mapped_adj[i]):
                valid = False
                break

        if valid==True:
            valid_perms.append(perm)

    partial_orders = [[0 for y in range(num_node)] for x in range(num_node)]
    for i in range(num_node):
        stabilized_aut = []
        for perm in valid_perms:
            if perm[i]==i:
                stabilized_aut.append(perm)
            else:
                partial_orders[perm[i]][i]=1

        valid_perms = stabilized_aut
    

    res = [-1 for x in range(num_node)]
    for i in range(num_node):
        largest_idx = -1
        for j in range(num_node):
            if partial_orders[i][j]==1 and j>largest_idx:
                largest_idx = j
        
        res[i] = largest_idx
    print("partial orders:")
    print(res)
    return res

def read_pattern_file(filename):
    pmtx = None
    num_node=0
    with open(filename) as p:
        for line in p:
            if line[0]=='v':
                num_node+=1
        p.seek(0, 0)

        pmtx = [[0 for y in range(num_node)] for x in range(num_node)]
        for line in p: 
            if line[0]=='e':
                v0 = int(line[2])
                v1 = int(line[4])
                pmtx[v0][v1] = 1
                pmtx[v1][v0] = 1
        print("pmtx:")
        print(pmtx)
        return pmtx

def node_degree(pmtx, idx):
    res = 0
    row = pmtx[idx]
    for i in row:
        if i==1:
            res+=1
    return res

def p1_decomine():

    num_node =  Var(name='num_node', dtype='int', is_arg=True)
    num_edge =  Var(name='num_edge', dtype='int', is_arg=True)
    num_jobs =  Var(name='num_jobs', dtype='int', is_arg=True)
    
    rowptr = Tensor('rowptr', (num_node+1,), dtype='int')
    colidx = Tensor('colidx', (num_edge,), dtype='int')
    edge_list =  Set(Tensor('edge_list', (num_jobs, 2), dtype='int'))
    count = Var(name='count', dtype='int', is_arg=False)

    code_style = FunctionCall

    class inner_subgraph_matching:

        def __init__(self, level, *path):
             self.level = level
             self.path = list(*path)

        def __call__(self, item):
            if self.level==2:
                v0 = item[0]
                v1 = item[1]
                v0_nb =  Set(colidx[rowptr[v0]:rowptr[v0+1]])
                v1_nb =  Set(colidx[rowptr[v1]:rowptr[v1+1]])

                candidate_set = v0_nb.intersection(v1_nb, code_style)
                return candidate_set.apply(inner_subgraph_matching(self.level+1, [v0, v1]))
            
            elif self.level==3:
                v0 = self.path[0]
                v1 = self.path[1]
                v2 = item
                v0_nb =  Set(colidx[rowptr[v0]:rowptr[v0+1]])
                v1_nb =  Set(colidx[rowptr[v1]:rowptr[v1+1]])
                v2_nb =  Set(colidx[rowptr[v2]:rowptr[v2+1]])

                candidate_v3 = v0_nb.intersection(v1_nb, code_style)
                candidate_v4 = v0_nb.intersection(v2_nb, code_style)
                candidate_v5 = v1_nb.intersection(v2_nb, code_style)

                candidate_shrink = candidate_v3.intersection(v2_nb, code_style)

                return candidate_v3.num_element() * candidate_v4.num_element() * candidate_v5.num_element() - candidate_shrink.num_element()

                
    
    def init_vals():
        return setval(count, 0)
    
    edge_list = edge_list.filter(PartialEdge)
    # edge_list = edge_list.filter(Edgefilter(rowptr, node_degree(pmtx, 0)-1))

    res = edge_list.apply(inner_subgraph_matching(2), init=init_vals).retval( Var(name='return_val', dtype='int', is_arg=False))
    
    #res.name =  'p'+str(pattern_size)+ '_' + os.path.basename(pattern_file_name).split('.')[0]
    res._gen_ir()
    # parallelize(res)
    code = codegen.cpu.print_cpp(res)
    print(code)

    # torch_rowptr, torch_colidx, torch_edge_list, num_node, num_edges, num_jobs = read_graph(False)
    # d = run.cpu.compile_and_run(code, num_edges, torch_edge_list, num_node, torch_rowptr, num_edges, torch_colidx)
    # print(d)


def p1_edge_induced():

    num_node =  Var(name='num_node', dtype='int', is_arg=True)
    num_edge =  Var(name='num_edge', dtype='int', is_arg=True)
    num_jobs =  Var(name='num_jobs', dtype='int', is_arg=True)
    
    rowptr = Tensor('rowptr', (num_node+1,), dtype='int')
    colidx = Tensor('colidx', (num_edge,), dtype='int')
    edge_list =  Set(Tensor('edge_list', (num_jobs, 2), dtype='int'))
    count = Var(name='count', dtype='int', is_arg=False)

    code_style = FunctionCall

    class inner_subgraph_matching:

        def __init__(self, level, *path):
             self.level = level
             self.path = list(*path)

        def __call__(self, item):
            if self.level == 6:
                return  Set(count).increment(1)

            if self.level==2:
                v0 = item[0]
                v1 = item[1]
                v0_nb =  Set(colidx[rowptr[v0]:rowptr[v0+1]])
                v1_nb =  Set(colidx[rowptr[v1]:rowptr[v1+1]])

                candidate_set = v0_nb.intersection(v1_nb, code_style)
                return candidate_set.apply(inner_subgraph_matching(self.level+1, [v0, v1]))
            
            elif self.level==3:
                v0 = self.path[0]
                v1 = self.path[1]
                v0_nb =  Set(colidx[rowptr[v0]:rowptr[v0+1]])
                v1_nb =  Set(colidx[rowptr[v1]:rowptr[v1+1]])

                candidate_v3 = v0_nb.intersection(v1_nb, code_style)
                return candidate_v3.apply(inner_subgraph_matching(self.level+1,  self.path + [item]))
            
            elif self.level==4:
                v0 = self.path[0]
                v2 = self.path[2]
                v0_nb =  Set(colidx[rowptr[v0]:rowptr[v0+1]])
                v2_nb =  Set(colidx[rowptr[v2]:rowptr[v2+1]])

                candidate_v4 = v0_nb.intersection(v2_nb, code_style)
                return candidate_v4.apply(inner_subgraph_matching(self.level+1, self.path + [item]))
            
            elif self.level==5:
                v1 = self.path[1]
                v2 = self.path[2]
                v1_nb =  Set(colidx[rowptr[v1]:rowptr[v1+1]])
                v2_nb =  Set(colidx[rowptr[v2]:rowptr[v2+1]])

                candidate_v5 = v1_nb.intersection(v2_nb, code_style)
                return candidate_v5.apply(inner_subgraph_matching(self.level+1, self.path + [item]))

                
    
    def init_vals():
        return setval(count, 0)
    
    edge_list = edge_list.filter(PartialEdge)
    # edge_list = edge_list.filter(Edgefilter(rowptr, node_degree(pmtx, 0)-1))

    res = edge_list.apply(inner_subgraph_matching(2), init=init_vals).retval( Var(name='return_val', dtype='int', is_arg=False))
    
    #res.name =  'p'+str(pattern_size)+ '_' + os.path.basename(pattern_file_name).split('.')[0]
    res._gen_ir()
    # parallelize(res)
    code = codegen.cpu.print_cpp(res)
    print(code)

    # torch_rowptr, torch_colidx, torch_edge_list, num_node, num_edges, num_jobs = read_graph(False)
    # d = run.cpu.compile_and_run(code, num_edges, torch_edge_list, num_node, torch_rowptr, num_edges, torch_colidx)
    # print(d)


def p2_decomine():
    num_node =  Var(name='num_node', dtype='int', is_arg=True)
    num_edge =  Var(name='num_edge', dtype='int', is_arg=True)
    num_jobs =  Var(name='num_jobs', dtype='int', is_arg=True)
    
    rowptr = Tensor('rowptr', (num_node+1,), dtype='int')
    colidx = Tensor('colidx', (num_edge,), dtype='int')
    edge_list =  Set(Tensor('edge_list', (num_jobs, 2), dtype='int'))
    count = Var(name='count', dtype='int', is_arg=False)

    code_style = FunctionCall

    class inner_subgraph_matching:

        def __init__(self, level, *path):
             self.level = level
             self.path = list(*path)

        def __call__(self, item):
            if self.level==2:
                v0 = item[0]
                v1 = item[1]
                v0_nb =  Set(colidx[rowptr[v0]:rowptr[v0+1]])
                v1_nb =  Set(colidx[rowptr[v1]:rowptr[v1+1]])

                candidate_set = v0_nb.intersection(v1_nb)
                return candidate_set.apply(inner_subgraph_matching(self.level+1, [v0, v1]))
            
            elif self.level==3:
                v0 = self.path[0]
                v1 = self.path[1]
                v2 = item
                v0_nb =  Set(colidx[rowptr[v0]:rowptr[v0+1]])
                v1_nb =  Set(colidx[rowptr[v1]:rowptr[v1+1]])
                v2_nb =  Set(colidx[rowptr[v2]:rowptr[v2+1]])

                candidate_v5 = v1_nb.intersection(v2_nb)

                candidate_v3v4 = candidate_v5.intersection(v0_nb)

                v5_nelem = candidate_v5.num_element()
                v3v4_nelem = candidate_v3v4.num_element()

                return v5_nelem * v3v4_nelem * v3v4_nelem - v3v4_nelem
       
    
    def init_vals():
        return setval(count, 0)
    
    # edge_list = edge_list.filter(PartialEdge)
    # edge_list = edge_list.filter(Edgefilter(rowptr, node_degree(pmtx, 0)-1))

    res = edge_list.apply(inner_subgraph_matching(2), init=init_vals).retval( Var(name='return_val', dtype='int', is_arg=False))
    
    #res.name =  'p'+str(pattern_size)+ '_' + os.path.basename(pattern_file_name).split('.')[0]
    res._gen_ir()
    # parallelize(res)
    code = codegen.cpu.print_cpp(res)
    print(code)

    # torch_rowptr, torch_colidx, torch_edge_list, num_node, num_edges, num_jobs = read_graph(False)
    # d = run.cpu.compile_and_run(code, num_edges, torch_edge_list, num_node, torch_rowptr, num_edges, torch_colidx)
    # print(d)



def p2_edge_induced():

    num_node =  Var(name='num_node', dtype='int', is_arg=True)
    num_edge =  Var(name='num_edge', dtype='int', is_arg=True)
    num_jobs =  Var(name='num_jobs', dtype='int', is_arg=True)
    
    rowptr = Tensor('rowptr', (num_node+1,), dtype='int')
    colidx = Tensor('colidx', (num_edge,), dtype='int')
    edge_list =  Set(Tensor('edge_list', (num_jobs, 2), dtype='int'))
    count = Var(name='count', dtype='int', is_arg=False)

    # code_style = FunctionCall

    class inner_subgraph_matching:

        def __init__(self, level, *path):
             self.level = level
             self.path = list(*path)

        def __call__(self, item):
            if self.level == 6:
                return  Set(count).increment(1)

            if self.level==2:
                v0 = item[0]
                v1 = item[1]
                v0_nb =  Set(colidx[rowptr[v0]:rowptr[v0+1]])
                v1_nb =  Set(colidx[rowptr[v1]:rowptr[v1+1]])

                candidate_set = v0_nb.intersection(v1_nb)
                return candidate_set.apply(inner_subgraph_matching(self.level+1, [v0, v1]))
            
            elif self.level==3:
                v0 = self.path[0]
                v1 = self.path[1]
                v2 = item
                v0_nb = Set(colidx[rowptr[v0]:rowptr[v0+1]])
                v1_nb = Set(colidx[rowptr[v1]:rowptr[v1+1]])
                v2_nb = Set(colidx[rowptr[v2]:rowptr[v2+1]])

                candidate_v3 = v0_nb.intersection(v1_nb).intersection(v2_nb)
                return candidate_v3.apply(inner_subgraph_matching(self.level+1,  self.path + [item]))
            
            elif self.level==4:
                v0 = self.path[0]
                v1 = self.path[1]
                v2 = self.path[2]
                v0_nb =  Set(colidx[rowptr[v0]:rowptr[v0+1]])
                v1_nb = Set(colidx[rowptr[v1]:rowptr[v1+1]])
                v2_nb =  Set(colidx[rowptr[v2]:rowptr[v2+1]])

                candidate_v4 = v0_nb.intersection(v1_nb).intersection(v2_nb)
                return candidate_v4.apply(inner_subgraph_matching(self.level+1, self.path + [item]))
            
            elif self.level==5:
                v1 = self.path[1]
                v2 = self.path[2]
                v1_nb = Set(colidx[rowptr[v1]:rowptr[v1+1]])
                v2_nb =  Set(colidx[rowptr[v2]:rowptr[v2+1]])

                candidate_v5 = v1_nb.intersection(v2_nb)
                return candidate_v5.apply(inner_subgraph_matching(self.level+1, self.path + [item]))

                
    
    def init_vals():
        return setval(count, 0)
    
    # edge_list = edge_list.filter(PartialEdge)
    # edge_list = edge_list.filter(Edgefilter(rowptr, node_degree(pmtx, 0)-1))

    res = edge_list.apply(inner_subgraph_matching(2), init=init_vals).retval( Var(name='return_val', dtype='int', is_arg=False))
    
    #res.name =  'p'+str(pattern_size)+ '_' + os.path.basename(pattern_file_name).split('.')[0]
    res._gen_ir()
    # parallelize(res)
    code = codegen.cpu.print_cpp(res)
    print(code)


def p3_edge_induced():
    num_node =  Var(name='num_node', dtype='int', is_arg=True)
    num_edge =  Var(name='num_edge', dtype='int', is_arg=True)
    num_jobs =  Var(name='num_jobs', dtype='int', is_arg=True)
    
    rowptr = Tensor('rowptr', (num_node+1,), dtype='int')
    colidx = Tensor('colidx', (num_edge,), dtype='int')
    edge_list =  Set(Tensor('edge_list', (num_jobs, 2), dtype='int'))
    count = Var(name='count', dtype='int', is_arg=False)

    # code_style = FunctionCall

    class inner_subgraph_matching:

        def __init__(self, level, *path):
             self.level = level
             self.path = list(*path)

        def __call__(self, item):
            if self.level == 6:
                return  Set(count).increment(1)

            if self.level==2:
                v0 = item[0]
                v1 = item[1]
                v0_nb =  Set(colidx[rowptr[v0]:rowptr[v0+1]])
                v1_nb =  Set(colidx[rowptr[v1]:rowptr[v1+1]])

                candidate_set = v0_nb.intersection(v1_nb)
                return candidate_set.apply(inner_subgraph_matching(self.level+1, [v0, v1]))
            
            elif self.level==3:
                v0 = self.path[0]
                v1 = self.path[1]
                v2 = item
                v0_nb = Set(colidx[rowptr[v0]:rowptr[v0+1]])
                v1_nb = Set(colidx[rowptr[v1]:rowptr[v1+1]])

                candidate_v3 = v0_nb.intersection(v1_nb)
                return candidate_v3.apply(inner_subgraph_matching(self.level+1,  self.path + [item]))
            
            elif self.level==4:
                v0 = self.path[0]
                v0_nb =  Set(colidx[rowptr[v0]:rowptr[v0+1]])

                candidate_v4 = v0_nb
                return candidate_v4.apply(inner_subgraph_matching(self.level+1, self.path + [item]))
            
            elif self.level==5:
                v1 = self.path[1]
                v4 = item
                v1_nb = Set(colidx[rowptr[v1]:rowptr[v1+1]])
                v4_nb =  Set(colidx[rowptr[v4]:rowptr[v4+1]])

                candidate_v5 = v1_nb.intersection(v4_nb)
                return candidate_v5.apply(inner_subgraph_matching(self.level+1, self.path + [item]))

                
    
    def init_vals():
        return setval(count, 0)
    
    # edge_list = edge_list.filter(PartialEdge)
    # edge_list = edge_list.filter(Edgefilter(rowptr, node_degree(pmtx, 0)-1))

    res = edge_list.apply(inner_subgraph_matching(2), init=init_vals).retval( Var(name='return_val', dtype='int', is_arg=False))
    
    #res.name =  'p'+str(pattern_size)+ '_' + os.path.basename(pattern_file_name).split('.')[0]
    res._gen_ir()
    # parallelize(res)
    code = codegen.cpu.print_cpp(res)
    print(code)

def p3_edge_induced2():
    num_node =  Var(name='num_node', dtype='int', is_arg=True)
    num_edge =  Var(name='num_edge', dtype='int', is_arg=True)
    num_jobs =  Var(name='num_jobs', dtype='int', is_arg=True)
    
    rowptr = Tensor('rowptr', (num_node+1,), dtype='int')
    colidx = Tensor('colidx', (num_edge,), dtype='int')
    edge_list =  Set(Tensor('edge_list', (num_jobs, 2), dtype='int'))
    count = Var(name='count', dtype='int', is_arg=False)

    # code_style = FunctionCall

    class inner_subgraph_matching:

        def __init__(self, level, *path):
             self.level = level
             self.path = list(*path)

        def __call__(self, item):
            if self.level == 6:
                return  Set(count).increment(1)

            if self.level==2:
                v0 = item[0]
                v1 = item[1]
                v1_nb =  Set(colidx[rowptr[v1]:rowptr[v1+1]])

                candidate_set = v1_nb
                return candidate_set.apply(inner_subgraph_matching(self.level+1, [v0, v1]))
            
            elif self.level==3:
                v0 = self.path[0]
                v1 = self.path[1]
                v0_nb = Set(colidx[rowptr[v0]:rowptr[v0+1]])
                v1_nb = Set(colidx[rowptr[v1]:rowptr[v1+1]])

                candidate_v3 = v0_nb.intersection(v1_nb)
                return candidate_v3.apply(inner_subgraph_matching(self.level+1,  self.path + [item]))
            
            elif self.level==4:
                v0 = self.path[0]
                v1 = self.path[1]
                v0_nb = Set(colidx[rowptr[v0]:rowptr[v0+1]])
                v1_nb = Set(colidx[rowptr[v1]:rowptr[v1+1]])

                candidate_v4 = v0_nb.intersection(v1_nb)
                return candidate_v4.apply(inner_subgraph_matching(self.level+1, self.path + [item]))
            
            elif self.level==5:
                v0 = self.path[0]
                v2 = self.path[2]
                v0_nb = Set(colidx[rowptr[v0]:rowptr[v0+1]])
                v2_nb =  Set(colidx[rowptr[v2]:rowptr[v2+1]])

                candidate_v5 = v0_nb.intersection(v2_nb)
                return candidate_v5.apply(inner_subgraph_matching(self.level+1, self.path + [item]))

                
    
    def init_vals():
        return setval(count, 0)
    
    # edge_list = edge_list.filter(PartialEdge)
    # edge_list = edge_list.filter(Edgefilter(rowptr, node_degree(pmtx, 0)-1))

    res = edge_list.apply(inner_subgraph_matching(2), init=init_vals).retval( Var(name='return_val', dtype='int', is_arg=False))
    
    #res.name =  'p'+str(pattern_size)+ '_' + os.path.basename(pattern_file_name).split('.')[0]
    res._gen_ir()
    # parallelize(res)
    code = codegen.cpu.print_cpp(res)
    print(code)


def p3_decomine():
    num_node =  Var(name='num_node', dtype='int', is_arg=True)
    num_edge =  Var(name='num_edge', dtype='int', is_arg=True)
    num_jobs =  Var(name='num_jobs', dtype='int', is_arg=True)
    
    rowptr = Tensor('rowptr', (num_node+1,), dtype='int')
    colidx = Tensor('colidx', (num_edge,), dtype='int')
    edge_list =  Set(Tensor('edge_list', (num_jobs, 2), dtype='int'))
    count = Var(name='count', dtype='int', is_arg=False)

    # code_style = FunctionCall

    class inner_subgraph_matching:

        def __init__(self, level, *path):
             self.level = level
             self.path = list(*path)

        def __call__(self, item):
            if self.level == 6:
                return  Set(count).increment(1)

            if self.level==2:
                v0 = item[0]
                v1 = item[1]
                v1_nb =  Set(colidx[rowptr[v1]:rowptr[v1+1]])

                candidate_v2 = v1_nb
                return candidate_v2.apply(inner_subgraph_matching(self.level+1, [v0, v1]))
            
            elif self.level==3:
                v0 = self.path[0]
                v1 = self.path[1]
                v2 = item
                v0_nb = Set(colidx[rowptr[v0]:rowptr[v0+1]])
                v1_nb = Set(colidx[rowptr[v1]:rowptr[v1+1]])
                v2_nb = Set(colidx[rowptr[v2]:rowptr[v2+1]])

                candidate_v3v4 = v0_nb.intersection(v1_nb)
                candidate_v5 = v0_nb.intersection(v2_nb)
                
                v5_nelem = candidate_v5.num_element()
                v3v4_nelem = candidate_v3v4.num_element()

                return v5_nelem * v3v4_nelem * v3v4_nelem - v3v4_nelem                
    
    def init_vals():
        return setval(count, 0)
    
    # edge_list = edge_list.filter(PartialEdge)
    # edge_list = edge_list.filter(Edgefilter(rowptr, node_degree(pmtx, 0)-1))

    res = edge_list.apply(inner_subgraph_matching(2), init=init_vals).retval( Var(name='return_val', dtype='int', is_arg=False))
    
    #res.name =  'p'+str(pattern_size)+ '_' + os.path.basename(pattern_file_name).split('.')[0]
    res._gen_ir()
    # parallelize(res)
    code = codegen.cpu.print_cpp(res)
    print(code)