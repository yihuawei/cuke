import torch
import run
from core.ast2ir import *
import codegen
import helpers

def test1():
    def func():
        A = Tensor('a', (10, 10))
        B = Tensor('b', (10, 10))
        C = Tensor('c', (10, 10))
        return A + B - C

    ast = func()
    code = codegen.cpu.print_cpp(ast._gen_ir())
    print(code)

    A = torch.rand(10, 10)
    B = torch.rand(10, 10)
    C = torch.rand(10, 10)

    d = run.cpu.compile_and_run(code, A, B, C)
    print(torch.equal(A + B - C, d))


def test2():
    def func():
        n = Var('n')
        m = Tensor('m', (n, 2))
        i = Var('idx')
        t = Tensor('t', m[i]._size())
        return m[i] + t

    ast = func()
    code = codegen.cpu.print_cpp(ast._gen_ir())

    n = 10
    m = torch.rand(n, 2)
    i = 5
    t = torch.rand(m[i].shape)

    d = run.cpu.compile_and_run(code, n, m, i, t)
    print(torch.equal(m[i] + t, d))



def test3():
    A = Tensor('a', (10, 2, 2))
    i = Var('i')
    j = Var('j')
    t = Tensor('t', A[i][j]._size())

    ast = A[i][j] + t

    print(helpers.get_input_nodes(ast))

    A = torch.rand(10, 2, 2)
    i = 1
    j = 1
    t = torch.rand(A[i][j].shape)

    d = run.cpu.compile_and_run(codegen.cpu.print_cpp(gen_ir(ast)), A, i, j, t)
    print(torch.equal(A[i][j] + t, d))



def test4():
    A = Tensor('a', (10, ))
    i = Var('i')
    t = Var('t', A.dtype)

    ast = A[i] + t
    ir = gen_ir(ast)
    code = codegen.cpu.print_cpp(ir)

    A = torch.rand(10)
    i = 1
    t = 2
    d = run.cpu.compile_and_run(code, A, i, t)
    print(d, A[i] + t)


def test6():
    A = Tensor('a', (10, ))
    idx = Tensor('idx', (5, ), dtype='int')
    t = Tensor('t', A[idx]._size())

    ast = A[idx] + t
    ir = gen_ir(ast)

    code = codegen.cpu.print_cpp(ir)
    A = torch.rand(10)
    idx = torch.IntTensor(5)
    t = torch.rand(A[idx].shape)
    d = run.cpu.compile_and_run(code, A, idx, t)

    print(d)

    print(torch.equal(d, A[idx] + t))


def test7():
    A = Tensor('a', (10, 10))
    idx = Tensor('idx', (5, ), dtype='int')
    t = Tensor('t', A[idx]._size())

    ast = A[idx] + t
    ir = gen_ir(ast)

    code = codegen.cpu.print_cpp(ir)
    A = torch.rand(10, 10)
    idx = torch.IntTensor(5)
    t = torch.rand(A[idx].shape)
    d = run.cpu.compile_and_run(code, A, idx, t)

    print(d)

    print(torch.equal(d, A[idx] + t))

def test8():
    A = Tensor('a', (10, 10))
    idx = Tensor('idx', (5, ), dtype='int')
    t = Tensor('t', A[0][idx]._size())

    ast = A[0][idx] + t
    ir = gen_ir(ast)

    code = codegen.cpu.print_cpp(ir)
    A = torch.rand(10, 10)
    idx = torch.IntTensor(5)
    t = torch.rand(A[0][idx].shape)
    d = run.cpu.compile_and_run(code, A, idx, t)

    print(d)

    print(torch.equal(d, A[0][idx] + t))

def test9():
    A = Tensor('a', (10, 10))
    i = Tensor('i', (5, ), dtype='int')
    j = Tensor('j', (4, ), dtype='int')
    t = Tensor('t', A[i][j]._size())

    ast = A[i][j] + t
    ir = gen_ir(ast)

    code = codegen.cpu.print_cpp(ir)
    A = torch.rand(10, 10)
    i = torch.IntTensor(5)
    j = torch.IntTensor(4)
    t = torch.rand(A[i][:,j].shape)
    d = run.cpu.compile_and_run(code, A, i, j, t)

    print(A[i][:,j] + t)

    print(torch.equal(d, A[i][:,j] + t))

def test10():
    A = Tensor('a', (10, 11, 12))
    i = Tensor('i', (5, ), dtype='int')
    x = Var('x', 'int')
    j = Tensor('j', (4, ), dtype='int')
    t = Tensor('t', A[i][j][x]._size())

    ast = A[i][j][x] + t
    print(helpers.get_input_nodes(ast))
    ir = gen_ir(ast)

    code = codegen.cpu.print_cpp(ir)
    A = torch.rand(10, 11, 12)
    i = torch.IntTensor(5)
    x = 2
    j = torch.IntTensor(4)
    t = torch.rand(A[i][:,j][:,:,x].shape)
    d = run.cpu.compile_and_run(code, A, i, j, x, t)

    print(A[i][:,j][:,:,x] + t)

    print(torch.equal(d, A[i][:,j][:,:,x] + t))


def test11():
    A = Tensor('a', (10, 11, 12))
    i = Tensor('i', (5, ), dtype='int')
    x = Var('x', 'int')
    y = Var('y', 'int')
    j = Tensor('j', (y, ), dtype='int')
    t = Tensor('t', A[i][x][j]._size())

    ast = A[i][x][j] + t
    print(helpers.get_input_nodes(ast))
    ir = gen_ir(ast)

    code = codegen.cpu.print_cpp(ir)
    A = torch.rand(10, 11, 12)
    i = torch.IntTensor(5)
    x = 2
    y = 3
    j = torch.IntTensor(y)
    t = torch.rand(A[i][:,x][:,j].shape)
    d = run.cpu.compile_and_run(code, y, A, i, x, j, t)

    print(A[i][:,x][:,j] + t)

    print(torch.equal(d, A[i][:,x][:,j] + t))


def test12():
    s1 = Var('s1', 'int')
    s2 = Var('s2', 'int')
    s3 = Var('s3', 'int')
    A = Tensor('a', (s1, s2, s3))
    b1 = Var('b1', 'int')
    v1 = Tensor('v1', (b1, ), dtype='int')
    b2 = Var('b2', 'int')
    v2 = Tensor('v2', (b2, ), dtype='int')
    x = Var('x', 'int')
    t = Tensor('t', A[v1][x][v2]._size())

    ast = A[v1][x][v2] + t
    print(helpers.get_input_nodes(ast))
    ir = gen_ir(ast)

    code = codegen.cpu.print_cpp(ir)
    s1 = 10
    s2 = 20
    s3 = 30
    A = torch.rand(s1, s2, s3)
    b1 = 3
    v1 = torch.IntTensor(b1)
    b2 = 4
    v2 = torch.IntTensor(b2)
    x = 2
    t = torch.rand(A[v1][:,x][:,v2].shape)
    d = run.cpu.compile_and_run(code, b1, b2, s3, s2, s1, A, v1, x, v2, t)

    print(A[v1][:,x][:,v2] + t)

    print(torch.equal(d, A[v1][:,x][:,v2] + t))


def test13():
    s1 = Var('s1', 'int')
    s2 = Var('s2', 'int')
    A = Tensor('a', (s1, s2))
    b1 = Var('b1', 'int')
    v1 = Tensor('v1', (b1, ), dtype='int')
    v2 = Tensor('v2', (b1, ), dtype='int')
    x = Var('x', 'int')
    t = Tensor('t', A[v1+v2][x]._size())

    ast = A[v1+v2][x] + t
    print(helpers.get_input_nodes(ast))
    ir = gen_ir(ast)

    code = codegen.cpu.print_cpp(ir)
    s1 = 10
    s2 = 20
    A = torch.rand(s1, s2)
    b1 = 3
    v1 = torch.IntTensor(b1)
    v2 = torch.IntTensor(b1)
    x = 2
    t = torch.rand(A[v1+v2][:,x].shape)
    d = run.cpu.compile_and_run(code, b1, s2, s1, A, v1, v2, x, t)

    print(A[v1+v2][:,x] + t)

    print(torch.equal(d, A[v1+v2][:,x] + t))


def test14():
    s1 = Var('s1', 'int')
    A = Tensor('A', (s1, ), dtype='int')
    B = Tensor('B', (A[0], A[1]))
    C = Tensor('C', B._size())

    ast = B + C
    print(helpers.get_input_nodes(ast))
    ir = gen_ir(ast)

    code = codegen.cpu.print_cpp(ir)
    s1 = 10
    A = torch.randint(5, 10, (s1, ), dtype=torch.int)
    B = torch.rand(A[0], A[1])
    C = torch.rand(B.shape)

    print(B+C)

    d = run.cpu.compile_and_run(code, s1, A, B, C)


    print(torch.equal(d, B+C))


def test15():
    A = Tensor('A', (100, ))
    B = Tensor('B', (100, ))

    ast = A[1:10] + B[1:10]
    print(helpers.get_input_nodes(ast))
    ir = gen_ir(ast)

    code = codegen.cpu.print_cpp(ir)
    A = torch.rand(100)
    B = torch.rand(100)
    d = run.cpu.compile_and_run(code, A, B)

    print(A[1:10] + B[1:10])
    print(torch.equal(A[1:10] + B[1:10], d))





def test16():
    A = Tensor('A', (100, 20))
    B = Tensor('B', (100, 20))

    ast = A[1:10] + B[1:10]
    print(helpers.get_input_nodes(ast))
    ir = gen_ir(ast)

    code = codegen.cpu.print_cpp(ir)
    A = torch.rand(100, 20)
    B = torch.rand(100, 20)
    d = run.cpu.compile_and_run(code, A, B)

    print(A[1:10] + B[1:10])
    print(torch.equal(A[1:10] + B[1:10], d))


def test17():
    A = Tensor('A', (100, 20))
    B = Tensor('B', (100, 20))

    ast = A[1:10][2:4] + B[1:10][2:4]
    print(helpers.get_input_nodes(ast))
    ir = gen_ir(ast)

    code = codegen.cpu.print_cpp(ir)
    A = torch.rand(100, 20)
    B = torch.rand(100, 20)
    d = run.cpu.compile_and_run(code, A, B)

    print(A[1:10][:, 2:4] + B[1:10][:, 2:4])
    print(d)
    print(torch.equal(A[1:10][:, 2:4] + B[1:10][:, 2:4], d))


def test18():
    A = Tensor('A', (100, 20))
    B = Tensor('B', (100, ), dtype='int')

    ast = A[1:10][B[2:4]] + A[1:10][B[1:3]]
    print(helpers.get_input_nodes(ast))
    ir = gen_ir(ast)

    code = codegen.cpu.print_cpp(ir)
    A = torch.rand(100, 20)
    B = torch.randint(0, 20, (100, )).to(torch.int32)
    d = run.cpu.compile_and_run(code, A, B)

    print(A[1:10][:, B[2:4]] + A[1:10][:, B[1:3]])
    print(d)
    print(torch.equal(A[1:10][:, B[2:4]] + A[1:10][:, B[1:3]], d))

def test19():
    nnodes = 100
    nedges = 300
    rowptr = Tensor('rowptr', (nnodes + 1, ), dtype='int')
    colidx = Tensor('colidx', (nedges, ), dtype='int')
    edge_list = Tensor('edge_list', (10, 2), dtype='int')
    ast = colidx[rowptr[edge_list[0][0]]:rowptr[edge_list[0][1]]] + colidx[rowptr[edge_list[0][0]]:rowptr[edge_list[0][1]]]

    print(helpers.get_input_nodes(ast))
    code = codegen.cpu.print_cpp(gen_ir(ast))
    print(code)




def apply_test1():
    num_edges = 20
    length = 50
    rowidx = Tensor('rowidx', (num_edges,), dtype='int')
    colidx = Tensor('colidx', (num_edges,), dtype='int')
    edge_idx = Tensor('edge_idx', (length,), dtype='int')


    def apply_func(edge_id):
        v0 = rowidx[edge_id]
        v1 = colidx[edge_id] + 1
        return v0 + v1

    res = edge_idx.apply(apply_func)
    code = codegen.cpu.print_cpp(gen_ir(res))
    print(helpers.get_input_nodes(res))

    edge_idx = torch.randint(0, num_edges, (length,)).to(torch.int32)
    rowidx = torch.randint(0, 1000, (num_edges,)).to(torch.int32)
    colidx = torch.randint(0, 1000, (num_edges,)).to(torch.int32)

    d = run.cpu.compile_and_run(code, edge_idx, rowidx, colidx)

    res = torch.zeros_like(edge_idx)
    for i in range(len(edge_idx)):
        e = edge_idx[i]
        v0 = rowidx[e]
        v1 = colidx[e] + 1
        res[i] = v0 + v1

    print(d)
    print(res)
    print(torch.equal(d, res))






def apply_test2():
    d1 = Var('d1')
    d2 = Var('d2')
    A = Tensor('A', (d1, d2))
    B = Tensor('B', (d2, ))
    ast = A.apply(lambda x: x+B, axis=0)
    ir = gen_ir(ast)

    code = codegen.cpu.print_cpp(ir)
    print(code)


def apply_test3():
    d1 = Var('d1')
    d2 = Var('d2')
    d3 = Var('d3')
    A = Tensor('A', (d1, d2), dtype='int')
    B = Tensor('B', (d2, ), dtype='int')
    C = Tensor('C', (d3, ), dtype='int')


    def apply_func(item):
        def apply_func2(item2):
            return C[item2] + B[item2]

        return item.apply(apply_func2)

    ast = A.apply(apply_func)
    ir = gen_ir(ast)

    code = codegen.cpu.print_cpp(ir)
    print(code)



def apply_test4():
    d1 = Var('d1')
    d2 = Var('d2')
    d3 = Var('d3')
    A = Tensor('A', (d1, d2), dtype='int')
    B = Tensor('B', (d2, ))


    def apply_func(item):
        def apply_func2(item2):
            return B[item2]

        return item.apply(apply_func2)

    ast = A.apply(apply_func)
    ir = gen_ir(ast)

    code = codegen.cpu.print_cpp(ir)
    print(code)




def test_aggr1():
    A = Tensor('A', (10, 20))
    indices = Tensor('idx', (30, ), dtype='int')
    res = A.aggr_sum(indices)

    code = codegen.cpu.print_cpp(res._gen_ir())
    print(code)


def spmv():
    m = Var('m', 'int')
    r = Var('r', 'int')
    rowidx = Tensor('ridx', (m, ), 'int')
    colidx = Tensor('cidx', (m, ), 'int')
    val = Tensor('val', (m, ), 'float')

    c = Var('c', 'int')
    y = Tensor('y', (c, ), 'float')

    res = y[colidx] * val
    res = res.aggr_sum(rowidx, size=r)

    code = codegen.cpu.print_cpp(res._gen_ir())
    print(code)

def test_matmul():
    d1 = Var('d1')
    d2 = Var('d2')
    d3 = Var('d3')
    A = Tensor('a', (d1, d2))
    B = Tensor('b', (d2, d3))
    C = A @ B
    ir = gen_ir(C)
    code = codegen.cpu.print_cpp(ir)
    print(helpers.get_input_nodes(ir))


    d1 = 100
    d3 = 30
    d2 = 20
    A = torch.rand(d1, d2)
    B = torch.rand(d2, d3)
    res = run.cpu.compile_and_run(code, d1, d3, d2, A, B)
    print(torch.equal(A @ B, res))


def test_einsum1():
    d1 = Var('d1')
    d2 = Var('d2')
    A = Tensor('a', (d1, ))
    B = Tensor('b', (d2, ))
    C = einsum('i,j->ij', A, B)
    ir = gen_ir(C)
    code = codegen.cpu.print_cpp(ir)
    print(helpers.get_input_nodes(ir))
    # print(code)


    d1 = 100
    d3 = 30
    d2 = 20
    A = torch.rand(d1, )
    B = torch.rand(d2, )
    res = run.cpu.compile_and_run(code, d1, d2, A, B)
    print(torch.equal(torch.einsum('i,j->ij', A, B), res))


def test_einsum2():
    d1 = Var('d1')
    d2 = Var('d2')
    d3 = Var('d3')
    d4 = Var('d4')
    A = Tensor('a', (d1, d2, d3))
    B = Tensor('b', (d1, d3, d4))
    C = einsum('bij,bjk->bik', A, B)
    ir = gen_ir(C)
    code = codegen.cpu.print_cpp(ir)
    print(helpers.get_input_nodes(ir))
    print(code)


    d1 = 8
    d2 = 20
    d3 = 30
    d4 = 10

    A = torch.rand(d1, d2, d3)
    B = torch.rand(d1, d3, d4)
    res = run.cpu.compile_and_run(code, d1, d2, d4, d3, A, B)
    res1 = torch.einsum('bij,bjk->bik', A, B)
    print(torch.norm(res1 - res))


def test_apply5():
    m = Var('nedges')
    n = Var('nnodes')
    rowidx = Tensor('rowidx', (m, ), dtype='int')
    colidx = Tensor('colidx', (m, ), dtype='int')
    rowptr = Tensor('rowptr', (n+1, ), dtype='int')

    v0 = rowidx[0]
    v1 = colidx[1]
    v0_nb = colidx[rowptr[v0]: rowptr[v0+1]].sum() +  colidx[rowptr[v1]: rowptr[v1+1]].sum()

    code = codegen.cpu.print_cpp(v0_nb._gen_ir())
    print(code)

def test22():
    d = Var('d')
    c = Var('c')
    D = Tensor('D', (10, ), dtype='int')
    A = Tensor('A', (d, d+d, D[c]*d))
    B = Tensor('B', (d, d+c, c*d))
    b1 = Var('b1')
    b2 = Var('b2')
    idx = Tensor('idx', (5, ), dtype='int')

    ast = A[1:3][idx][b1:b2] + B[1:3][idx][b1:b2]
    print(helpers.get_input_nodes(ast))
    ir = gen_ir(ast)

    code = codegen.cpu.print_cpp(ir)
    d = 7
    c = 6
    D = torch.randint(10, 15, (10, ), dtype=torch.int)
    A = torch.rand(d, d+d, D[c]*d)
    B = torch.rand(d, d+c, c*d)
    b1 = 4
    b2 = 5
    idx = torch.IntTensor(5)

    d = run.cpu.compile_and_run(code, b2, b1, D, c, d, A, idx, B)

    print(A[1:3][:,idx][:,:,b1:b2] + B[1:3][:,idx][:,:,b1:b2])
    print(torch.equal(A[1:3][:,idx][:,:,b1:b2] + B[1:3][:,idx][:,:,b1:b2], d))

def test23():
    d1 = Var('d1')
    d2 = Var('d2')
    A = Tensor('A', (d1, d2))
    B = Tensor('B', (d2, ))
    ast = A.apply(lambda x: x+B, axis=0)
    ir = gen_ir(ast)
    print(helpers.get_input_nodes(ir))

    code = codegen.cpu.print_cpp(ir)

    d1 = 10
    d2 = 20
    A = torch.ones(d1, d2)
    B = torch.ones(d2)
    res = run.cpu.compile_and_run(code, d1, d2, A, B)
    print(res)



def test24():
    nnodes = Var('nnodes')
    max_d = Var('max_d')
    G = Tensor('G', (nnodes, max_d), dtype='int')
    V = Tensor('V', (nnodes, ), dtype='int')

    def backtrack1(v0):
        def backtrack2(v1):
            return (G[v1] + G[v0]).size()

        res = G[v0].apply(backtrack2)
        res = res.sum()

        return res

    res = V.apply(backtrack1)
    res = res.sum()
    ir = gen_ir(res)
    print(helpers.get_input_nodes(ir))

    code = codegen.cpu.print_cpp(ir)

    nnodes = 100
    max_d = 50
    G = torch.randint(0,100, (nnodes, max_d), dtype=torch.int)
    V = torch.arange(0, nnodes, dtype=torch.int)

    res = run.cpu.compile_and_run(code, nnodes,V, max_d, G)
    print(res)





def test25():
    d1 = Var('d1')
    d2 = Var('d2')
    d3 = Var('d3')
    A = Tensor('A', (d1, d2, d3))
    B = Tensor('B', (d1, d2))
    ast = A.apply(lambda x: x+B, axis=2)
    ir = gen_ir(ast)
    print(helpers.get_input_nodes(ir))

    code = codegen.cpu.print_cpp(ir)

    d1 = 10
    d2 = 20
    d3 = 30
    A = torch.ones(d1, d2, d3)
    B = torch.ones(d1, d2)
    res = run.cpu.compile_and_run(code, d3, d1, d2, A, B)
    print(res)


def test26():
    A = Tensor('a', (10, ))
    ast = A.sum(axis=0)
    ir = gen_ir(ast)
    print(helpers.get_input_nodes(ir))
    code = codegen.cpu.print_cpp(ir)

    A = torch.rand(10)
    init = 0.0
    res = run.cpu.compile_and_run(code, A)
    print(res, torch.sum(A))

def test27():
    A = Tensor('a', (10, 20))
    init = Zeros(A[1]._size())
    ast = A.reduce(lambda a,b: a+b, init, axis=1)
    ir = gen_ir(ast)
    print(helpers.get_input_nodes(ir))
    code = codegen.cpu.print_cpp(ir)
    print(code)

    A = torch.rand(10, 20)
    res = run.cpu.compile_and_run(code, A)
    print(res - torch.sum(A, dim=1))

def test28():
    A = Tensor('a', (10, 20, 5))
    ast = A.sum(axis=1)
    ir = gen_ir(ast)
    print(helpers.get_input_nodes(ir))
    code = codegen.cpu.print_cpp(ir)

    A = torch.rand(10, 20, 5)
    res = run.cpu.compile_and_run(code, A)
    print(res - torch.sum(A, dim=1))

def test29():
    A = Tensor('A', (10, ))
    B = Tensor('B', (10, ), dtype='int')
    res = A[B[0]] + A[B[1]]

    code = codegen.cpu.print_cpp(gen_ir(res))
    print(code)



if __name__ == "__main__":
    # test1()
    # test2()
    # test3()
    # test4()
    # test6()
    # test7()
    # test8()
    # test9()
    # test10()
    # test11()
    # test12()
    # test13()
    # test14()
    # test15()
    # test16()
    # test17()
    # test18()
    # test19()
    apply_test1()
    apply_test2()
    apply_test3()
    apply_test4()
    # test_aggr1()
    # spmv()
    # test_einsum1()
    # apply_test2()
    # test_apply5()
    # test27()