import sys
from codegen import *
from batch.ast import *
import run
import torch
from helpers import new_op
import opt

@new_op
def bvv(a, b):
    return apply(lambda x, y: einsum('i,i->', x, y), (a, b))

@new_op
def bsv(a, b):
    return apply(lambda x, y: x * y, (a, b))

@new_op
def bvm(a, b):
    return apply(lambda x, y: einsum('i,ij->j', x, y), (a, b))

@new_op
def bov(a, b):
    return apply(lambda x, y: einsum('i,j->ij', x, y), (a, b))

def transE():
    nnodes = Var('nnodes')
    nedges = Var('nedges')
    dim = Var('dim')
    batch_size = Var('batch_size')
    Eemb = Tensor('Eemb', (nnodes, dim))
    Remb = Tensor('Remb', (nedges, dim))
    h = Tensor('h', (batch_size, ), dtype='int')
    t = Tensor('t', (batch_size, ), dtype='int')
    r = Tensor('r', (batch_size, ), dtype='int')
    vh = Eemb[h]
    vt = Eemb[t]
    vr = Remb[r]

    res = vh - vt + vr

    code = codegen.cpu.print_cpp(opt.fuse.fuse(res._gen_ir()))

    print(code)


def transH():
    nnodes = Var('nnodes')
    nedges = Var('nedges')
    dim = Var('dim')
    batch_size = Var('batch_size')
    Eemb = Tensor('Eemb', (nnodes, dim))
    Remb = Tensor('Remb', (nedges, dim))
    Pemb = Tensor('Pemb', (nedges, dim))
    h = Tensor('h', (batch_size, ), dtype='int')
    t = Tensor('t', (batch_size, ), dtype='int')
    r = Tensor('r', (batch_size, ), dtype='int')
    vh = Eemb[h]
    vt = Eemb[t]
    vr = Remb[r]
    vp = Pemb[r]

    res = vh - vt + vr - bsv(bvv(vp, vh - vt), vp)

    code = codegen.cpu.print_cpp(opt.fuse.fuse(res._gen_ir()))
    # ast = res._gen_ir()
    # fuse_operators(ast)
    # tile_loop(ast)
    # parallel(ast)
    # add_smem(ast)
    # code = codegen.cpu.print_cpp(ast)
    # code = codegen.gpu.print_cuda(ast)
    print(code)
    # h = torch.randint(0, 9999, (4096, )).cuda(0)
    # r = torch.randint(0, 100, (4096, )).cuda(0)
    # t = torch.randint(0, 9999, (4096, )).cuda(0)
    # eemb = torch.rand((9999, 512)).cuda(0)
    # remb = torch.rand((100, 512)).cuda(0)
    # pemb = torch.rand((100, 512)).cuda(0)

    # y = eemb[h] + remb[r] - eemb[t] - torch.einsum('a,ab->ab', torch.einsum('ab,ab->a', pemb[r], eemb[h]-eemb[t]), pemb[r])
    # print(y)
    
    # x = run.gpu.compile_and_run(code, 4096, 512, 0, eemb, h,t, 0, remb, r, pemb)
    # print(x)


def transR():
    nnodes = Var('nnodes')
    nedges = Var('nedges')
    dim = Var('dim')
    batch_size = Var('batch_size')
    Eemb = Tensor('Eemb', (nnodes, dim))
    Remb = Tensor('Remb', (nedges, dim))
    Proj = Tensor('Proj', (nedges, dim, dim))
    h = Tensor('h', (batch_size, ), dtype='int')
    t = Tensor('t', (batch_size, ), dtype='int')
    r = Tensor('r', (batch_size, ), dtype='int')
    vh = Eemb[h]
    vt = Eemb[t]
    mr = Proj[r]
    vr = Remb[r]

    res = bvm(vh -vt, mr) + vr

    code = codegen.cpu.print_cpp(res._gen_ir())
    
    # ast = res._gen_ir()
    # fuse_operators(ast)
    # todo decouple operators
    # tile_loop(ast)
    # parallel(ast)
    # add_smem(ast)
    # code = codegen.cpu.print_cpp(ast)
    # code = codegen.gpu.print_cuda(ast)
    print(code)
    # h = torch.randint(0, 9999, (4096, )).cuda(0)
    # r = torch.randint(0, 100, (4096, )).cuda(0)
    # t = torch.randint(0, 9999, (4096, )).cuda(0)
    # eemb = torch.rand((9999, 512)).cuda(0)
    # remb = torch.rand((100, 512)).cuda(0)
    # pemb = torch.rand((100, 512, 512)).cuda(0)

    # y = torch.einsum('ab,abc->ac', eemb[h] - eemb[t], pemb[r]) + remb[r]
    # print(y)
    
    # x = run.gpu.compile_and_run(code, 4096, 512, 0, eemb, h,t, 0, pemb, r, remb)
    # print(x)
    

def transF():
    nnodes = Var('nnodes')
    nedges = Var('nedges')
    dim = Var('dim')
    batch_size = Var('batch_size')
    Eemb = Tensor('Eemb', (nnodes, dim))
    Remb = Tensor('Remb', (nedges, dim))
    h = Tensor('h', (batch_size, ), dtype='int')
    t = Tensor('t', (batch_size, ), dtype='int')
    r = Tensor('r', (batch_size, ), dtype='int')
    vh = Eemb[h]
    vt = Eemb[t]
    vr = Remb[r]
    
    # alpha = Const(val=2, dtype='float')
    # alpha = Batch(alpha)
    # alpha = 2

    res = bvv(vh, vt) - bvv(vh - vt, vr)
    
    code = codegen.cpu.print_cpp(res._gen_ir())
    # ast = res._gen_ir()
    # fuse_operators(ast)
    # tile_loop(ast)
    # parallel(ast)
    # add_smem(ast)
    # code = codegen.gpu.print_cuda(ast)
    print(code)
    # h = torch.randint(0, 9999, (4096, )).cuda(0)
    # r = torch.randint(0, 100, (4096, )).cuda(0)
    # t = torch.randint(0, 9999, (4096, )).cuda(0)
    # eemb = torch.rand((9999, 512)).cuda(0)
    # remb = torch.rand((100, 512)).cuda(0)

    # y = torch.einsum('ab,ab->a', eemb[h], eemb[t]) - torch.einsum('ab,ab->a',(eemb[h] - eemb[t]), remb[r])
    # print(y)
    
    # x = run.gpu.compile_and_run(code, 4096, 512, 0, eemb, h,t, 0, remb, r)
    # print(x)

def RESCAL():
    nnodes = Var('nnodes')
    nedges = Var('nedges')
    dim = Var('dim')
    batch_size = Var('batch_size')
    Eemb = Tensor('Eemb', (nnodes, dim))
    Proj = Tensor('Proj', (nedges, dim, dim))
    h = Tensor('h', (batch_size, ), dtype='int')
    t = Tensor('t', (batch_size, ), dtype='int')
    r = Tensor('r', (batch_size, ), dtype='int')
    vh = Eemb[h]
    vt = Eemb[t]
    mr = Proj[r]

    res = bvv(bvm(vh, mr), vt)

    code = codegen.cpu.print_cpp(res._gen_ir())
    
    # ast = res._gen_ir()
    
    # fuse_operators(ast)
    # tile_loop(ast)
    # parallel(ast)
    # add_smem(ast)
    # traversal call funcs to opt ir
    # code = codegen.cpu.print_cpp(ast)
    # code = codegen.gpu.print_cuda(ast)
    print(code)

    # h = torch.randint(0, 9999, (4096, )).cuda(0)
    # r = torch.randint(0, 100, (4096, )).cuda(0)
    # t = torch.randint(0, 9999, (4096, )).cuda(0)
    # eemb = torch.rand((9999, 512)).cuda(0)
    # remb = torch.rand((100, 512, 512)).cuda(0)

    # y = torch.einsum('ab,ab->a', torch.einsum('ab,abc->ac', eemb[h], remb[r]), eemb[t])
    # print(y, y.shape)
    
    # x = run.gpu.compile_and_run(code, 4096, 512, 0, eemb, h, 0, remb, r, t)
    # print(x)

def test():
    nnodes = Var('nnodes')
    nedges = Var('nedges')
    dim = Var('dim')
    batch_size = Var('batch_size')
    Eemb = Tensor('Eemb', (nnodes, dim))
    Remb = Tensor('Remb', (nedges, dim))
    Proj = Tensor('Proj', (nedges, dim, dim))
    h = Tensor('h', (batch_size, ), dtype='int')
    t = Tensor('t', (batch_size, ), dtype='int')
    r = Tensor('r', (batch_size, ), dtype='int')
    vh = Eemb[h]
    vt = Eemb[t]
    vr = Remb[r]

    res = bov(vh+vr, vt-vr)

    code = codegen.cpu.print_cpp(res._gen_ir())
    print(code)
    # h = torch.randint(0, 9999, (4096, )).cuda(0)
    # r = torch.randint(0, 100, (4096, )).cuda(0)
    # t = torch.randint(0, 9999, (4096, )).cuda(0)
    # eemb = torch.rand((9999, 512)).cuda(0)
    # remb = torch.rand((100, 512)).cuda(0)
    #
    # y = torch.einsum('ab,ac->abc', eemb[h] + remb[r], eemb[t] - remb[r])
    # print(y)
    #
    # x = run.gpu.compile_and_run(code, 4096, 512, 0, eemb, h, 0, remb, r, t)
    # print(x)

if __name__ == "__main__":
    # transE() # success
    transH() # success
    # transR() # success
    # transF() # success
    # RESCAL() # success
    # test()