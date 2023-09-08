import codegen
from batch.ast import *

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
    vh = Batch(Eemb[h])
    vt = Batch(Eemb[t])
    vr = Batch(Remb[r])

    res = vh - vt + vr

    code = codegen.cpu.print_cpp(res._gen_ir())

    print(code)

def transH():
    nnodes = Var('nnodes')
    nedges = Var('nedges')
    dim = Var('dim')
    batch_size = Var('batch_size')
    Eemb = Tensor('Eemb', (nnodes, dim))
    Remb = Tensor('Remb', (nedges, dim))
    h = Tensor('h', (batch_size, ), dtype='int')
    t = Tensor('t', (batch_size, ), dtype='int')
    r = Tensor('r', (batch_size, ), dtype='int')
    vh = Batch(Eemb[h])
    vt = Batch(Eemb[t])
    vr = Batch(Remb[r])

    res = vh - vt + vr - bsv(bvv(vr, vh - vt), vr)

    code = codegen.cpu.print_cpp(res._gen_ir())

    print(code)

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
    vh = Batch(Eemb[h])
    vt = Batch(Eemb[t])
    mr = Batch(Proj[r])
    vr = Batch(Remb[r])

    res = bvm(vh -vt, mr) + vr

    code = codegen.cpu.print_cpp(res._gen_ir())

    print(code)

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
    vh = Batch(Eemb[h])
    vt = Batch(Eemb[t])
    vr = Batch(Remb[r])

    res = bvv(vh, vt) - bvv(vh - vt, vr)

    code = codegen.cpu.print_cpp(res._gen_ir())

    print(code)

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
    vh = Batch(Eemb[h])
    vt = Batch(Eemb[t])
    mr = Batch(Proj[r])

    res = bvv(bvm(vh, mr), vt)

    code = codegen.cpu.print_cpp(res._gen_ir())

    print(code)



if __name__ == "__main__":
    transE()
    transH()
    transR()
    transF()
    RESCAL()
