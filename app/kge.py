from ext.batch.ast2ir import *
from core.ast2ir import *
import codegen

if __name__ == "__main__":
    nnodes = Var('nnodes')
    nedges = Var('nedges')
    dim = Var('dim')
    batch_size = Var('batch_size')
    Eemb = Tensor('Eemb', (nnodes, dim))
    Remb = Tensor('Remb', (nedges, dim))
    h = Tensor('h', (batch_size, ), dtype='int')
    t = Tensor('t', (batch_size, ), dtype='int')
    r = Tensor('r', (batch_size, ), dtype='int')

    res = BVec(Eemb[h]) - BVec(Eemb[t]) + BVec(Remb[r])

    code = codegen.cpu.print_cpp(res._gen_ir())

    print(code)

