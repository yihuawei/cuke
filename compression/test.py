import helpers
from core.asg import *
from compression.asg import *
import codegen
from core.opt.fuse import fuse
import torch
import run


def compression():
    nblocks = 50
    input = Tensor('input', (nblocks, 32), dtype='float')
    quant_res = (input * 1000).round()
    lorenzo_res = quant_res[:][0:32] - quant_res[:][-1:31]
    encode_nbits = lorenzo_res.abs().max(axis=1).nbits()
    ofs = encode_nbits.prefix_sum(inclusive=False)
    compressed_res = apply(lambda x, y: truncate(x, y),lorenzo_res, encode_nbits, out_ofs=ofs)
    res = compressed_res
    code = codegen.cpu.print_cpp(res._gen_ir())
    code = '#include "../../compression/util.h"\n' + code

    data = torch.rand(nblocks, 32)

    d = run.cpu.compile_and_run(code, data)
    print(d)


def invert_lorenzo():
    nblocks = 50
    input = Tensor('input', (nblocks, 32), dtype='float')
    quant_res = (input * 1000).round()
    lorenzo_res = quant_res[:][0:32] - quant_res[:][-1:31]

    recover_res = Tensor('recovered', input._size(), dtype='float')
    recover_res = setval(recover_res, recover_res[:][-1:31] + lorenzo_res[:][0:32])

    res = recover_res

    code = codegen.cpu.print_cpp(res._gen_ir())
    code = '#include "../../compression/util.h"\n' + code

    data = torch.rand(nblocks, 32)

    print((data * 1000).round())

    d = run.cpu.compile_and_run(code, data)
    print(d)





if __name__ == "__main__":
    compression()
    # invert_lorenzo()