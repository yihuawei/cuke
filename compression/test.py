from core.asg import *
from compression.asg import *
import codegen
from core.opt.fuse import fuse


def compression():
    input = Tensor('input', (50, 32), dtype='float')
    quant_res = (input * 1000).round()
    lorenzo_res = quant_res.apply(lambda x:x[0:32]-x[-1:31], axis=0)
    encode_nbits = lorenzo_res.abs().max(axis=1).nbits()
    ofs = encode_nbits.prefix_sum()
    compressed_res = apply(lambda x, y: truncate(x, y),lorenzo_res, encode_nbits, out_ofs=ofs)
    res = compressed_res
    code = codegen.cpu.print_cpp(fuse(res._gen_ir()))
    print(code)


if __name__ == "__main__":
    compression()