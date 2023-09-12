import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.ast2ir import *
import run
from core import helpers
import codegen
import torch

def run_demo():
    def func():
        A = Tensor('a', (10, 10))
        B = Tensor('b', (10, 10))
        C = Tensor('c', (10, 10))
        return A + B - C

    ast = func()
    code = codegen.cpu.gen_cpp(gen_ir(ast))

    # code = codegen.cpu.print_cpp(InterchangeLoop(ast._gen_ir(), [0, 1]))

    A = torch.rand(10, 10)
    B = torch.rand(10, 10)
    C = torch.rand(10, 10)

    d = run.cpu.compile_and_run(code, A, B, C)
    print(torch.equal(A + B - C, d))



if __name__ == "__main__":
    run_demo()
