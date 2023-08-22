import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.ir import *
from codegen.cpu import *

def PrintCCode(ir):
	code = ''
	for d in ir:
		if d:
			code += to_string(d)
	print(code)

def BuildNestedLoopByIR():
	ir = []

	N = Scalar('int', 'N')
	A = Ndarray('int', (N, N), 'A')

	outer_loop = Loop(0, N, 1, [])
	inner_loop = Loop(0, N, 1, [])
	outer_loop.body.append(inner_loop)

	lhs = Index(Index(A, outer_loop.iterate), inner_loop.iterate)
	rhs = Index(Index(A, Expr(outer_loop.iterate, 1, '-')), Expr(inner_loop.iterate, 1, '-'))

	inner_loop_body = Assignment(lhs, Expr(rhs, 1, '+'))
	inner_loop.body.append(inner_loop_body)

	ir.extend([Decl(N)])
	ir.extend([Decl(A)])
	ir.extend([outer_loop])

	return ir


def LoopInterchange(ir):
	optimized_ir = []
	for item in ir:
		if type(item)==Loop:
			outer_loop = item
			inner_loop = item.body[0]
			body = inner_loop.body[0]

			inner_loop.body.clear()
			outer_loop.body.clear()
			interchanged_loop = inner_loop
			interchanged_loop.body.append(outer_loop)
			interchanged_loop.body[0].body.append(body)
			optimized_ir.append(interchanged_loop)
		else:
			optimized_ir.append(item)
	return optimized_ir

if __name__ == "__main__":
	ir = BuildNestedLoopByIR()
	print("Loop before interchange:")
	PrintCCode(ir)

	optimized_ir = LoopInterchange(ir)
	print("Loop after interchange:")
	PrintCCode(optimized_ir)



