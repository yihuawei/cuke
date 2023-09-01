import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.ir import *


def PrintCCode(ir):
	code = ''
	for d in ir:
		if d:
			code += to_string(d)
	print(code)

# Original Loop:
# for (int _l0 = 0; _l0 < N; _l0 += 1) {
# 	A0[_l0] = _l0;
# } 
# for (int _l1 = 0; _l1 < N; _l1 += 1) {
# 	A1[_l1] = _l1;
# } 
def BuildTwoLoopByIR():
	ir = []

	N = Scalar('int', 'N')
	A0 = Ndarray('int', (N, ), 'A0')
	A1 = Ndarray('int', (N, ), 'A1')

	loop0 = Loop(0, N, 1, [])
	loop1 = Loop(0, N, 1, [])
	

	body0 = Assignment(Index(A0, loop0.iterate), loop0.iterate)
	body1 = Assignment(Index(A1, loop1.iterate), loop1.iterate)

	loop0.body.append(body0)
	loop1.body.append(body1)

	ir.extend([Decl(N)])
	ir.extend([Decl(A0)])
	ir.extend([Decl(A1)])
	ir.extend([loop0])
	ir.extend([loop1])

	return ir


# Loop after fusion:
# int N;
# torch::Tensor obj_A0 = torch::empty({N}, at::kInt);
# auto A0 = obj_A0.accessor<int, 1>();
# torch::Tensor obj_A1 = torch::empty({N}, at::kInt);
# auto A1 = obj_A1.accessor<int, 1>();
# for (int _l0 = 0; _l0 < N; _l0 += 1) {
# 	A0[_l0] = _l0;
# 	A1[_l0] = _l0;
# } 
def LoopFusion(ir):
	optimized_ir = []
	fused_loops = []
	for item in ir:
		if type(item)==Loop:
			if len(fused_loops) == 0:
				fused_loops.append(item)
				optimized_ir.append(item)
			else:
				if fused_loops[0].start == item.start and fused_loops[0].end == item.end and fused_loops[0].step == item.step:
					for statement in item.body:
						lhs = None
						rhs = None
						if type(statement) == Assignment:
							lhs = statement.lhs
							rhs = statement.rhs
						elif type(statement) == Expr:
							lhs = statement.left
							rhs = statement.right
						if type(lhs) == Index:
							statement.lhs.index = fused_loops[0].iterate
						if type(rhs) == Index:
							statement.rhs.index = fused_loops[0].iterate
						if type(rhs) == Scalar:
							statement.rhs = fused_loops[0].iterate				
					fused_loops[0].body.append(item.body[0])
			
		else:
			optimized_ir.append(item)
	return optimized_ir

if __name__ == "__main__":
	ir = BuildTwoLoopByIR()
	print("Loop before fusion:")
	PrintCCode(ir)

	optimized_ir = LoopFusion(ir)
	print("Loop after fusion:")
	PrintCCode(optimized_ir)



