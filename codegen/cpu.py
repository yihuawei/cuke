from core.ast2ir import *
from cset.ast2ir import *
import helpers
import batch
import cset



def to_string(ir):
    match ir.__class__.__name__:
        case 'Expr':
            if ir.op in op_mapping.values():
                return f"({to_string(ir.left)}" + f" {ir.op} " + f"{to_string(ir.right)})"
            elif ir.op == 'bigger':
                return f"({to_string(ir.left)} > {to_string(ir.right)} ? ({to_string(ir.left)}) : ({to_string(ir.right)}))"
            elif ir.op == 'smaller':
                return f"({to_string(ir.left)} < {to_string(ir.right)} ? ({to_string(ir.left)}) : ({to_string(ir.right)}))"
        case 'Assignment':
            if ir.op is None:
                return f"{to_string(ir.lhs)} = {to_string(ir.rhs)};\n"
            else:
                return f"{to_string(ir.lhs)} {ir.op}= {to_string(ir.rhs)};\n"
        case 'Loop':
            code = f"for (int {to_string(ir.iterate)} = {to_string(ir.start)}; {to_string(ir.iterate)} < {to_string(ir.end)}; {to_string(ir.iterate)} += {to_string(ir.step)}) {{\n"
            for e in ir.body:
                if e:
                    code += to_string(e)
            code += "} \n"
            return code
        case 'FilterLoop':
            code = f"for (int {to_string(ir.iterate)} = {to_string(ir.start)}; {to_string(ir.iterate)} < {to_string(ir.end)}; {to_string(ir.iterate)} += {to_string(ir.step)}) {{\n"
            for e in ir.body:
                if e:
                    code += to_string(e)
            if ir.cond:
                code += f"if({to_string(ir.cond)}){{\n"
                for e in ir.cond_body:
                    if e:
                        code += to_string(e)
                code += "} \n"
            code += "} \n"
            return code
        case 'Not':
            return f"!{to_string(ir.dobject)}"
        case 'Scalar' | 'Ndarray' | 'Ref':
            return ir.name()
        case 'Literal':
            return str(ir.val)
        case 'Indexing':
            if type(ir.dobject) == Slice:
                if ir.dobject.step == 1 or (type(ir.dobject.step) == Literal and ir.dobject.step.val == 1):
                    return f'(({to_string(ir.dobject.start)})+({to_string(ir.idx)}))'
                else:
                    return f'(({to_string(ir.dobject.start)})+({to_string(ir.dobject.step)})*({to_string(ir.idx)}))'
            else:
                return f'{to_string(ir.dobject)}[{to_string(ir.idx)}]'
        # case 'Search':
        #     code = f"BinarySearch({to_string(ir.dobject)}, {to_string(ir.start)}, {to_string(ir.end)}, {to_string(ir.item)})"
        #     return code
        case 'Decl':
            # variables are passed in as pytorch arguments
            if type(ir.dobject) == Scalar:
                if not ir.dobject.is_arg:
                    # it is a zero or one
                    if ir.dobject.val != None:
                        return f"{ir.dobject.dtype} {ir.dobject.name()} = {to_string(ir.dobject.val)};\n"
                    else:
                        return f"{ir.dobject.dtype} {ir.dobject.name()};\n"
                else:
                    return ''
            elif type(ir.dobject) == Ndarray:
                code = ''
                if not ir.dobject.is_arg:
                    if ir.dobject.val != None:
                        code = f'torch::Tensor obj_{ir.dobject.name()} = torch::{"ones" if ir.dobject.val == 1 else "zeros"}({{{",".join([to_string(s) for s in ir.dobject.size])}}}, at::k{"Int" if ir.dobject.dtype=="int" else "Float"});\n'
                    else:
                        code = f'torch::Tensor obj_{ir.dobject.name()} = torch::empty({{{",".join([to_string(s) for s in ir.dobject.size])}}}, at::k{"Int" if ir.dobject.dtype=="int" else "Float"});\n'

                code += f'auto {ir.dobject.name()} = obj_{ir.dobject.name()}.accessor<{ir.dobject.dtype}, {len(ir.dobject.size)}>();\n'
                return code
        case 'Math':
            return f"{ir.type}({to_string(ir.val)})"
        case _:
            return str(ir)


def gen_cpp(ast, ir):
    def action(node, res):
        if node.valid == True:
            if type(node) == Var or type(node) == One or type(node) == Zero or type(node) == Ones or type(node) == Zeros or type(node) == Tensor:
                res.extend(node.decl)
            elif type(node) == TensorOp:
                res.extend(node.decl)
                res.extend(node.compute)
            elif type(node) == batch.ast.BatchOp:
                res.extend(node.decl)
                res.extend(node.compute)
            elif type(node) == cset.ast.Set:
                res.extend(node.decl)
            elif type(node) == cset.ast.SetOp:
                res.extend(node.decl)
                res.extend(node.compute)

    t = helpers.Traversal(action)
    ir.extend(t(ast))

def print_cpp(ast):
    ir = []
    gen_cpp(ast, ir)


    args = helpers.get_input_nodes(ast)
    args = ', '.join([f'torch::Tensor obj_{a}' if type(args[a]) == Tensor else f'{args[a].dtype} {a}' for a in args])

    code = ''
    for d in ir:
        if d:
            code += to_string(d)


    if type(ast.eval) == Scalar:
        rtype = ast.dtype
        code += f'return {ast.eval.name()};\n'
    elif type(ast.eval) == Ndarray:
        rtype = 'torch::Tensor'
        code += f'return obj_{ast.eval.name()};\n'
    else:
        raise TypeError('wrong output type', ast.eval)

    with open('codegen/cpp_template.cpp', 'r') as f:
        c_code = f.read()
        c_code = c_code.replace('RTYPE', rtype).replace('FNAME', ast.name).replace('ARGS', args).replace('CODE', code)
    return c_code