from core.ast import *
from core.ir import *
from cset.ast2ir import *

class Traversal:

    def __init__(self, action):
        self.action = action

    def _post_traverse(self, node, visited, res):
        import batch
        if not isinstance(node, ASTNode):
            return
        if node in visited:
            return
        else:
            visited.add(node)

        if type(node) == Var or type(node) == One or type(node) == Zero:
            self.action(node, res)
        elif type(node) == Const:
            if node.dtype == 'slice':
                self._post_traverse(node.val.start, visited, res)
                self._post_traverse(node.val.stop, visited, res)
                self._post_traverse(node.val.step, visited, res)
        elif type(node) == Tensor or type(node) == Ones or type(node) == Zeros:
            for s in node.fix_size:
                self._post_traverse(s, visited, res)
            for s in node.ref_size:
                self._post_traverse(s, visited, res)
            self.action(node, res)
        elif type(node) == TensorOp:
            for s in node.fix_size:
                self._post_traverse(s, visited, res)
            for s in node.ref_size:
                self._post_traverse(s, visited, res)
            for c in node.operators:
                self._post_traverse(c, visited, res)
            self.action(node, res)
        elif type(node) == batch.ast.Batch:
            self._post_traverse(node.base, visited, res)
        elif type(node) == batch.ast.BatchOp:
            for c in node.operators:
                self._post_traverse(c, visited, res)
            self.action(node, res)
        elif type(node) == Set:
            self._post_traverse(node.storage, visited, res)
            self._post_traverse(node.nelem, visited, res)
            self.action(node, res)
        elif type(node) == SetOp:
            self._post_traverse(node.storage, visited, res)
            self._post_traverse(node.nelem, visited, res)
            for c in node.operators:
                self._post_traverse(c, visited, res)
            self.action(node, res)

    def __call__(self, ast):
        visited = set()
        res = []
        self._post_traverse(ast, visited, res)
        return res

def get_input_nodes(ast):
    def action(node, res):
        if type(node) == Var or type(node) == Tensor:
            if node.is_arg:
                res.append([node.name, node])

    t = Traversal(action)
    return dict(t(ast))

def get_ir_of_size(size):
    ir_size = []
    for s in size:
        assert isinstance(s, ASTNode)
        s._gen_ir()
        ir_size.append(s.eval)
    return ir_size