

# #How to calculate the new lower bounds and new upper bounds
# New lower bound(i) = Original lower bound(i) - (tile_size(i) - 1)
# if i is an expression of k:  
#     New upper bound(i) = Original upper bound(i) + (tile_size(k) -1) 
# else
#     New upper bound(i) = Original upper bound(i)

# #input loop
# tile_size = [3, 4, 5] 
# tile_size -1 = [2, 3, 4]
# for (int _l0 = 0; _l0 < 20; _l0 += 1) {
    
#     for (int _l1 = _l0 + 3; _l1 < _l0 + 10; _l1 += 1) {
        
#         for (int _l2 = _l1 + 20; _l2 < _l1 + 30; _l2 += 1) {
            
#             A[_l0 + 1][_l1][_l2] = B[_l0][_l1][_l2] + 2;
#         } 
#     } 
# } 

# #Original lower bounds of _l0 loop: 0
# #Original upper bounds of _l0 loop: 20

# def RoundLowerBound(original_lower_bound, tile_size):
#     return ceil(original_lower_bound/tile_size) * tile_size; 


# tiled loop 
# for (       int _l0 = RoundLowerBound(0 - (tile_size[0]-1), tile_size[0])
#                 _l0 < 20; 
#                 _l0 += tile_size[0]) {
    
#     #Original lower bounds of _l1 loop: _l0 + 3
#     #Original upper bounds of _l1 loop: _l0 + 10
    
#     #For tile loop
#     # New lower bound(i) = Original lower bound(i) - (tile_size(i) - 1)
#     # if i is an expression of k:  
#     #     New upper bound(i) = Original upper bound(i) + (tile_size(k) -1) 
#     # else
#     #     New upper bound(i) = Original upper bound(i)

#     for (   int _l1 = RoundLowerBound(_l0 + 3 - (tile_size[1]-1), tile_size[1]); 
#                 _l1 < _l0 + 10 + (tile_size[0]); 
#                 _l1 += tile_size[1]) {
        
#         for (int _l2 = RoundLowerBound(_l1 + 20 - (tile_size[2]-1), tile_size[2]); 
#                  _l2 <  _l1 + 30 + (tile_size[1]);
#                   _l2 += tile_size[2]) {
            
#             #For point loop
#             #corresponding tile loop index for point-loop level(i) = iT 
#             #new lower bound(i) = max(iT, original lower bound(iT))
#             #new upper bound(i) = min(iT + tile_size(i)-1, original upper bound(iT)))
            
#             #Original lower bound(_l0p) = original lower bound(_l0) = 0
#             #Original upper bound(_l0p) = original upper bound(_l0) = 20


#             for (int _l0p = max(_l0, 0); 
#                     _l0p <  min(_l0 + tile_size[0], 20); 
#                     _l0p += 1) {
                
#                 for (int _l1p = max(_l1, _l0p + 3); 
#                          _l1p < min(_l1 + tile_size[1], _l0p + 10); 
#                          _l1p += 1) {
                    
#                     for (int _l2p = max(_l2 , _l1p + 20); 
#                              _l2p < min(_l2 + tile_size[1], _l1p + 30);
#                              _l2p += 1) {
#                         #A[_l3 + 1][_l4][_l5] = B[_l3][_l4][_l5] + 2;
#         } 
#     } 
# } 




import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.ir import *
from codegen.cpu import *

def PrintCCode(ir):
	code = ''
	for d in ir:
		# if d:
			code += to_string(d)
	print(code)

def Loop0():
    ir = []

    L = Scalar('int', 'L')
    M = Scalar('int', 'M')
    N = Scalar('int', 'N')
    A = Ndarray('int', (N, M, L), 'A')
    B = Ndarray('int', (N, M, L), 'B')

    loopi = Loop(0, 20, 1, [])
    loopj = Loop(Expr(loopi.iterate, 3, '+'),  Expr(loopi.iterate, 10, '+'), 1, [])
    loopk = Loop(Expr(loopj.iterate, 20, '+'), Expr(loopj.iterate, 30, '+'), 1, [])

    loopi.body.append(loopj)
    loopj.body.append(loopk)

    lhs1 = Index(Index(Index(A, Expr(loopi.iterate, 1, '+')), loopj.iterate), loopk.iterate)
    rhs1 = Index(Index(Index(B, loopi.iterate), loopj.iterate), loopk.iterate)
	
    # body = Assignment(lhs, Expr(rhs1, rhs2, '+'))
    loopk.body.extend([Assignment(lhs1, Expr(rhs1, 2, '+'))])

    ir.extend([Decl(L)])
    ir.extend([Decl(M)])
    ir.extend([Decl(N)])
    ir.extend([Decl(A)])
    ir.extend([Decl(B)])
    ir.extend([loopi])

    return ir


def GetKeyInfo(loop_ir):
    def _GetKeyInfo(loop_ir, lower_bounds, upper_bounds, index_dict, level):
        if not type(loop_ir)==Loop:
            return
        if type(loop_ir)==Loop:
            lower_bounds.append(loop_ir.start)
            upper_bounds.append(loop_ir.end)
            index_dict[loop_ir.iterate] = level
            _GetKeyInfo(loop_ir.body[0], lower_bounds, upper_bounds, index_dict, level+1)

    index_dict = {}
    lower_bounds = []
    upper_bounds = []
    _GetKeyInfo(loop_ir, lower_bounds, upper_bounds, index_dict, 0)
    return lower_bounds, upper_bounds, index_dict


def GetNewLowerBound(lower_bound_expr, tile_size):
    # New lower bound(i) = Original lower bound(i) - (tile_size(i) - 1)
    return Expr(lower_bound_expr, Expr(tile_size, 1, '-'), '-')

def GetNewUpperBound(upper_bound_expr, index_dict, tile_size_list):
    # if i is an expression of k:  
    #     New upper bound(i) = Original upper bound(i) + (tile_size(k) -1) 
    # else
    #     New upper bound(i) = Original upper bound(i)

    if type(upper_bound_expr)==Expr: # upper_bound_expr.left + upper_bound_expr.right 
        # upper_bound_expr.left  is k in the algorithm 
        iterator_index = upper_bound_expr.left
        return Expr(upper_bound_expr, tile_size_list[index_dict[iterator_index]], '+')
    else:
        return upper_bound_expr



def LoopTiling(ir, tile_size = []):
    for ir_item in ir:
        if type(ir_item) == Loop:
            # for (int _l0 = 0; _l0 < 20; _l0 += 1) {
            # for (int _l1 = _l0 + 3; _l1 < _l0 + 10; _l1 += 1) {
            # for (int _l2 = _l1 + 20; _l2 < _l1 + 30; _l2 += 1) {   
            #lower_bounds and upper_bounds are two lists recording the IRs of 
            #all corresponding lower bounds and upper bounds. 

            #index_dict is an dict recording the mapping between the index IR and loop index
            #Since _l0, _l1 and _l2 are the scalar objects instead of a number. 
            
            #lower_bounds is an array: [0, _l0 + 3, _l1 + 20]
            #upper_bounds is an array: [20, _l0 + 10, _l1 + 30]
            #index_dict is a map: { _l0: 0, 
            #                       _l1: 1,  
            #                       _l2: 2}
            lower_bounds, upper_bounds, index_dict = GetKeyInfo(ir_item)
            # PrintCCode(lower_bounds)
            # PrintCCode(upper_bounds)

            # for item in index_dict.items():
            #     PrintCCode([item[0]])
            #     PrintCCode([item[1]])

            for upper_bound_expr in upper_bounds:
                #Type(upper_bound_expr) is an Exper or a nunmber
                new_upper_bound = GetNewUpperBound(upper_bound_expr, index_dict, tile_size)
                PrintCCode([new_upper_bound])

    
	

if __name__ == "__main__":
    loop0_ir = Loop0()  # 3 level loop
    PrintCCode(loop0_ir)

    # LoopTiling(loop0_ir, tile_size = [10, 4, 5])

    # loop0_ir_after_tiling = LoopTiling(loop0_ir, tile_size = [3,4, 5])
    # PrintCCode(loop0_ir_after_tiling)


# #How to calculate the new lower bounds and new upper bounds
# New lower bound(i) = Original lower bound(i) - (tile_size(i) - 1)
# if i is an expression of k:  
#     New upper bound(i) = Original upper bound(i) + (tile_size(k) -1) 
# else
#     New upper bound(i) = Original upper bound(i)

# #input loop
# tile_size = [3, 4, 5] 
# tile_size -1 = [2, 3, 4]
# for (int _l0 = 0; _l0 < 20; _l0 += 1) {
    
#     for (int _l1 = _l0 + 3; _l1 < _l0 + 10; _l1 += 1) {
        
#         for (int _l2 = _l1 + 20; _l2 < _l1 + 30; _l2 += 1) {
            
#             A[_l0 + 1][_l1][_l2] = B[_l0][_l1][_l2] + 2;
#         } 
#     } 
# } 

# #Original lower bounds of _l0 loop: 0
# #Original upper bounds of _l0 loop: 20

# def RoundLowerBound(original_lower_bound, tile_size):
#     return ceil(original_lower_bound/tile_size) * tile_size; 


# tiled loop 
# for (       int _l0 = RoundLowerBound(0 - (tile_size[0]-1), tile_size[0])
#                 _l0 < 20; 
#                 _l0 += tile_size[0]) {
    
#     #Original lower bounds of _l1 loop: _l0 + 3
#     #Original upper bounds of _l1 loop: _l0 + 10
    
#     #For tile loop
#     # New lower bound(i) = Original lower bound(i) - (tile_size(i) - 1)
#     # if i is an expression of k:  
#     #     New upper bound(i) = Original upper bound(i) + (tile_size(k) -1) 
#     # else
#     #     New upper bound(i) = Original upper bound(i)

#     for (   int _l1 = RoundLowerBound(_l0 + 3 - (tile_size[1]-1), tile_size[1]); 
#                 _l1 < _l0 + 10 + (tile_size[0]); 
#                 _l1 += tile_size[1]) {
        
#         for (int _l2 = RoundLowerBound(_l1 + 20 - (tile_size[2]-1), tile_size[2]); 
#                  _l2 <  _l1 + 30 + (tile_size[1]);
#                   _l2 += tile_size[2]) {
            
#             #For point loop
#             #corresponding tile loop index for point-loop level(i) = iT 
#             #new lower bound(i) = max(iT, original lower bound(iT))
#             #new upper bound(i) = min(iT + tile_size(i)-1, original upper bound(iT)))
            
#             #Original lower bound(_l0p) = original lower bound(_l0) = 0
#             #Original upper bound(_l0p) = original upper bound(_l0) = 20


#             for (int _l0p = max(_l0, 0); 
#                     _l0p <  min(_l0 + tile_size[0], 20); 
#                     _l0p += 1) {
                
#                 for (int _l1p = max(_l1, _l0p + 3); 
#                          _l1p < min(_l1 + tile_size[1], _l0p + 10); 
#                          _l1p += 1) {
                    
#                     for (int _l2p = max(_l2 , _l1p + 20); 
#                              _l2p < min(_l2 + tile_size[1], _l1p + 30);
#                              _l2p += 1) {
#                         #A[_l3 + 1][_l4][_l5] = B[_l3][_l4][_l5] + 2;
#         } 
#     } 
# } 