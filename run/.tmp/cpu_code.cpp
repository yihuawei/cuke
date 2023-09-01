#include <torch/extension.h>

torch::Tensor add_index_index_index_A_c7_idx_c9_index_index_index_B_c12_idx_c14(int b2, int b1, int d, torch::Tensor obj_A, torch::Tensor obj_idx, torch::Tensor obj_B)
{
    int s6;
s6 = 100 - 0;
int s7;
s7 = s6 / 1;
int s8;
s8 = b2 - b1;
int s9;
s9 = s8 / 1;
auto A = obj_A.accessor<float, 3>();
auto idx = obj_idx.accessor<int, 1>();
auto B = obj_B.accessor<float, 3>();
torch::Tensor obj_arr10 = torch::empty({s7,5,s9}, at::kFloat);
auto arr10 = obj_arr10.accessor<float, 3>();
for (int _l0 = 0; _l0 < s7; _l0 += 1) {
for (int _l1 = 0; _l1 < 5; _l1 += 1) {
for (int _l2 = 0; _l2 < s9; _l2 += 1) {
arr10[_l0][_l1][_l2] = A[((0)+(1)*(_l0))][idx[_l1]][((b1)+(1)*(_l2))] + B[((0)+(1)*(_l0))][idx[_l1]][((b1)+(1)*(_l2))];
} 
} 
} 
return obj_arr10;

}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &add_index_index_index_A_c7_idx_c9_index_index_index_B_c12_idx_c14);
}