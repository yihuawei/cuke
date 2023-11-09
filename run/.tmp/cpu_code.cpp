#include <torch/extension.h>

torch::Tensor func()
{
    int L;
int M;
int N;
torch::Tensor obj_A = torch::empty({N,M,L}, at::kInt);
auto A = obj_A.accessor<int, 3>();
for (int _l3 = 0; _l3 < L; _l3 += 1) {
for (int _l4 = 0; _l4 < M; _l4 += 1) {
for (int _l5 = 0; _l5 < N; _l5 += 1) {
A[_l5 + 1][_l4 + 1][_l3] = A[_l5][_l4][_l3] + A[_l5][_l4 + 1][_l3 + 1];
} 
} 
} 
return obj_A;

}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &func);
}