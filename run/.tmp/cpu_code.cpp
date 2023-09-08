#include <torch/extension.h>

torch::Tensor apply_A__c0_item_of_A(int d1, int d2, torch::Tensor obj_A)
{
    auto A = obj_A.accessor<float, 2>();
torch::Tensor obj_arr7 = torch::empty({d1,d2}, at::kFloat);
auto arr7 = obj_arr7.accessor<float, 2>();
auto B = obj_B.accessor<float, 1>();
torch::Tensor obj_arr5 = torch::empty({d2}, at::kFloat);
auto arr5 = obj_arr5.accessor<float, 1>();
for (int _l0 = 0; _l0 < d1; _l0 += 1) {
for (int _l1 = 0; _l1 < d2; _l1 += 1) {
arr5[_l1] = A[_l0][_l1] + B[_l1];
arr7[_l0][_l1] = arr5[_l1];
} 
} 
return obj_arr7;

}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &apply_A__c0_item_of_A);
}