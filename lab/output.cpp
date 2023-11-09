torch::Tensor add_add_add_a_b_add_c_d_e(torch::Tensor obj_a, torch::Tensor obj_b, torch::Tensor obj_c, torch::Tensor obj_d, torch::Tensor obj_e)
{
    auto a = obj_a.accessor<float, 2>();
    auto b = obj_b.accessor<float, 2>();
    torch::Tensor obj_arr2 = torch::empty({30,30}, at::kFloat);
    auto arr2 = obj_arr2.accessor<float, 2>();
    //First Loop
    // for (int _l0 = 0; _l0 < 30; _l0 += 1) {
    //     for (int _l1 = 0; _l1 < 30; _l1 += 1) {
    //         arr2[_l0][_l1] = a[_l0][_l1] + b[_l0][_l1];
    //     } 
    // } 


    
    auto c = obj_c.accessor<float, 2>();
    auto d = obj_d.accessor<float, 2>();
    torch::Tensor obj_arr7 = torch::empty({30,30}, at::kFloat);
    auto arr7 = obj_arr7.accessor<float, 2>();

    //Second Loop
    // for (int _l2 = 0; _l2 < 30; _l2 += 1) {
    //     for (int _l3 = 0; _l3 < 30; _l3 += 1) {
    //         arr7[_l2][_l3] = c[_l2][_l3] + d[_l2][_l3];
    //     } 
    // } 


    // torch::Tensor obj_arr10 = torch::empty({30,30}, at::kFloat);
    // auto arr10 = obj_arr10.accessor<float, 2>();
    // //Third Loop
    // for (int _l4 = 0; _l4 < 30; _l4 += 1) {
    //     for (int _l5 = 0; _l5 < 30; _l5 += 1) {
    //         // arr2[_l0][_l1] = a[_l0][_l1] + b[_l0][_l1];
    //         // arr7[_l2][_l3] = c[_l2][_l3] + d[_l2][_l3];
    //         arr10[_l4][_l5] =  a[_l4][_l5] + b[_l4][_l5] + c[_l4][_l5] + d[_l4][_l5];
    //     } 
    // } 
    
    //Fourth Loop
    auto e = obj_e.accessor<float, 2>();
    torch::Tensor obj_arr14 = torch::empty({30,30}, at::kFloat);
    auto arr14 = obj_arr14.accessor<float, 2>();
    for (int _l6 = 0; _l6 < 30; _l6 += 1) {
        for (int _l7 = 0; _l7 < 30; _l7 += 1) {
            //arr10[_l4][_l5] =  a[_l4][_l5] + b[_l4][_l5] + c[_l4][_l5] + d[_l4][_l5];
            arr14[_l6][_l7] = a[_l6][_l7] + b[_l6][_l7] + c[_l6][_l7] + d[_l6][_l7]; + e[_l6][_l7];
        } 
    } 
return obj_arr14;

}