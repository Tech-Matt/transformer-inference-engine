#include "unity.h"
#include "tensor.h"

void setUp(void) {}
void tearDown(void) {}

void test_tensor_alloc_free(void) {
    int dim[2] = {2, 3};
    float data[6] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};

    Tensor *t = Tensor_new(2, dim, data);

    TEST_ASSERT_EQUAL_INT(2, t->ndim);
    TEST_ASSERT_EQUAL_INT(2, t->dim[0]);
    TEST_ASSERT_EQUAL_INT(3, t->dim[1]);
    
    // Check strides: for [2, 3], strides should be [3, 1]
    TEST_ASSERT_EQUAL_INT(3, t->strides[0]);
    TEST_ASSERT_EQUAL_INT(1, t->strides[1]);

    TEST_ASSERT_FLOAT_WITHIN(1e-5f, 1.0f, t->data[0]);
    TEST_ASSERT_FLOAT_WITHIN(1e-5f, 6.0f, t->data[5]);

    Tensor_free(t);
}

void test_tensor_add(void) {
    int dim[1] = {3};
    float data_a[3] = {1.0f, 2.0f, 3.0f};
    float data_b[3] = {10.0f, 20.0f, 30.0f};
    float data_out[3] = {0.0f, 0.0f, 0.0f};

    Tensor *a = Tensor_new(1, dim, data_a);
    Tensor *b = Tensor_new(1, dim, data_b);
    Tensor *out = Tensor_new(1, dim, data_out);

    tensor_add(a, b, out);

    TEST_ASSERT_FLOAT_WITHIN(1e-5f, 11.0f, out->data[0]);
    TEST_ASSERT_FLOAT_WITHIN(1e-5f, 22.0f, out->data[1]);
    TEST_ASSERT_FLOAT_WITHIN(1e-5f, 33.0f, out->data[2]);

    Tensor_free(a);
    Tensor_free(b);
    Tensor_free(out);
}

void test_tensor_matmul(void) {
    // a is [2, 2]
    int dim_a[2] = {2, 2};
    float data_a[4] = {1.0f, 2.0f,
                       3.0f, 4.0f};

    // b is [2, 3]
    int dim_b[2] = {2, 3};
    float data_b[6] = {1.0f, 2.0f, 3.0f,
                       4.0f, 5.0f, 6.0f};

    // out is [2, 3]
    int dim_out[2] = {2, 3};
    float data_out[6] = {0};

    Tensor *a = Tensor_new(2, dim_a, data_a);
    Tensor *b = Tensor_new(2, dim_b, data_b);
    Tensor *out = Tensor_new(2, dim_out, data_out);

    tensor_matmul(a, b, out);

    // out[0][0]: 1*1 + 2*4 = 9
    // out[0][1]: 1*2 + 2*5 = 12
    // out[0][2]: 1*3 + 2*6 = 15
    // out[1][0]: 3*1 + 4*4 = 19
    // out[1][1]: 3*2 + 4*5 = 26
    // out[1][2]: 3*3 + 4*6 = 33

    TEST_ASSERT_FLOAT_WITHIN(1e-5f, 9.0f, out->data[0]);
    TEST_ASSERT_FLOAT_WITHIN(1e-5f, 12.0f, out->data[1]);
    TEST_ASSERT_FLOAT_WITHIN(1e-5f, 15.0f, out->data[2]);
    TEST_ASSERT_FLOAT_WITHIN(1e-5f, 19.0f, out->data[3]);
    TEST_ASSERT_FLOAT_WITHIN(1e-5f, 26.0f, out->data[4]);
    TEST_ASSERT_FLOAT_WITHIN(1e-5f, 33.0f, out->data[5]);

    Tensor_free(a);
    Tensor_free(b);
    Tensor_free(out);
}

void test_tensor_transpose(void) {
    int dim_a[2] = {2, 3};
    float data_a[6] = {1.0f, 2.0f, 3.0f,
                       4.0f, 5.0f, 6.0f};

    int dim_out[2] = {3, 2};
    float data_out[6] = {0};

    Tensor *a = Tensor_new(2, dim_a, data_a);
    Tensor *out = Tensor_new(2, dim_out, data_out);

    tensor_transpose(a, out);

    // Expected transposed layout:
    // 1 4
    // 2 5
    // 3 6

    TEST_ASSERT_FLOAT_WITHIN(1e-5f, 1.0f, out->data[0]);
    TEST_ASSERT_FLOAT_WITHIN(1e-5f, 4.0f, out->data[1]);
    TEST_ASSERT_FLOAT_WITHIN(1e-5f, 2.0f, out->data[2]);
    TEST_ASSERT_FLOAT_WITHIN(1e-5f, 5.0f, out->data[3]);
    TEST_ASSERT_FLOAT_WITHIN(1e-5f, 3.0f, out->data[4]);
    TEST_ASSERT_FLOAT_WITHIN(1e-5f, 6.0f, out->data[5]);

    Tensor_free(a);
    Tensor_free(out);
}

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(test_tensor_alloc_free);
    RUN_TEST(test_tensor_add);
    RUN_TEST(test_tensor_matmul);
    RUN_TEST(test_tensor_transpose);
    return UNITY_END();
}