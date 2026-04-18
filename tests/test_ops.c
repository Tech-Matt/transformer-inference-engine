#include "unity.h"
#include "ops.h"
#include "tensor.h"

void setUp(void) {}
void tearDown(void) {}

void test_relu(void) 
{
    int dim[1] = {4};
    float in_data[4] = {-2.0f, -0.5f, 0.0f, 3.0f};
    float out_data[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    Tensor *in = Tensor_new(1, dim, in_data);
    Tensor *out = Tensor_new(1, dim, out_data);

    relu(in, out);

    TEST_ASSERT_FLOAT_WITHIN_MESSAGE(1e-6f, 0.0f, out->data[0], "test_relu: wrong output value");
    TEST_ASSERT_FLOAT_WITHIN_MESSAGE(1e-6f, 0.0f, out->data[1], "test_relu: wrong output value");
    TEST_ASSERT_FLOAT_WITHIN_MESSAGE(1e-6f, 0.0f, out->data[2], "test_relu: wrong output value");
    TEST_ASSERT_FLOAT_WITHIN_MESSAGE(1e-6f, 3.0f, out->data[3], "test_relu: wrong output value");

    Tensor_free(in);
    Tensor_free(out);
}

void test_softmax(void) {
    int dim[1] = {3};
    float in_data[3] = {0.0f, 0.0f, 0.0f};
    float out_data[3] = {0.0f, 0.0f, 0.0f};

    Tensor *in = Tensor_new(1, dim, in_data);
    Tensor *out = Tensor_new(1, dim, out_data);

    softmax(in, out);

    TEST_ASSERT_FLOAT_WITHIN_MESSAGE(1e-5f, 0.333333f, out->data[0], "test_softmax: wrong output[0]");
    TEST_ASSERT_FLOAT_WITHIN_MESSAGE(1e-5f, 0.333333f, out->data[1], "test_softmax: wrong output[1]");
    TEST_ASSERT_FLOAT_WITHIN_MESSAGE(1e-5f, 0.333333f, out->data[2], "test_softmax: wrong output[2]");

    Tensor_free(in);
    Tensor_free(out);
}

void test_log_softmax(void) {
    int dim[1] = {2};
    float in_data[2] = {0.0f, 0.0f};
    float out_data[2] = {0.0f, 0.0f};

    Tensor *in = Tensor_new(1, dim, in_data);
    Tensor *out = Tensor_new(1, dim, out_data);

    log_softmax(in, out);

    TEST_ASSERT_FLOAT_WITHIN_MESSAGE(1e-5f, -0.693147f, out->data[0], "test_log_softmax: wrong output[0]");
    TEST_ASSERT_FLOAT_WITHIN_MESSAGE(1e-5f, -0.693147f, out->data[1], "test_log_softmax: wrong output[1]");

    Tensor_free(in);
    Tensor_free(out);
}

void test_mean(void) {
    int dim[2] = {2, 2};
    float in_data[4] = {1.0f, 2.0f, 3.0f, 4.0f};

    int out_dim[2] = {1, 2};
    float out_data[2] = {0.0f, 0.0f};

    Tensor *in = Tensor_new(2, dim, in_data);
    Tensor *out = Tensor_new(2, out_dim, out_data);

    mean(in, out, 0); // reduce over axis 0 (rows)

    TEST_ASSERT_FLOAT_WITHIN_MESSAGE(1e-5f, 2.0f, out->data[0], "test_mean: wrong output[0]"); // (1+3)/2 = 2.0
    TEST_ASSERT_FLOAT_WITHIN_MESSAGE(1e-5f, 3.0f, out->data[1], "test_mean: wrong output[1]"); // (2+4)/2 = 3.0

    Tensor_free(in);
    Tensor_free(out);
}

void test_var(void) {
    int dim[1] = {3};
    float in_data[3] = {1.0f, 2.0f, 3.0f};

    int out_dim[1] = {1};
    float out_data[1] = {0.0f};

    Tensor *in = Tensor_new(1, dim, in_data);
    Tensor *out = Tensor_new(1, out_dim, out_data);

    var(in, out, 0);

    TEST_ASSERT_FLOAT_WITHIN_MESSAGE(1e-5f, 0.666666f, out->data[0], "test_var: wrong output[0]");

    Tensor_free(in);
    Tensor_free(out);
}

void test_softmax_stability(void) {
    int dim[1] = {2};
    // Very large values would normally cause e^1000 to overflow to infinity
    float in_data[2] = {1000.0f, 1000.0f};
    float out_data[2] = {0.0f, 0.0f};

    Tensor *in = Tensor_new(1, dim, in_data);
    Tensor *out = Tensor_new(1, dim, out_data);

    softmax(in, out);

    // Because softmax subtracts the max value first, this should stay perfectly stable
    TEST_ASSERT_FLOAT_WITHIN_MESSAGE(1e-5f, 0.5f, out->data[0], "Stability failed: should be 0.5");
    TEST_ASSERT_FLOAT_WITHIN_MESSAGE(1e-5f, 0.5f, out->data[1], "Stability failed: should be 0.5");

    Tensor_free(in);
    Tensor_free(out);
}

void test_log_softmax_translation(void) {
    int dim[1] = {2};
    // Log softmax is translation invariant: adding a constant to inputs doesn't change output.
    // [101.0, 102.0] should have the exact same output as [1.0, 2.0].
    float in_data[2] = {101.0f, 102.0f};
    float out_data[2] = {0.0f, 0.0f};

    Tensor *in = Tensor_new(1, dim, in_data);
    Tensor *out = Tensor_new(1, dim, out_data);

    log_softmax(in, out);

    // log(e^0 + e^1) = 1.31326.  0 - 1.31326 = -1.31326, 1 - 1.31326 = -0.31326
    TEST_ASSERT_FLOAT_WITHIN_MESSAGE(1e-5f, -1.31326f, out->data[0], "Translation invariance failed [0]");
    TEST_ASSERT_FLOAT_WITHIN_MESSAGE(1e-5f, -0.31326f, out->data[1], "Translation invariance failed [1]");

    Tensor_free(in);
    Tensor_free(out);
}

void test_mean_axis1(void) {
    int dim[2] = {2, 2};
    // [[1, 2],
    //  [3, 4]]
    float in_data[4] = {1.0f, 2.0f, 3.0f, 4.0f};

    // Reducing over axis 1 (columns) should yield a shape of [2, 1]
    int out_dim[2] = {2, 1};
    float out_data[2] = {0.0f, 0.0f};

    Tensor *in = Tensor_new(2, dim, in_data);
    Tensor *out = Tensor_new(2, out_dim, out_data);

    mean(in, out, 1);

    TEST_ASSERT_FLOAT_WITHIN_MESSAGE(1e-5f, 1.5f, out->data[0], "(1+2)/2 = 1.5");
    TEST_ASSERT_FLOAT_WITHIN_MESSAGE(1e-5f, 3.5f, out->data[1], "(3+4)/2 = 3.5");

    Tensor_free(in);
    Tensor_free(out);
}

void test_var_zero(void) {
    int dim[1] = {4};
    // Identical elements have no variance
    float in_data[4] = {7.3f, 7.3f, 7.3f, 7.3f};
    int out_dim[1] = {1};
    float out_data[1] = {0.0f};

    Tensor *in = Tensor_new(1, dim, in_data);
    Tensor *out = Tensor_new(1, out_dim, out_data);

    var(in, out, 0);

    TEST_ASSERT_FLOAT_WITHIN_MESSAGE(1e-5f, 0.0f, out->data[0], "Variance of constants should be exactly 0");

    Tensor_free(in);
    Tensor_free(out);
}

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(test_relu);
    RUN_TEST(test_softmax);
    RUN_TEST(test_log_softmax);
    RUN_TEST(test_mean);
    RUN_TEST(test_var);
    RUN_TEST(test_softmax_stability);
    RUN_TEST(test_log_softmax_translation);
    RUN_TEST(test_mean_axis1);
    RUN_TEST(test_var_zero);
    return UNITY_END();
}