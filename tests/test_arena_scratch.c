#include <time.h>
#include <stdint.h>
#include <stdio.h>

int main() {

    struct timespec start, end;
    

    uint8_t memory[1024];
    float * ptr_a = (float *)&memory[0];
    float * ptr_b = (float *)&memory[1];

    // Start the clock
    clock_gettime(CLOCK_MONOTONIC, &start);

    // Benchmark 1 byte alignment
    for (int i = 0; i < 100000000; i++) {
        *ptr_a = i * 2.0f;
    }

    // Stop the clock
    clock_gettime(CLOCK_MONOTONIC, &end);

    // Calculate total time in milliseconds
    long elapsed_ms = (end.tv_sec - start.tv_sec) * 1000 + (end.tv_nsec - start.tv_nsec) / 1000000;
    printf("Aligned time:   %ld ms\n", elapsed_ms);

    // Start the clock
    clock_gettime(CLOCK_MONOTONIC, &start);

    // Benchmark 2 byte alignment
    for (int i = 0; i < 100000000; i++) {
        *ptr_b = i * 2.0f;
    }

    // Stop the clock
    clock_gettime(CLOCK_MONOTONIC, &end);

    // Calculate total time in milliseconds
    elapsed_ms = (end.tv_sec - start.tv_sec) * 1000 + (end.tv_nsec - start.tv_nsec) / 1000000;
    printf("Misaligned time:   %ld ms\n", elapsed_ms);
    

    return 0;
}