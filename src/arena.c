/**
 * @file arena.c
 * @author Mattia Rizzo (mattia.rizzo.un@gmail.com)
 * @brief 
 * @version 0.1
 * @date 2026-04-28
 * 
 * @copyright Copyright (c) 2026
 * SPDX-License-Identifier: MIT 
 */

#include "arena.h"

#ifndef DEFAULT_ALIGNMENT
#define DEFAULT_ALIGNMENT 16
#endif

/**
 * @brief Rounds an offset up to the nearest multiple of 'align'.
 * 
 * Works by adding (align - 1) to push the value to or past the next 
 * alignment boundary, then masks off the lower bits to snap to the 
 * exact boundary. 
 * Note: 'align' MUST be a power of two for the bitwise mask to work.
 */
#define ALIGN_FORWARD(offset, align) (((offset) + ((align) - 1)) & ~((align) - 1))


void arena_init(Arena *a, void *buffer, size_t size) {
    a->buffer = (uint8_t *)buffer;
    a->size = size;
    a->offset = 0;
}

void * arena_alloc(Arena *a, size_t size) {
    // Align the memory offset to ensure speed + chance to use SIMD optimizations
    a->offset = ALIGN_FORWARD(a->offset, DEFAULT_ALIGNMENT);

    // If offset is out of bounds, or the requested size exceeds the
    // remaining capacity, return NULL. Arena is full.
    if (a->offset > a->size || size > a->size - a->offset) {
        return NULL;
    }
    
    // Save the pointer to be returned
    void * ret_ptr = &(a->buffer[a->offset]);

    // Update offset to the next free memory location
    a->offset += size;

    return ret_ptr;
}


void arena_reset(Arena *a) {
    // It's as simple as that :)
    a->offset = 0;
}