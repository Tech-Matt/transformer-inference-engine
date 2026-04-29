/**
 * @file arena.h
 * @author Mattia Rizzo (mattia.rizzo.un@gmail.com)
 * @brief 
 * @version 0.1
 * @date 2026-04-29
 * 
 * @copyright Copyright (c) 2026
 * SPDX-License-Identifier: MIT
 */

#include <stddef.h>
#include <stdint.h>

/**
 * @brief The Arena Allocator
 * 
 * The Arena allocator type. Used to allocate data on a contiguous block of
 * memory. Data is never overwritten, unless the memory block is full and the
 * entire Arena is reset. To allocate a new data element, the simple pointer
 * offset is advanced to the next free block.
 * 
 * @param buffer
 * @param size
 * @param offset
 */
typedef struct {
    uint8_t *buffer;
    size_t size;
    size_t offset;
} Arena;

/**
 * @brief Initializes an Arena allocator with a pre-allocated memory buffer.
 * 
 * @param a Pointer to the Arena structure to be initialized.
 * @param buffer Pointer to the contiguous memory block to be managed.
 * @param size Total size of the provided buffer. 
 */
void arena_init(Arena *a, void *buffer, size_t size);


/**
 * @brief Allocates a block of memory from the arena.
 * 
 * The returned pointer is automatically aligned to the DEFAULT_ALIGNMENT
 * boundary (e.g. 16 bytes). If the request exceeds the remaining capacity
 * of the arena, this function returns NULL.
 * 
 * @param a Pointer to the Arena.
 * @param size Number of bytes requested.
 * @return void* Pointer to the newly allocated memory, or NULL if out of memory.
 */
void * arena_alloc(Arena *a, size_t size);


/**
 * @brief Resets the arena, effectively freeing all existing allocations. 
 * 
 * @param a Pointer to the Arena.
 */
void arena_reset(Arena *a);
