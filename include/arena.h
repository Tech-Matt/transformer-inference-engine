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


/**
 * @brief Bookmarks the current offset of the arena.
 * Useful for allocating temporary memory that can be safely rolled back.
 *
 * @note This function MUST be strictly paired with arena_free_to_bookmark()
 *
 * @return size_t The current memory offset.
 */
static inline size_t arena_get_bookmark(Arena *a) {
    return a->offset;
}

/**
 * @brief Rolls back the arena offset to a previously saved bookmark.
 * It effectively frees all temporary allocations made since the bookmark.
 *
 * @warning STRICT LIFO LIFETIME: This effectively destroys ALL allocations
 * made on this arena since the bookmark was taken. You must absolutely
 * ensure that no persistent/unrelated allocations occured on this arena between
 * arena_get_bookmark() and this call.
 *
 * @param a Pointer to an Arena allocator
 * @param bookmark The offset to roll back to.
 */
static inline void arena_free_to_bookmark(Arena *a, size_t bookmark) {
    a->offset = bookmark;
}
