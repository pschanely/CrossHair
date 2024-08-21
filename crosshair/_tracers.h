#ifndef _COVERAGE_TRACER_H
#define _COVERAGE_TRACER_H

#include "structmember.h"

#include <Python.h>

#define RET_OK               0
#define RET_ERROR           -1
#define RET_DISABLE_TRACING  1

typedef int BOOL;
#define FALSE   0
#define TRUE    1


#define DEFINE_VEC(N, T, INITNAME, PUSHNAME) \
typedef struct N {int count; int capacity; T * items;} N ; \
void INITNAME (N * vec, int cap) \
{ \
    vec->count = 0; \
    vec->capacity = cap; \
    vec->items = PyMem_Malloc(sizeof(T) * cap); \
    memset(vec->items, 0, sizeof(T) * cap); \
} \
int PUSHNAME (N * vec, T item) \
{ \
    int count = vec->count; \
    int capacity = vec->capacity; \
    T* items = vec->items; \
    if (count >= capacity) \
    { \
        size_t halfsize = sizeof(T) * capacity; \
        vec->capacity = (capacity *= 2); \
        items = PyMem_Realloc(vec->items, halfsize << 1); \
        if (items == NULL) \
        { \
            return RET_ERROR; \
        } \
        memset(((unsigned char *) items) + halfsize, 0, halfsize); \
        vec->items = items; \
    } \
    items[count] = item; \
    vec->count++; \
    return RET_OK; \
}

typedef struct HandlerTable {
    PyObject * entries[256];
} HandlerTable;


typedef struct FrameAndCallback {
    PyObject* frame;
    PyObject* callback;
} FrameAndCallback;


typedef struct CodeAndStacks {
    PyCodeObject* code_obj;
    int64_t *stacks;
} CodeAndStacks;


DEFINE_VEC(FrameAndCallbackVec, FrameAndCallback, init_framecbvec, push_framecb);
DEFINE_VEC(ModuleVec, PyObject*, init_modulevec, push_module);
DEFINE_VEC(TableVec, HandlerTable, init_tablevec, push_table_entry)

typedef struct CTracer {
    PyObject_HEAD
    ModuleVec modules;
    TableVec handlers;
    FrameAndCallbackVec postop_callbacks;
    BOOL enabled;
    BOOL handling;
    BOOL trace_all_opcodes;
    int thread_id;
} CTracer;

extern PyTypeObject CTracerType;

typedef struct TraceSwap {
    PyObject_HEAD
    BOOL noop;
    BOOL disabling;
    PyObject* tracer;
} TraceSwap;

extern PyTypeObject TraceSwapType;

#endif /* _COVERAGE_TRACER_H */
