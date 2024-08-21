#include "_tracers_pycompat.h"
#include "internal/pycore_code.h"


// This file includes a modified version of CPython's mark_stacks
// implementation from:
// https://github.com/python/cpython/blob/v3.12.0/Objects/frameobject.c

// The shared source code is licensed under the PSF license and is
// copyright © 2001-2023 Python Software Foundation; All Rights Reserved

// See the "LICENSE" file for complete license details on CrossHair.



#define UNINITIALIZED -9
#define OVERFLOWED -8
#define EMPTY_STACK 0

static int64_t
_ch_pop_to_level(int64_t stack, int level) {
    if (level == 0) {
        return EMPTY_STACK;
    }
    if (level < stack) {
        return level;
    } else {
        return stack;
    }
}


#if PY_VERSION_HEX >= 0x030D0000
#if PY_VERSION_HEX < 0x030E0000
//    ===========
//    Python 3.13
//    ===========

// from Include/internal/pycore_opcode_metadata.h
const uint8_t _ch_PyOpcode_Caches[256] = {
    [JUMP_BACKWARD] = 1,
    [TO_BOOL] = 3,
    [BINARY_SUBSCR] = 1,
    [STORE_SUBSCR] = 1,
    [SEND] = 1,
    [UNPACK_SEQUENCE] = 1,
    [STORE_ATTR] = 4,
    [LOAD_GLOBAL] = 4,
    [LOAD_SUPER_ATTR] = 1,
    [LOAD_ATTR] = 9,
    [COMPARE_OP] = 1,
    [CONTAINS_OP] = 1,
    [POP_JUMP_IF_TRUE] = 1,
    [POP_JUMP_IF_FALSE] = 1,
    [POP_JUMP_IF_NONE] = 1,
    [POP_JUMP_IF_NOT_NONE] = 1,
    [FOR_ITER] = 1,
    [CALL] = 3,
    [BINARY_OP] = 1,
};

// from Include/internal/pycore_opcode_metadata.h
const uint8_t _ch_PyOpcode_Deopt[256] = {
    [BEFORE_ASYNC_WITH] = BEFORE_ASYNC_WITH,
    [BEFORE_WITH] = BEFORE_WITH,
    [BINARY_OP] = BINARY_OP,
    [BINARY_OP_ADD_FLOAT] = BINARY_OP,
    [BINARY_OP_ADD_INT] = BINARY_OP,
    [BINARY_OP_ADD_UNICODE] = BINARY_OP,
    [BINARY_OP_INPLACE_ADD_UNICODE] = BINARY_OP,
    [BINARY_OP_MULTIPLY_FLOAT] = BINARY_OP,
    [BINARY_OP_MULTIPLY_INT] = BINARY_OP,
    [BINARY_OP_SUBTRACT_FLOAT] = BINARY_OP,
    [BINARY_OP_SUBTRACT_INT] = BINARY_OP,
    [BINARY_SLICE] = BINARY_SLICE,
    [BINARY_SUBSCR] = BINARY_SUBSCR,
    [BINARY_SUBSCR_DICT] = BINARY_SUBSCR,
    [BINARY_SUBSCR_GETITEM] = BINARY_SUBSCR,
    [BINARY_SUBSCR_LIST_INT] = BINARY_SUBSCR,
    [BINARY_SUBSCR_STR_INT] = BINARY_SUBSCR,
    [BINARY_SUBSCR_TUPLE_INT] = BINARY_SUBSCR,
    [BUILD_CONST_KEY_MAP] = BUILD_CONST_KEY_MAP,
    [BUILD_LIST] = BUILD_LIST,
    [BUILD_MAP] = BUILD_MAP,
    [BUILD_SET] = BUILD_SET,
    [BUILD_SLICE] = BUILD_SLICE,
    [BUILD_STRING] = BUILD_STRING,
    [BUILD_TUPLE] = BUILD_TUPLE,
    [CACHE] = CACHE,
    [CALL] = CALL,
    [CALL_ALLOC_AND_ENTER_INIT] = CALL,
    [CALL_BOUND_METHOD_EXACT_ARGS] = CALL,
    [CALL_BOUND_METHOD_GENERAL] = CALL,
    [CALL_BUILTIN_CLASS] = CALL,
    [CALL_BUILTIN_FAST] = CALL,
    [CALL_BUILTIN_FAST_WITH_KEYWORDS] = CALL,
    [CALL_BUILTIN_O] = CALL,
    [CALL_FUNCTION_EX] = CALL_FUNCTION_EX,
    [CALL_INTRINSIC_1] = CALL_INTRINSIC_1,
    [CALL_INTRINSIC_2] = CALL_INTRINSIC_2,
    [CALL_ISINSTANCE] = CALL,
    [CALL_KW] = CALL_KW,
    [CALL_LEN] = CALL,
    [CALL_LIST_APPEND] = CALL,
    [CALL_METHOD_DESCRIPTOR_FAST] = CALL,
    [CALL_METHOD_DESCRIPTOR_FAST_WITH_KEYWORDS] = CALL,
    [CALL_METHOD_DESCRIPTOR_NOARGS] = CALL,
    [CALL_METHOD_DESCRIPTOR_O] = CALL,
    [CALL_NON_PY_GENERAL] = CALL,
    [CALL_PY_EXACT_ARGS] = CALL,
    [CALL_PY_GENERAL] = CALL,
    [CALL_STR_1] = CALL,
    [CALL_TUPLE_1] = CALL,
    [CALL_TYPE_1] = CALL,
    [CHECK_EG_MATCH] = CHECK_EG_MATCH,
    [CHECK_EXC_MATCH] = CHECK_EXC_MATCH,
    [CLEANUP_THROW] = CLEANUP_THROW,
    [COMPARE_OP] = COMPARE_OP,
    [COMPARE_OP_FLOAT] = COMPARE_OP,
    [COMPARE_OP_INT] = COMPARE_OP,
    [COMPARE_OP_STR] = COMPARE_OP,
    [CONTAINS_OP] = CONTAINS_OP,
    [CONTAINS_OP_DICT] = CONTAINS_OP,
    [CONTAINS_OP_SET] = CONTAINS_OP,
    [CONVERT_VALUE] = CONVERT_VALUE,
    [COPY] = COPY,
    [COPY_FREE_VARS] = COPY_FREE_VARS,
    [DELETE_ATTR] = DELETE_ATTR,
    [DELETE_DEREF] = DELETE_DEREF,
    [DELETE_FAST] = DELETE_FAST,
    [DELETE_GLOBAL] = DELETE_GLOBAL,
    [DELETE_NAME] = DELETE_NAME,
    [DELETE_SUBSCR] = DELETE_SUBSCR,
    [DICT_MERGE] = DICT_MERGE,
    [DICT_UPDATE] = DICT_UPDATE,
    [END_ASYNC_FOR] = END_ASYNC_FOR,
    [END_FOR] = END_FOR,
    [END_SEND] = END_SEND,
    [ENTER_EXECUTOR] = ENTER_EXECUTOR,
    [EXIT_INIT_CHECK] = EXIT_INIT_CHECK,
    [EXTENDED_ARG] = EXTENDED_ARG,
    [FORMAT_SIMPLE] = FORMAT_SIMPLE,
    [FORMAT_WITH_SPEC] = FORMAT_WITH_SPEC,
    [FOR_ITER] = FOR_ITER,
    [FOR_ITER_GEN] = FOR_ITER,
    [FOR_ITER_LIST] = FOR_ITER,
    [FOR_ITER_RANGE] = FOR_ITER,
    [FOR_ITER_TUPLE] = FOR_ITER,
    [GET_AITER] = GET_AITER,
    [GET_ANEXT] = GET_ANEXT,
    [GET_AWAITABLE] = GET_AWAITABLE,
    [GET_ITER] = GET_ITER,
    [GET_LEN] = GET_LEN,
    [GET_YIELD_FROM_ITER] = GET_YIELD_FROM_ITER,
    [IMPORT_FROM] = IMPORT_FROM,
    [IMPORT_NAME] = IMPORT_NAME,
    [INSTRUMENTED_CALL] = INSTRUMENTED_CALL,
    [INSTRUMENTED_CALL_FUNCTION_EX] = INSTRUMENTED_CALL_FUNCTION_EX,
    [INSTRUMENTED_CALL_KW] = INSTRUMENTED_CALL_KW,
    [INSTRUMENTED_END_FOR] = INSTRUMENTED_END_FOR,
    [INSTRUMENTED_END_SEND] = INSTRUMENTED_END_SEND,
    [INSTRUMENTED_FOR_ITER] = INSTRUMENTED_FOR_ITER,
    [INSTRUMENTED_INSTRUCTION] = INSTRUMENTED_INSTRUCTION,
    [INSTRUMENTED_JUMP_BACKWARD] = INSTRUMENTED_JUMP_BACKWARD,
    [INSTRUMENTED_JUMP_FORWARD] = INSTRUMENTED_JUMP_FORWARD,
    [INSTRUMENTED_LINE] = INSTRUMENTED_LINE,
    [INSTRUMENTED_LOAD_SUPER_ATTR] = INSTRUMENTED_LOAD_SUPER_ATTR,
    [INSTRUMENTED_POP_JUMP_IF_FALSE] = INSTRUMENTED_POP_JUMP_IF_FALSE,
    [INSTRUMENTED_POP_JUMP_IF_NONE] = INSTRUMENTED_POP_JUMP_IF_NONE,
    [INSTRUMENTED_POP_JUMP_IF_NOT_NONE] = INSTRUMENTED_POP_JUMP_IF_NOT_NONE,
    [INSTRUMENTED_POP_JUMP_IF_TRUE] = INSTRUMENTED_POP_JUMP_IF_TRUE,
    [INSTRUMENTED_RESUME] = INSTRUMENTED_RESUME,
    [INSTRUMENTED_RETURN_CONST] = INSTRUMENTED_RETURN_CONST,
    [INSTRUMENTED_RETURN_VALUE] = INSTRUMENTED_RETURN_VALUE,
    [INSTRUMENTED_YIELD_VALUE] = INSTRUMENTED_YIELD_VALUE,
    [INTERPRETER_EXIT] = INTERPRETER_EXIT,
    [IS_OP] = IS_OP,
    [JUMP_BACKWARD] = JUMP_BACKWARD,
    [JUMP_BACKWARD_NO_INTERRUPT] = JUMP_BACKWARD_NO_INTERRUPT,
    [JUMP_FORWARD] = JUMP_FORWARD,
    [LIST_APPEND] = LIST_APPEND,
    [LIST_EXTEND] = LIST_EXTEND,
    [LOAD_ASSERTION_ERROR] = LOAD_ASSERTION_ERROR,
    [LOAD_ATTR] = LOAD_ATTR,
    [LOAD_ATTR_CLASS] = LOAD_ATTR,
    [LOAD_ATTR_GETATTRIBUTE_OVERRIDDEN] = LOAD_ATTR,
    [LOAD_ATTR_INSTANCE_VALUE] = LOAD_ATTR,
    [LOAD_ATTR_METHOD_LAZY_DICT] = LOAD_ATTR,
    [LOAD_ATTR_METHOD_NO_DICT] = LOAD_ATTR,
    [LOAD_ATTR_METHOD_WITH_VALUES] = LOAD_ATTR,
    [LOAD_ATTR_MODULE] = LOAD_ATTR,
    [LOAD_ATTR_NONDESCRIPTOR_NO_DICT] = LOAD_ATTR,
    [LOAD_ATTR_NONDESCRIPTOR_WITH_VALUES] = LOAD_ATTR,
    [LOAD_ATTR_PROPERTY] = LOAD_ATTR,
    [LOAD_ATTR_SLOT] = LOAD_ATTR,
    [LOAD_ATTR_WITH_HINT] = LOAD_ATTR,
    [LOAD_BUILD_CLASS] = LOAD_BUILD_CLASS,
    [LOAD_CONST] = LOAD_CONST,
    [LOAD_DEREF] = LOAD_DEREF,
    [LOAD_FAST] = LOAD_FAST,
    [LOAD_FAST_AND_CLEAR] = LOAD_FAST_AND_CLEAR,
    [LOAD_FAST_CHECK] = LOAD_FAST_CHECK,
    [LOAD_FAST_LOAD_FAST] = LOAD_FAST_LOAD_FAST,
    [LOAD_FROM_DICT_OR_DEREF] = LOAD_FROM_DICT_OR_DEREF,
    [LOAD_FROM_DICT_OR_GLOBALS] = LOAD_FROM_DICT_OR_GLOBALS,
    [LOAD_GLOBAL] = LOAD_GLOBAL,
    [LOAD_GLOBAL_BUILTIN] = LOAD_GLOBAL,
    [LOAD_GLOBAL_MODULE] = LOAD_GLOBAL,
    [LOAD_LOCALS] = LOAD_LOCALS,
    [LOAD_NAME] = LOAD_NAME,
    [LOAD_SUPER_ATTR] = LOAD_SUPER_ATTR,
    [LOAD_SUPER_ATTR_ATTR] = LOAD_SUPER_ATTR,
    [LOAD_SUPER_ATTR_METHOD] = LOAD_SUPER_ATTR,
    [MAKE_CELL] = MAKE_CELL,
    [MAKE_FUNCTION] = MAKE_FUNCTION,
    [MAP_ADD] = MAP_ADD,
    [MATCH_CLASS] = MATCH_CLASS,
    [MATCH_KEYS] = MATCH_KEYS,
    [MATCH_MAPPING] = MATCH_MAPPING,
    [MATCH_SEQUENCE] = MATCH_SEQUENCE,
    [NOP] = NOP,
    [POP_EXCEPT] = POP_EXCEPT,
    [POP_JUMP_IF_FALSE] = POP_JUMP_IF_FALSE,
    [POP_JUMP_IF_NONE] = POP_JUMP_IF_NONE,
    [POP_JUMP_IF_NOT_NONE] = POP_JUMP_IF_NOT_NONE,
    [POP_JUMP_IF_TRUE] = POP_JUMP_IF_TRUE,
    [POP_TOP] = POP_TOP,
    [PUSH_EXC_INFO] = PUSH_EXC_INFO,
    [PUSH_NULL] = PUSH_NULL,
    [RAISE_VARARGS] = RAISE_VARARGS,
    [RERAISE] = RERAISE,
    [RESERVED] = RESERVED,
    [RESUME] = RESUME,
    [RESUME_CHECK] = RESUME,
    [RETURN_CONST] = RETURN_CONST,
    [RETURN_GENERATOR] = RETURN_GENERATOR,
    [RETURN_VALUE] = RETURN_VALUE,
    [SEND] = SEND,
    [SEND_GEN] = SEND,
    [SETUP_ANNOTATIONS] = SETUP_ANNOTATIONS,
    [SET_ADD] = SET_ADD,
    [SET_FUNCTION_ATTRIBUTE] = SET_FUNCTION_ATTRIBUTE,
    [SET_UPDATE] = SET_UPDATE,
    [STORE_ATTR] = STORE_ATTR,
    [STORE_ATTR_INSTANCE_VALUE] = STORE_ATTR,
    [STORE_ATTR_SLOT] = STORE_ATTR,
    [STORE_ATTR_WITH_HINT] = STORE_ATTR,
    [STORE_DEREF] = STORE_DEREF,
    [STORE_FAST] = STORE_FAST,
    [STORE_FAST_LOAD_FAST] = STORE_FAST_LOAD_FAST,
    [STORE_FAST_STORE_FAST] = STORE_FAST_STORE_FAST,
    [STORE_GLOBAL] = STORE_GLOBAL,
    [STORE_NAME] = STORE_NAME,
    [STORE_SLICE] = STORE_SLICE,
    [STORE_SUBSCR] = STORE_SUBSCR,
    [STORE_SUBSCR_DICT] = STORE_SUBSCR,
    [STORE_SUBSCR_LIST_INT] = STORE_SUBSCR,
    [SWAP] = SWAP,
    [TO_BOOL] = TO_BOOL,
    [TO_BOOL_ALWAYS_TRUE] = TO_BOOL,
    [TO_BOOL_BOOL] = TO_BOOL,
    [TO_BOOL_INT] = TO_BOOL,
    [TO_BOOL_LIST] = TO_BOOL,
    [TO_BOOL_NONE] = TO_BOOL,
    [TO_BOOL_STR] = TO_BOOL,
    [UNARY_INVERT] = UNARY_INVERT,
    [UNARY_NEGATIVE] = UNARY_NEGATIVE,
    [UNARY_NOT] = UNARY_NOT,
    [UNPACK_EX] = UNPACK_EX,
    [UNPACK_SEQUENCE] = UNPACK_SEQUENCE,
    [UNPACK_SEQUENCE_LIST] = UNPACK_SEQUENCE,
    [UNPACK_SEQUENCE_TUPLE] = UNPACK_SEQUENCE,
    [UNPACK_SEQUENCE_TWO_TUPLE] = UNPACK_SEQUENCE,
    [WITH_EXCEPT_START] = WITH_EXCEPT_START,
    [YIELD_VALUE] = YIELD_VALUE,
};

// from Python/instrumentation.c
static const uint8_t _ch_DE_INSTRUMENT[256] = {
    [INSTRUMENTED_RESUME] = RESUME,
    [INSTRUMENTED_RETURN_VALUE] = RETURN_VALUE,
    [INSTRUMENTED_RETURN_CONST] = RETURN_CONST,
    [INSTRUMENTED_CALL] = CALL,
    [INSTRUMENTED_CALL_KW] = CALL_KW,
    [INSTRUMENTED_CALL_FUNCTION_EX] = CALL_FUNCTION_EX,
    [INSTRUMENTED_YIELD_VALUE] = YIELD_VALUE,
    [INSTRUMENTED_JUMP_FORWARD] = JUMP_FORWARD,
    [INSTRUMENTED_JUMP_BACKWARD] = JUMP_BACKWARD,
    [INSTRUMENTED_POP_JUMP_IF_FALSE] = POP_JUMP_IF_FALSE,
    [INSTRUMENTED_POP_JUMP_IF_TRUE] = POP_JUMP_IF_TRUE,
    [INSTRUMENTED_POP_JUMP_IF_NONE] = POP_JUMP_IF_NONE,
    [INSTRUMENTED_POP_JUMP_IF_NOT_NONE] = POP_JUMP_IF_NOT_NONE,
    [INSTRUMENTED_FOR_ITER] = FOR_ITER,
    [INSTRUMENTED_END_FOR] = END_FOR,
    [INSTRUMENTED_END_SEND] = END_SEND,
    [INSTRUMENTED_LOAD_SUPER_ATTR] = LOAD_SUPER_ATTR,
};

#endif
#endif

#if PY_VERSION_HEX >= 0x030C0000
#if PY_VERSION_HEX < 0x030D0000
//    ===========
//    Python 3.12
//    ===========

const uint8_t _ch_PyOpcode_Caches[256] = {
    [BINARY_SUBSCR] = 1,
    [STORE_SUBSCR] = 1,
    [UNPACK_SEQUENCE] = 1,
    [FOR_ITER] = 1,
    [STORE_ATTR] = 4,
    [LOAD_ATTR] = 9,
    [COMPARE_OP] = 1,
    [LOAD_GLOBAL] = 4,
    [BINARY_OP] = 1,
    [SEND] = 1,
    [LOAD_SUPER_ATTR] = 1,
    [CALL] = 3,
};

const uint8_t _ch_PyOpcode_Deopt[256] = {
    [BEFORE_ASYNC_WITH] = BEFORE_ASYNC_WITH,
    [BEFORE_WITH] = BEFORE_WITH,
    [BINARY_OP] = BINARY_OP,
    [BINARY_OP_ADD_FLOAT] = BINARY_OP,
    [BINARY_OP_ADD_INT] = BINARY_OP,
    [BINARY_OP_ADD_UNICODE] = BINARY_OP,
    [BINARY_OP_INPLACE_ADD_UNICODE] = BINARY_OP,
    [BINARY_OP_MULTIPLY_FLOAT] = BINARY_OP,
    [BINARY_OP_MULTIPLY_INT] = BINARY_OP,
    [BINARY_OP_SUBTRACT_FLOAT] = BINARY_OP,
    [BINARY_OP_SUBTRACT_INT] = BINARY_OP,
    [BINARY_SLICE] = BINARY_SLICE,
    [BINARY_SUBSCR] = BINARY_SUBSCR,
    [BINARY_SUBSCR_DICT] = BINARY_SUBSCR,
    [BINARY_SUBSCR_GETITEM] = BINARY_SUBSCR,
    [BINARY_SUBSCR_LIST_INT] = BINARY_SUBSCR,
    [BINARY_SUBSCR_TUPLE_INT] = BINARY_SUBSCR,
    [BUILD_CONST_KEY_MAP] = BUILD_CONST_KEY_MAP,
    [BUILD_LIST] = BUILD_LIST,
    [BUILD_MAP] = BUILD_MAP,
    [BUILD_SET] = BUILD_SET,
    [BUILD_SLICE] = BUILD_SLICE,
    [BUILD_STRING] = BUILD_STRING,
    [BUILD_TUPLE] = BUILD_TUPLE,
    [CACHE] = CACHE,
    [CALL] = CALL,
    [CALL_BOUND_METHOD_EXACT_ARGS] = CALL,
    [CALL_BUILTIN_CLASS] = CALL,
    [CALL_BUILTIN_FAST_WITH_KEYWORDS] = CALL,
    [CALL_FUNCTION_EX] = CALL_FUNCTION_EX,
    [CALL_INTRINSIC_1] = CALL_INTRINSIC_1,
    [CALL_INTRINSIC_2] = CALL_INTRINSIC_2,
    [CALL_METHOD_DESCRIPTOR_FAST_WITH_KEYWORDS] = CALL,
    [CALL_NO_KW_BUILTIN_FAST] = CALL,
    [CALL_NO_KW_BUILTIN_O] = CALL,
    [CALL_NO_KW_ISINSTANCE] = CALL,
    [CALL_NO_KW_LEN] = CALL,
    [CALL_NO_KW_LIST_APPEND] = CALL,
    [CALL_NO_KW_METHOD_DESCRIPTOR_FAST] = CALL,
    [CALL_NO_KW_METHOD_DESCRIPTOR_NOARGS] = CALL,
    [CALL_NO_KW_METHOD_DESCRIPTOR_O] = CALL,
    [CALL_NO_KW_STR_1] = CALL,
    [CALL_NO_KW_TUPLE_1] = CALL,
    [CALL_NO_KW_TYPE_1] = CALL,
    [CALL_PY_EXACT_ARGS] = CALL,
    [CALL_PY_WITH_DEFAULTS] = CALL,
    [CHECK_EG_MATCH] = CHECK_EG_MATCH,
    [CHECK_EXC_MATCH] = CHECK_EXC_MATCH,
    [CLEANUP_THROW] = CLEANUP_THROW,
    [COMPARE_OP] = COMPARE_OP,
    [COMPARE_OP_FLOAT] = COMPARE_OP,
    [COMPARE_OP_INT] = COMPARE_OP,
    [COMPARE_OP_STR] = COMPARE_OP,
    [CONTAINS_OP] = CONTAINS_OP,
    [COPY] = COPY,
    [COPY_FREE_VARS] = COPY_FREE_VARS,
    [DELETE_ATTR] = DELETE_ATTR,
    [DELETE_DEREF] = DELETE_DEREF,
    [DELETE_FAST] = DELETE_FAST,
    [DELETE_GLOBAL] = DELETE_GLOBAL,
    [DELETE_NAME] = DELETE_NAME,
    [DELETE_SUBSCR] = DELETE_SUBSCR,
    [DICT_MERGE] = DICT_MERGE,
    [DICT_UPDATE] = DICT_UPDATE,
    [END_ASYNC_FOR] = END_ASYNC_FOR,
    [END_FOR] = END_FOR,
    [END_SEND] = END_SEND,
    [EXTENDED_ARG] = EXTENDED_ARG,
    [FORMAT_VALUE] = FORMAT_VALUE,
    [FOR_ITER] = FOR_ITER,
    [FOR_ITER_GEN] = FOR_ITER,
    [FOR_ITER_LIST] = FOR_ITER,
    [FOR_ITER_RANGE] = FOR_ITER,
    [FOR_ITER_TUPLE] = FOR_ITER,
    [GET_AITER] = GET_AITER,
    [GET_ANEXT] = GET_ANEXT,
    [GET_AWAITABLE] = GET_AWAITABLE,
    [GET_ITER] = GET_ITER,
    [GET_LEN] = GET_LEN,
    [GET_YIELD_FROM_ITER] = GET_YIELD_FROM_ITER,
    [IMPORT_FROM] = IMPORT_FROM,
    [IMPORT_NAME] = IMPORT_NAME,
    [INSTRUMENTED_CALL] = INSTRUMENTED_CALL,
    [INSTRUMENTED_CALL_FUNCTION_EX] = INSTRUMENTED_CALL_FUNCTION_EX,
    [INSTRUMENTED_END_FOR] = INSTRUMENTED_END_FOR,
    [INSTRUMENTED_END_SEND] = INSTRUMENTED_END_SEND,
    [INSTRUMENTED_FOR_ITER] = INSTRUMENTED_FOR_ITER,
    [INSTRUMENTED_INSTRUCTION] = INSTRUMENTED_INSTRUCTION,
    [INSTRUMENTED_JUMP_BACKWARD] = INSTRUMENTED_JUMP_BACKWARD,
    [INSTRUMENTED_JUMP_FORWARD] = INSTRUMENTED_JUMP_FORWARD,
    [INSTRUMENTED_LINE] = INSTRUMENTED_LINE,
    [INSTRUMENTED_LOAD_SUPER_ATTR] = INSTRUMENTED_LOAD_SUPER_ATTR,
    [INSTRUMENTED_POP_JUMP_IF_FALSE] = INSTRUMENTED_POP_JUMP_IF_FALSE,
    [INSTRUMENTED_POP_JUMP_IF_NONE] = INSTRUMENTED_POP_JUMP_IF_NONE,
    [INSTRUMENTED_POP_JUMP_IF_NOT_NONE] = INSTRUMENTED_POP_JUMP_IF_NOT_NONE,
    [INSTRUMENTED_POP_JUMP_IF_TRUE] = INSTRUMENTED_POP_JUMP_IF_TRUE,
    [INSTRUMENTED_RESUME] = INSTRUMENTED_RESUME,
    [INSTRUMENTED_RETURN_CONST] = INSTRUMENTED_RETURN_CONST,
    [INSTRUMENTED_RETURN_VALUE] = INSTRUMENTED_RETURN_VALUE,
    [INSTRUMENTED_YIELD_VALUE] = INSTRUMENTED_YIELD_VALUE,
    [INTERPRETER_EXIT] = INTERPRETER_EXIT,
    [IS_OP] = IS_OP,
    [JUMP_BACKWARD] = JUMP_BACKWARD,
    [JUMP_BACKWARD_NO_INTERRUPT] = JUMP_BACKWARD_NO_INTERRUPT,
    [JUMP_FORWARD] = JUMP_FORWARD,
    [KW_NAMES] = KW_NAMES,
    [LIST_APPEND] = LIST_APPEND,
    [LIST_EXTEND] = LIST_EXTEND,
    [LOAD_ASSERTION_ERROR] = LOAD_ASSERTION_ERROR,
    [LOAD_ATTR] = LOAD_ATTR,
    [LOAD_ATTR_CLASS] = LOAD_ATTR,
    [LOAD_ATTR_GETATTRIBUTE_OVERRIDDEN] = LOAD_ATTR,
    [LOAD_ATTR_INSTANCE_VALUE] = LOAD_ATTR,
    [LOAD_ATTR_METHOD_LAZY_DICT] = LOAD_ATTR,
    [LOAD_ATTR_METHOD_NO_DICT] = LOAD_ATTR,
    [LOAD_ATTR_METHOD_WITH_VALUES] = LOAD_ATTR,
    [LOAD_ATTR_MODULE] = LOAD_ATTR,
    [LOAD_ATTR_PROPERTY] = LOAD_ATTR,
    [LOAD_ATTR_SLOT] = LOAD_ATTR,
    [LOAD_ATTR_WITH_HINT] = LOAD_ATTR,
    [LOAD_BUILD_CLASS] = LOAD_BUILD_CLASS,
    [LOAD_CLOSURE] = LOAD_CLOSURE,
    [LOAD_CONST] = LOAD_CONST,
    [LOAD_CONST__LOAD_FAST] = LOAD_CONST,
    [LOAD_DEREF] = LOAD_DEREF,
    [LOAD_FAST] = LOAD_FAST,
    [LOAD_FAST_AND_CLEAR] = LOAD_FAST_AND_CLEAR,
    [LOAD_FAST_CHECK] = LOAD_FAST_CHECK,
    [LOAD_FAST__LOAD_CONST] = LOAD_FAST,
    [LOAD_FAST__LOAD_FAST] = LOAD_FAST,
    [LOAD_FROM_DICT_OR_DEREF] = LOAD_FROM_DICT_OR_DEREF,
    [LOAD_FROM_DICT_OR_GLOBALS] = LOAD_FROM_DICT_OR_GLOBALS,
    [LOAD_GLOBAL] = LOAD_GLOBAL,
    [LOAD_GLOBAL_BUILTIN] = LOAD_GLOBAL,
    [LOAD_GLOBAL_MODULE] = LOAD_GLOBAL,
    [LOAD_LOCALS] = LOAD_LOCALS,
    [LOAD_NAME] = LOAD_NAME,
    [LOAD_SUPER_ATTR] = LOAD_SUPER_ATTR,
    [LOAD_SUPER_ATTR_ATTR] = LOAD_SUPER_ATTR,
    [LOAD_SUPER_ATTR_METHOD] = LOAD_SUPER_ATTR,
    [MAKE_CELL] = MAKE_CELL,
    [MAKE_FUNCTION] = MAKE_FUNCTION,
    [MAP_ADD] = MAP_ADD,
    [MATCH_CLASS] = MATCH_CLASS,
    [MATCH_KEYS] = MATCH_KEYS,
    [MATCH_MAPPING] = MATCH_MAPPING,
    [MATCH_SEQUENCE] = MATCH_SEQUENCE,
    [NOP] = NOP,
    [POP_EXCEPT] = POP_EXCEPT,
    [POP_JUMP_IF_FALSE] = POP_JUMP_IF_FALSE,
    [POP_JUMP_IF_NONE] = POP_JUMP_IF_NONE,
    [POP_JUMP_IF_NOT_NONE] = POP_JUMP_IF_NOT_NONE,
    [POP_JUMP_IF_TRUE] = POP_JUMP_IF_TRUE,
    [POP_TOP] = POP_TOP,
    [PUSH_EXC_INFO] = PUSH_EXC_INFO,
    [PUSH_NULL] = PUSH_NULL,
    [RAISE_VARARGS] = RAISE_VARARGS,
    [RERAISE] = RERAISE,
    [RESERVED] = RESERVED,
    [RESUME] = RESUME,
    [RETURN_CONST] = RETURN_CONST,
    [RETURN_GENERATOR] = RETURN_GENERATOR,
    [RETURN_VALUE] = RETURN_VALUE,
    [SEND] = SEND,
    [SEND_GEN] = SEND,
    [SETUP_ANNOTATIONS] = SETUP_ANNOTATIONS,
    [SET_ADD] = SET_ADD,
    [SET_UPDATE] = SET_UPDATE,
    [STORE_ATTR] = STORE_ATTR,
    [STORE_ATTR_INSTANCE_VALUE] = STORE_ATTR,
    [STORE_ATTR_SLOT] = STORE_ATTR,
    [STORE_ATTR_WITH_HINT] = STORE_ATTR,
    [STORE_DEREF] = STORE_DEREF,
    [STORE_FAST] = STORE_FAST,
    [STORE_FAST__LOAD_FAST] = STORE_FAST,
    [STORE_FAST__STORE_FAST] = STORE_FAST,
    [STORE_GLOBAL] = STORE_GLOBAL,
    [STORE_NAME] = STORE_NAME,
    [STORE_SLICE] = STORE_SLICE,
    [STORE_SUBSCR] = STORE_SUBSCR,
    [STORE_SUBSCR_DICT] = STORE_SUBSCR,
    [STORE_SUBSCR_LIST_INT] = STORE_SUBSCR,
    [SWAP] = SWAP,
    [UNARY_INVERT] = UNARY_INVERT,
    [UNARY_NEGATIVE] = UNARY_NEGATIVE,
    [UNARY_NOT] = UNARY_NOT,
    [UNPACK_EX] = UNPACK_EX,
    [UNPACK_SEQUENCE] = UNPACK_SEQUENCE,
    [UNPACK_SEQUENCE_LIST] = UNPACK_SEQUENCE,
    [UNPACK_SEQUENCE_TUPLE] = UNPACK_SEQUENCE,
    [UNPACK_SEQUENCE_TWO_TUPLE] = UNPACK_SEQUENCE,
    [WITH_EXCEPT_START] = WITH_EXCEPT_START,
    [YIELD_VALUE] = YIELD_VALUE,
};

static const uint8_t _ch_DE_INSTRUMENT[256] = {
    [INSTRUMENTED_RESUME] = RESUME,
    [INSTRUMENTED_RETURN_VALUE] = RETURN_VALUE,
    [INSTRUMENTED_RETURN_CONST] = RETURN_CONST,
    [INSTRUMENTED_CALL] = CALL,
    [INSTRUMENTED_CALL_FUNCTION_EX] = CALL_FUNCTION_EX,
    [INSTRUMENTED_YIELD_VALUE] = YIELD_VALUE,
    [INSTRUMENTED_JUMP_FORWARD] = JUMP_FORWARD,
    [INSTRUMENTED_JUMP_BACKWARD] = JUMP_BACKWARD,
    [INSTRUMENTED_POP_JUMP_IF_FALSE] = POP_JUMP_IF_FALSE,
    [INSTRUMENTED_POP_JUMP_IF_TRUE] = POP_JUMP_IF_TRUE,
    [INSTRUMENTED_POP_JUMP_IF_NONE] = POP_JUMP_IF_NONE,
    [INSTRUMENTED_POP_JUMP_IF_NOT_NONE] = POP_JUMP_IF_NOT_NONE,
    [INSTRUMENTED_FOR_ITER] = FOR_ITER,
    [INSTRUMENTED_END_FOR] = END_FOR,
    [INSTRUMENTED_END_SEND] = END_SEND,
    [INSTRUMENTED_LOAD_SUPER_ATTR] = LOAD_SUPER_ATTR,
};

#endif
#endif

static const uint8_t _ch_TRACABLE_INSTRUCTIONS[256] = {
    // This must be manually kept in sync the the various
    // instructions that we care about on the python side.
    [MAP_ADD] = 1,
    [BINARY_SUBSCR] = 1,
    [BINARY_SLICE] = 1,
    [CONTAINS_OP] = 1,
    [BUILD_STRING] = 1,
    [FORMAT_VALUE] = 1,
    [UNARY_NOT] = 1,
    [SET_ADD] = 1,
    [IS_OP] = 1,
    [BINARY_OP] = 1,
    [CALL] = 1,
    [CALL_FUNCTION_EX] = 1,
};


/* Get the underlying opcode, stripping instrumentation */
int _ch_Py_GetBaseOpcode(PyCodeObject *code, int i)
{
    int opcode = _PyCode_CODE(code)[i].op.code;
    if (opcode == INSTRUMENTED_LINE) {
        opcode = code->_co_monitoring->lines[i].original_opcode;
    }
    if (opcode == INSTRUMENTED_INSTRUCTION) {
        opcode = code->_co_monitoring->per_instruction_opcodes[i];
    }
    int deinstrumented = _ch_DE_INSTRUMENT[opcode];
    if (deinstrumented) {
        return deinstrumented;
    }
    return _ch_PyOpcode_Deopt[opcode];
}

static int64_t *
_ch_mark_stacks(PyCodeObject *code_obj, int len)
{
    // This differs from the corresponding function in CPython in a few ways:
    // 1) The returned stack values represent stack depths, not stack content types.
    // 2) The low bit of each depth is a flag that indicates whether this is an
    // instruction that we trace, or it could follow an instruction that we trace.
    PyObject *co_code = PyCode_GetCode(code_obj);
    // printf("co_code %d\n", co_code);
    if (co_code == NULL) {
        return NULL;
    }
    _Py_CODEUNIT *code = (_Py_CODEUNIT *)PyBytes_AS_STRING(co_code);
    int64_t *stacks = PyMem_New(int64_t, len+1);
    if (stacks == NULL) {
        PyErr_NoMemory();
        Py_DECREF(co_code);
        return NULL;
    }
    uint8_t *enabled_tracing = PyMem_New(uint8_t, len+1);
    if (enabled_tracing == NULL) {
        PyErr_NoMemory();
        PyMem_Free(stacks);
        Py_DECREF(co_code);
        return NULL;
    }
    int i, j, opcode;

    for (int i = 1; i <= len; i++) {
        stacks[i] = UNINITIALIZED;
        enabled_tracing[i] = 0;
    }
    stacks[0] = EMPTY_STACK;
#if PY_VERSION_HEX < 0x030D0000
    // Python 3.12
    if (code_obj->co_flags & (CO_GENERATOR | CO_COROUTINE | CO_ASYNC_GENERATOR))
    {
        // Generators get sent None while starting:
        stacks[0]++;
    }
#endif
    int todo = 1;
    while (todo) {
        todo = 0;
        /* Scan instructions */
        for (i = 0; i < len;) {
            int64_t next_stack = stacks[i];
            opcode = _ch_Py_GetBaseOpcode(code_obj, i);
            uint8_t trace_enabled_here = _ch_TRACABLE_INSTRUCTIONS[opcode];
            enabled_tracing[i] |= trace_enabled_here;
            int oparg = 0;
            while (opcode == EXTENDED_ARG) {
                oparg = (oparg << 8) | code[i].op.arg;
                i++;
                opcode = _ch_Py_GetBaseOpcode(code_obj, i);
                stacks[i] = next_stack;
            }
            int next_i = i + _ch_PyOpcode_Caches[opcode] + 1;
            if (next_stack == UNINITIALIZED) {
                i = next_i;
                continue;
            }
            oparg = (oparg << 8) | code[i].op.arg;
            // printf("at %d opcode %d oparg %d priorstack %d\n", i, opcode, oparg, next_stack);
            switch (opcode) {
                case POP_JUMP_IF_FALSE:
                case POP_JUMP_IF_TRUE:
                case POP_JUMP_IF_NONE:
                case POP_JUMP_IF_NOT_NONE:
                {
                    int64_t target_stack;
                    int j = next_i + oparg;
                    assert(j < len);
                    next_stack--;
                    target_stack = next_stack;
                    assert(stacks[j] == UNINITIALIZED || stacks[j] == target_stack);
                    stacks[j] = target_stack;
                    enabled_tracing[j] |= trace_enabled_here;
                    stacks[next_i] = next_stack;
                    break;
                }
                case SEND:
                    j = oparg + i + INLINE_CACHE_ENTRIES_SEND + 1;
                    assert(j < len);
                    assert(stacks[j] == UNINITIALIZED || stacks[j] == next_stack);
                    stacks[j] = next_stack;
                    enabled_tracing[j] |= trace_enabled_here;
                    stacks[next_i] = next_stack;
                    break;
                case JUMP_FORWARD:
                    j = oparg + i + 1;
                    assert(j < len);
                    assert(stacks[j] == UNINITIALIZED || stacks[j] == next_stack);
                    stacks[j] = next_stack;
                    enabled_tracing[j] |= trace_enabled_here;
                    break;
                case JUMP_BACKWARD:
                case JUMP_BACKWARD_NO_INTERRUPT:
                    j = next_i - oparg;
                    assert(j >= 0);
                    assert(j < len);
                    if (stacks[j] == UNINITIALIZED && j < i) {
                        todo = 1;
                    }
                    assert(stacks[j] == UNINITIALIZED || stacks[j] == next_stack);
                    stacks[j] = next_stack;
                    enabled_tracing[j] |= trace_enabled_here;
                    break;
                case GET_ITER:
                case GET_AITER:
                    stacks[next_i] = next_stack;
                    break;
                case FOR_ITER:
                {
                    int64_t target_stack = next_stack + 1;
                    stacks[next_i] = target_stack;
                    j = oparg + 1 + INLINE_CACHE_ENTRIES_FOR_ITER + i;
                    assert(j < len);
                    assert(stacks[j] == UNINITIALIZED || stacks[j] == target_stack);
                    stacks[j] = target_stack;
                    enabled_tracing[j] |= trace_enabled_here;
                    break;
                }
                case END_ASYNC_FOR:
#if PY_VERSION_HEX < 0x030D0000
                   // Python 3.12
                    next_stack--;
#else
                   // Python 3.13+
                    next_stack--;
                    next_stack--;
#endif
                    stacks[next_i] = next_stack;
                    break;
                case PUSH_EXC_INFO:
                    next_stack++;
                    stacks[next_i] = next_stack;
                    break;
                case POP_EXCEPT:
                    next_stack--;
                    stacks[next_i] = next_stack;
                    break;
                case RETURN_VALUE:
                    break;
                case RETURN_CONST:
                    break;
                case RAISE_VARARGS:
                    break;
                case RERAISE:
                    /* End of block */
                    break;
                case PUSH_NULL:
                    next_stack++;
                    stacks[next_i] = next_stack;
                    break;
                case LOAD_GLOBAL:
                {
                    int j = oparg;
                    next_stack++;
                    if (j & 1) {
                        next_stack++;
                    }
                    stacks[next_i] = next_stack;
                    break;
                }
                case LOAD_ATTR:
                {
                    int j = oparg;
                    if (j & 1) {
                        next_stack--;
                        next_stack++;
                        next_stack++;
                    }
                    stacks[next_i] = next_stack;
                    break;
                }
                case SWAP:
                {
                    stacks[next_i] = next_stack;
                    break;
                }
                case COPY:
                {
                    next_stack++;
                    stacks[next_i] = next_stack;
                    break;
                }
                case CACHE:
                case RESERVED:
                {
                    assert(0);
                }
                default:
                {
                    int delta = PyCompile_OpcodeStackEffect(opcode, oparg);
                    assert(delta != PY_INVALID_STACK_EFFECT);
                    while (delta < 0) {
                        next_stack--;
                        delta++;
                    }
                    while (delta > 0) {
                        next_stack++;
                        delta--;
                    }
                    stacks[next_i] = next_stack;
                }
            }
            enabled_tracing[next_i] |= trace_enabled_here;
            i = next_i;
        }
        /* Scan exception table */
        unsigned char *start = (unsigned char *)PyBytes_AS_STRING(code_obj->co_exceptiontable);
        unsigned char *end = start + PyBytes_GET_SIZE(code_obj->co_exceptiontable);
        unsigned char *scan = start;
        while (scan < end) {
            int start_offset, size, handler;
            scan = parse_varint(scan, &start_offset);
            assert(start_offset >= 0 && start_offset < len);
            scan = parse_varint(scan, &size);
            assert(size >= 0 && start_offset+size <= len);
            scan = parse_varint(scan, &handler);
            assert(handler >= 0 && handler < len);
            int depth_and_lasti;
            scan = parse_varint(scan, &depth_and_lasti);
            int level = depth_and_lasti >> 1;
            int lasti = depth_and_lasti & 1;
            // printf("scan %d, end %d, start_offset %d, size %d, handler %d, level %d, lasti %d\n",
            //     scan, end, start_offset, size, handler, level, lasti);
            if (stacks[start_offset] != UNINITIALIZED) {
                if (stacks[handler] == UNINITIALIZED) {
                    todo = 1;
                    uint64_t target_stack = _ch_pop_to_level(stacks[start_offset], level);
                    if (lasti) {
                        target_stack++;
                    }
                    target_stack++;
                    stacks[handler] = target_stack;
                }
            }
        }
    }
    Py_DECREF(co_code);
    for (i = 0; i < len; i++) {
        if (stacks[i] >= 0) {
            stacks[i] = (stacks[i] << 1) | enabled_tracing[i];
        }
    }
    PyMem_Free(enabled_tracing);
    return stacks;
}
