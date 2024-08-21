
/*
 * Portions of this source was derived from the amazing coverage.py project;
 * specifically this file:
 * https://github.com/nedbat/coveragepy/blob/f0f4761a459e1601c5b0c1043db5895e31c66e80/coverage/ctracer/tracer.c
 *
 * The shared code portions are licensed under the Apache License:
 *   http://www.apache.org/licenses/LICENSE-2.0
 *   https://github.com/nedbat/coveragepy/blob/master/NOTICE.txt
 *
 * See the "LICENSE" file for complete license details on CrossHair.
*/


#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <opcode.h>

#define Py_BUILD_CORE

#if PY_VERSION_HEX >= 0x030C0000
#include "_mark_stacks.h"
#endif

#include "_tracers_pycompat.h"
#include "_tracers.h"

#include "frameobject.h"

#if PY_VERSION_HEX >= 0x030B0000
#include "internal/pycore_frame.h"
#endif

static int
pyint_as_int(PyObject * pyint, int *pint)
{
    int the_int = (int)PyLong_AsLong(pyint);
    if (the_int == -1 && PyErr_Occurred()) {
        return RET_ERROR;
    }

    *pint = the_int;
    return RET_OK;
}


static void trace_frame(PyFrameObject *frame)
{
#if PY_VERSION_HEX >= 0x03000000
    PyObject_SetAttrString((PyObject*)frame, "f_trace_opcodes", Py_True);
    PyObject_SetAttrString((PyObject*)frame, "f_trace_lines", Py_False);
#else
    frame->f_trace_lines = 0;
    frame->f_trace_opcodes = 1;
#endif
}

static void dont_trace_frame(PyFrameObject *frame)
{
#if PY_VERSION_HEX >= 0x03000000
    PyObject_SetAttrString((PyObject*)frame, "f_trace_opcodes", Py_False);
    PyObject_SetAttrString((PyObject*)frame, "f_trace_lines", Py_False);
#else
    frame->f_trace = NULL;
    frame->f_trace_lines = 0;
    frame->f_trace_opcodes = 0;
#endif
}

static int
CTracer_init(CTracer *self, PyObject *args_unused, PyObject *kwds_unused)
{
    init_framecbvec(&self->postop_callbacks, 5);
    init_modulevec(&self->modules, 5);
    init_tablevec(&self->handlers, 3);
    self->enabled = FALSE;
    self->handling = FALSE;
    self->trace_all_opcodes = FALSE;
    return RET_OK;
}

static void
CTracer_dealloc(CTracer *self)
{
    ModuleVec* modules = &self->modules;
    for(int i=0; i< modules->count; i++) {
        Py_DECREF(modules->items[i]);
    }
    PyMem_Free(self->postop_callbacks.items);
    PyMem_Free(self->modules.items);
    PyMem_Free(self->handlers.items);
    Py_TYPE(self)->tp_free((PyObject*)self);
}

#if PY_VERSION_HEX >= 0x030C0000
// Python 3.12

#define _CODE_STACK_CACHE_CAPACITY 64
static CodeAndStacks _CODE_STACK_CACHE[_CODE_STACK_CACHE_CAPACITY];
static int _CODE_STACK_CACHE_SIZE = 0;

static int64_t *
_ch_get_stacks(PyCodeObject *code_obj)
{
    // // To disable stack cache (NOTE: leaks stack data):
    // int codelen = (int)Py_SIZE(code_obj);
    // return _ch_mark_stacks(code_obj, codelen);

    int entry_pos = 0;
    CodeAndStacks entry = {NULL, NULL};
    for(; entry_pos < _CODE_STACK_CACHE_SIZE; entry_pos++) {
        entry = _CODE_STACK_CACHE[entry_pos];
        if (entry.code_obj == code_obj) {
            break;
        }
    }

    if (entry_pos == _CODE_STACK_CACHE_SIZE) {
        if (_CODE_STACK_CACHE_SIZE == _CODE_STACK_CACHE_CAPACITY) {
            // Purge the last entry:
            entry_pos = _CODE_STACK_CACHE_CAPACITY - 1;
            CodeAndStacks todelete = _CODE_STACK_CACHE[entry_pos];
            PyMem_Free(todelete.stacks);
            Py_DECREF(todelete.code_obj);
            _CODE_STACK_CACHE_SIZE = _CODE_STACK_CACHE_CAPACITY - 1;
        }
        int codelen = (int)Py_SIZE(code_obj);
        entry = (CodeAndStacks){code_obj, _ch_mark_stacks(code_obj, codelen)};
        _CODE_STACK_CACHE[_CODE_STACK_CACHE_SIZE++] = entry;
        Py_INCREF(code_obj);
    }

    if (entry_pos > 0) {
        // LRU-like behavior; swap a hit with a (randomized) earlier entry:
        int mid = rand() % entry_pos;
        _CODE_STACK_CACHE[entry_pos] = _CODE_STACK_CACHE[mid];
        _CODE_STACK_CACHE[mid] = entry;
    }
    return entry.stacks;
}

#endif

static PyObject *
CTracer_push_module(CTracer *self, PyObject *args)
{
    PyObject *tracing_module;
    if (!PyArg_ParseTuple(args, "O", &tracing_module)) {
        return NULL;
    }
    Py_INCREF(tracing_module);
    push_module(&self->modules, tracing_module);
    TableVec* tables = &self->handlers;

    PyObject* wanted = PyObject_GetAttrString(tracing_module, "opcodes_wanted");
    if (wanted == NULL || !PyFrozenSet_Check(wanted))
    {
        PyErr_SetString(PyExc_TypeError, "opcodes_wanted must be frozenset instance");
        return NULL;
    }
    PyObject* wanted_itr = PyObject_GetIter(wanted);
    if (wanted_itr == NULL)
    {
        return NULL;
    }
    PyObject* wanted_item = NULL;
    while((wanted_item=PyIter_Next(wanted_itr)))
    {
        int opcode;
        if (pyint_as_int(wanted_item, &opcode) == RET_ERROR)
        {
            Py_DECREF(wanted_item);
            printf("WARNING: Non-integer found in wanted_opcodes; ignoring\n");
            PyErr_Clear();
            continue;
        }
        Py_DECREF(wanted_item);
        if (opcode < 0 || opcode >=256)
        {
            if (opcode != 256) {  // 256 is used as an explicit ignore marker
                printf("WARNING: out-of-range opcode found in wanted_opcodes; ignoring\n");
            }
            continue;
        }
#if PY_VERSION_HEX >= 0x030C0000
// Python 3.12
        if (! _ch_TRACABLE_INSTRUCTIONS[opcode]) {
            self->trace_all_opcodes = TRUE;
            // sys.monitoring also will need to be reset, but that happens at the python layer above
        }
#endif
        for(int table_idx=0; ; table_idx++)
        {
            if (table_idx >= tables->count) {
                HandlerTable newtable = {.entries = {0}};
                push_table_entry(tables, newtable);
            }
            HandlerTable * table = &(tables->items[table_idx]);
            if (table->entries[opcode] == NULL)
            {
                table->entries[opcode] = tracing_module;
                break;
            }
        }
    }
    Py_DECREF(wanted_itr);
    if (PyErr_Occurred()) {
        return NULL;
    }
    Py_RETURN_NONE;
}

static PyObject *
CTracer_pop_module(CTracer *self, PyObject *args)
{
    PyObject *module;
    if (!PyArg_ParseTuple(args, "O", &module)) {
        return NULL;
    }
    ModuleVec* modules = &self->modules;
    if (modules->count < 1) {
        PyErr_SetString(PyExc_ValueError, "No tracing modules are installed");
        return NULL;
    }
    if (module != modules->items[modules->count - 1]) {
        PyErr_SetString(PyExc_ValueError, "Tracing module poped out-of-order");
        return NULL;
    }
    modules->count--;
    Py_XDECREF(module);

    TableVec* tables = &self->handlers;
    for(int table_idx = 0; table_idx < self->handlers.count; table_idx++) {
        for(int opcode = 0; opcode < 256; opcode++) {
            if (tables->items[table_idx].entries[opcode] == module) {
                tables->items[table_idx].entries[opcode] = NULL;
            }
        }
    }

#if PY_VERSION_HEX >= 0x030C0000
// Python 3.12
    if (self->trace_all_opcodes) {
        BOOL continue_to_trace_all_opcodes = FALSE;
        HandlerTable top_table = tables->items[0];
        for(int opcode = 0; opcode < 256; opcode++) {
            if (top_table.entries[opcode] != NULL && ! _ch_TRACABLE_INSTRUCTIONS[opcode]) {
                continue_to_trace_all_opcodes = TRUE;
                break;
            }
        }
        self->trace_all_opcodes = continue_to_trace_all_opcodes;
        // sys.monitoring may need to be reset, but that happens at the python layer above
        // TODO: if we move this into the c layer, we wouldn't have to reset monitoring so much.
    }
#endif

    Py_RETURN_NONE;
}


static PyObject *
CTracer_get_modules(CTracer *self, PyObject *unused_args)
{
    ModuleVec* modules = &self->modules;
    int count = modules->count;
    PyObject* python_val = PyList_New(count);
    for (int i = 0; i < count; ++i)
    {
        PyObject* module = Py_BuildValue("O", modules->items[i]);
        PyList_SetItem(python_val, i, module);
    }
    return python_val;
}


/*
 * Parts of the trace function.
 */

static void repr_print(PyObject *obj) {
    PyObject* repr = PyObject_Repr(obj);
    PyObject* str = PyUnicode_AsEncodedString(repr, "utf-8", "~E~");
    const char *bytes = PyBytes_AS_STRING(str);

    printf("REPR: %s\n", bytes);

    Py_XDECREF(repr);
    Py_XDECREF(str);
}

static int
CTracer_handle_opcode(CTracer *self, PyCodeObject* pCode, int lasti)
{

#if PY_VERSION_HEX >= 0x030C0000
// Python 3.12
    if (!self->trace_all_opcodes) {
        int64_t *stacks = _ch_get_stacks(pCode);
        uint8_t at_enabled_position = stacks[lasti / 2] & 1;
        if (!at_enabled_position) {
            return RET_DISABLE_TRACING;
        }
    }
#endif

    int ret = RET_OK;

    PyFrameObject* frame = PyEval_GetFrame();
    PyObject* code_bytes_object = PyCode_GetCode(pCode);
    unsigned char * code_bytes = (unsigned char *)PyBytes_AS_STRING(code_bytes_object);

    self->handling = TRUE;
    BOOL ran_handler = FALSE;

    FrameAndCallbackVec* vec = &self->postop_callbacks;
    int cb_count = vec->count;
    if (cb_count > 0)
    {
        FrameAndCallback fcb = vec->items[cb_count - 1];
        // Check that the top callback is for this frame (it might be for a caller's frame instead)
        if (fcb.frame == (PyObject*)frame)
        {
            ran_handler = TRUE;
            PyObject* cb = fcb.callback;
            PyObject* result = NULL;
            result = PyObject_CallObject(cb, NULL);
            if (result == NULL)
            {
                self->handling = FALSE;
                Py_XDECREF(code_bytes_object);
                return RET_ERROR;
            }
            Py_DECREF(result);
            vec->count--;
            Py_DECREF(cb);
        }
    }

    int opcode = code_bytes[lasti];


    TableVec* tables = &self->handlers;
    int count = tables->count;
    HandlerTable* first_table = tables->items;
    int table_idx = 0;
    for(; table_idx < count; table_idx++) {
        // TODO: it feels like we should be able to go in reverse order, and
        // break after the first hit - investigate.
        PyObject* handler = first_table[table_idx].entries[opcode];
        if (handler == NULL) {
            continue;
        }
        ran_handler = TRUE;

        PyObject * arglist = Py_BuildValue("Osi", frame, "opcode", opcode);
        if (arglist == NULL) // (out of memory)
        {
            ret = RET_ERROR;
            break;
        }
        PyObject * result = PyObject_CallObject(handler, arglist);
        Py_DECREF(arglist);
        if (result == NULL)
        {
            ret = RET_ERROR;
            break;
        }
        Py_DECREF(result);
    }
    // repr_print(frame);
    // printf("lasti %d, line %d, cb_count %d, ran_handler %d, ret %d\n", lasti, PyFrame_GetLineNumber(frame), cb_count, ran_handler, ret);
    self->handling = FALSE;
    Py_XDECREF(code_bytes_object);

    return ret;
}


int EndsWith(const char *str, const char *suffix)
{
    if (!str || !suffix)
        return 0;
    size_t lenstr = strlen(str);
    size_t lensuffix = strlen(suffix);
    if (lensuffix >  lenstr)
        return 0;
    return strncmp(str + lenstr - lensuffix, suffix, lensuffix) == 0;
}

/*
 * The Trace Function
 */
static int
CTracer_trace(CTracer *self, PyFrameObject *frame, int what, PyObject *arg_unused)
{
    // printf("%x trace: f:%x %d @ %d\n", (int)self, (int)frame, what, PyFrame_GetLineNumber(frame));
    // struct timeval stop, start;
    // gettimeofday(&start, NULL);
    int ret = RET_OK;
    PyCodeObject* pCode = NULL;
    switch (what) {
    case PyTrace_CALL: {
        // const char * funcname = PyUnicode_AsUTF8(PyFrame_GetCode(frame)->co_name);
        // printf("func  { %s @ %s %d\n", funcname, filename, PyFrame_GetLineNumber(frame));

        pCode = PyFrame_GetCode(frame);
        // TODO: cache this result, perhaps?:
        const char * filename = PyUnicode_AsUTF8(pCode->co_filename);
        if (EndsWith(filename, "z3types.py") ||
            EndsWith(filename, "z3core.py") ||
            EndsWith(filename, "z3.py"))
        {
            dont_trace_frame(frame);
        } else {
            trace_frame(frame);
        }
        break;
    }
    case PyTrace_OPCODE: {
        pCode = PyFrame_GetCode(frame);
        int lasti = PyFrame_GetLasti(frame);
        if (CTracer_handle_opcode(self, pCode, lasti) < 0) { // == RET_ERROR) {
            ret = RET_ERROR;
        }
        break;
    }
    }

    // gettimeofday(&stop, NULL);
    // unsigned long delta = (stop.tv_sec - start.tv_sec) * 1000000 + stop.tv_usec - start.tv_usec;
    // if (delta > 5000) {
    //     printf("took %lu us\n", delta);
    //     const char * funcname = PyUnicode_AsUTF8(PyFrame_GetCode(frame)->co_name);
    //     const char * filename = PyUnicode_AsUTF8(PyFrame_GetCode(frame)->co_filename);
    //     printf(" func %s @ %s %d\n", funcname, filename, PyFrame_GetLineNumber(frame));
    // }

    Py_XDECREF(pCode);
    return ret;
}


/*
 * Python has two ways to set the trace function: sys.settrace(fn), which
 * takes a Python callable, and PyEval_SetTrace(func, obj), which takes
 * a C function and a Python object.  The way these work together is that
 * sys.settrace(pyfn) calls PyEval_SetTrace(builtin_func, pyfn), using the
 * Python callable as the object in PyEval_SetTrace.  So sys.gettrace()
 * simply returns the Python object used as the second argument to
 * PyEval_SetTrace.  So sys.gettrace() will return our self parameter, which
 * means it must be callable to be used in sys.settrace().
 *
 * So we make ourself callable, equivalent to invoking our trace function.
 */
static PyObject *
CTracer_call(CTracer *self, PyObject *args, PyObject *kwds)
{
    PyFrameObject *frame;
    PyObject *what_str;
    PyObject *arg;
    int lineno = 0;
    int what;
    PyObject *ret = NULL;
    PyObject * ascii = NULL;
    // printf("NOTE: CTracer called via python.\n");
    static char *what_names[] = {
        "call", "exception", "line", "return",
        "c_call", "c_exception", "c_return",
        "opcode",
        NULL
        };

    static char *kwlist[] = {"frame", "event", "arg", "lineno", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O!O!O|i:Tracer_call", kwlist,
            &PyFrame_Type, &frame, &PyUnicode_Type, &what_str, &arg, &lineno)) {
        goto done;
    }

    /* In Python, the what argument is a string, we need to find an int
       for the C function. */
    for (what = 0; what_names[what]; what++) {
        int should_break;
        ascii = PyUnicode_AsASCIIString(what_str);
        should_break = !strcmp(PyBytes_AS_STRING(ascii), what_names[what]);
        Py_DECREF(ascii);
        if (should_break) {
            break;
        }
    }

    /* Invoke the C function, and return ourselves. */
    if (CTracer_trace(self, frame, what, arg) == RET_OK) {
        Py_INCREF(self);
        ret = (PyObject *)self;
    } else {
        return NULL;
    }

    /* For better speed, install ourselves the C way so that future calls go
       directly to CTracer_trace, without this intermediate function.

       Only do this if this is a CALL event, since new trace functions only
       take effect then.  If we don't condition it on CALL, then we'll clobber
       the new trace function before it has a chance to get called.  To
       understand why, there are three internal values to track: frame.f_trace,
       c_tracefunc, and c_traceobj.  They are explained here:
       https://nedbatchelder.com/text/trace-function.html

       Without the conditional on PyTrace_CALL, this is what happens:

            def func():                 #   f_trace         c_tracefunc     c_traceobj
                                        #   --------------  --------------  --------------
                                        #   CTracer         CTracer.trace   CTracer
                sys.settrace(my_func)
                                        #   CTracer         trampoline      my_func
                        # Now Python calls trampoline(CTracer), which calls this function
                        # which calls PyEval_SetTrace below, setting us as the tracer again:
                                        #   CTracer         CTracer.trace   CTracer
                        # and it's as if the settrace never happened.
        */
    if (what == PyTrace_CALL) {
        PyEval_SetTrace((Py_tracefunc)CTracer_trace, (PyObject*)self);
    }

done:
    return ret;
}

static PyObject *
CTracer_push_postop_callback(CTracer *self, PyObject *args)
{
    PyObject *frame;
    PyObject *callback;
    if (!PyArg_ParseTuple(args, "OO", &frame, &callback)) {
        return NULL;
    }
#if PY_VERSION_HEX < 0x030C0000
    // Python 3.12+ uses None callbacks to signal that a callback MIGHT happen
    if Py_IsNone(callback) {
        Py_RETURN_NONE;
    }
#endif
    Py_INCREF(callback);
    FrameAndCallback fcb = {frame, callback};
    push_framecb(&self->postop_callbacks, fcb);
    Py_RETURN_NONE;
}

static PyObject *
CTracer_start(CTracer *self, PyObject *args_unused)
{
#if PY_VERSION_HEX >= 0x030C0000
    // use sys.monitoring in Python 3.12
    //     int tool_id = 4;
    //     int event_id = PY_MONITORING_EVENT_INSTRUCTION;
    //     func = _PyMonitoring_RegisterCallback(tool_id, event_id, (PyObject*)self);
    self->thread_id = PyThreadState_GetID(PyThreadState_Get());

#else
    // Enable opcode tracing in all callers:
    PyFrameObject * frame = PyEval_GetFrame();
#if PY_VERSION_HEX < 0x03000000
    while(frame != NULL && frame->f_trace_opcodes != 1) {
        trace_frame(frame);
        frame = _PyFrame_GetBackBorrow(frame);
    }
#else
    while(frame != NULL) {
        trace_frame(frame);
        frame = _PyFrame_GetBackBorrow(frame);
    }
#endif
    PyEval_SetTrace((Py_tracefunc)CTracer_trace, (PyObject*)self);
#endif
    self->enabled = TRUE;
    // printf(" -- -- trace start -- --\n");

    Py_RETURN_NONE;
}

static PyObject *
CTracer_stop(CTracer *self, PyObject *args_unused)
{
#if PY_VERSION_HEX < 0x030C0000
    PyEval_SetTrace(NULL, NULL);
#endif
    self->enabled = FALSE;

    // printf(" -- -- trace stop -- --\n");
    Py_RETURN_NONE;
}

static PyObject* _CH_SYS_MONITORING_DISABLE = NULL;

static PyObject *
CTracer_instruction_monitor(CTracer *self, PyObject *args)
{
    if (!self->enabled) {
        Py_RETURN_NONE;
    }

    int thread_id = PyThreadState_GetID(PyThreadState_Get());
    if (thread_id != self->thread_id) {
        Py_RETURN_NONE;
    }

    PyCodeObject* pCode;
    int lasti;
    if (!PyArg_ParseTuple(args, "Oi", &pCode, &lasti)) {
        return NULL;
    }

    const char * filename = PyUnicode_AsUTF8(pCode->co_filename);
    if (EndsWith(filename, "z3types.py") ||
        EndsWith(filename, "z3core.py") ||
        EndsWith(filename, "z3.py"))
    {
        Py_RETURN_NONE;
    }

    // const char * fnname = PyUnicode_AsUTF8(pCode->co_name);
    // printf("CTracer_instruction_monitor %s %d\n", fnname, lasti);

    int ret = CTracer_handle_opcode(self, pCode, lasti);
    switch(ret) {
        case RET_ERROR:
        return NULL;

        case RET_OK:
        Py_RETURN_NONE;

        case RET_DISABLE_TRACING:
        if (_CH_SYS_MONITORING_DISABLE == NULL) {
            PyObject* sys_module = PyImport_ImportModule("sys");
            PyObject* monitoring = PyObject_GetAttrString(sys_module, "monitoring");
            _CH_SYS_MONITORING_DISABLE = PyObject_GetAttrString(monitoring, "DISABLE");
            Py_DECREF(sys_module);
            Py_DECREF(monitoring);
        }
        Py_INCREF(_CH_SYS_MONITORING_DISABLE);
        return _CH_SYS_MONITORING_DISABLE;

        default:
        return NULL;
    }
}

static PyObject *
CTracer_is_handling(CTracer *self, PyObject *args_unused)
{
    return PyBool_FromLong(self->handling);
}

static PyObject *
CTracer_enabled(CTracer *self, PyObject *args_unused)
{
    return PyBool_FromLong(self->enabled & !self->handling);
}

static PyMemberDef
CTracer_members[] = {
    { NULL }
};

static PyMethodDef
CTracer_methods[] = {
    { "start", (PyCFunction) CTracer_start, METH_VARARGS,
            PyDoc_STR("Start the tracer") },

    { "stop", (PyCFunction) CTracer_stop, METH_VARARGS,
            PyDoc_STR("Stop the tracer") },

    {"instruction_monitor", (PyCFunction) CTracer_instruction_monitor, METH_VARARGS,
            PyDoc_STR("Callback for sys.monitoring instruction events") },

    { "enabled", (PyCFunction) CTracer_enabled, METH_VARARGS,
            PyDoc_STR("Check if the tracer is enabled") },

    { "is_handling", (PyCFunction) CTracer_is_handling, METH_VARARGS,
            PyDoc_STR("Check if the tracer is currently handling an opcode") },

    { "push_postop_callback", (PyCFunction) CTracer_push_postop_callback, METH_VARARGS,
            PyDoc_STR("Add a post-op callback") },

    { "pop_module", (PyCFunction) CTracer_pop_module, METH_VARARGS,
            PyDoc_STR("Remove a module to the tracer") },

    { "push_module", (PyCFunction) CTracer_push_module, METH_VARARGS,
            PyDoc_STR("Add a module to the tracer") },

    { "get_modules", (PyCFunction) CTracer_get_modules, METH_VARARGS,
            PyDoc_STR("Get a list of modules") },

    { NULL }
};

PyTypeObject
CTracerType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "coverage.CTracer",        /*tp_name*/
    sizeof(CTracer),           /*tp_basicsize*/
    0,                         /*tp_itemsize*/
    (destructor)CTracer_dealloc, /*tp_dealloc*/
    0,                         /*tp_print*/
    0,                         /*tp_getattr*/
    0,                         /*tp_setattr*/
    0,                         /*tp_compare*/
    0,                         /*tp_repr*/
    0,                         /*tp_as_number*/
    0,                         /*tp_as_sequence*/
    0,                         /*tp_as_mapping*/
    0,                         /*tp_hash */
    (ternaryfunc)CTracer_call, /*tp_call*/
    0,                         /*tp_str*/
    0,                         /*tp_getattro*/
    0,                         /*tp_setattro*/
    0,                         /*tp_as_buffer*/
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, /*tp_flags*/
    "CTracer objects",         /* tp_doc */
    0,                         /* tp_traverse */
    0,                         /* tp_clear */
    0,                         /* tp_richcompare */
    0,                         /* tp_weaklistoffset */
    0,                         /* tp_iter */
    0,                         /* tp_iternext */
    CTracer_methods,           /* tp_methods */
    CTracer_members,           /* tp_members */
    0,                         /* tp_getset */
    0,                         /* tp_base */
    0,                         /* tp_dict */
    0,                         /* tp_descr_get */
    0,                         /* tp_descr_set */
    0,                         /* tp_dictoffset */
    (initproc)CTracer_init,    /* tp_init */
    0,                         /* tp_alloc */
    0,                         /* tp_new */
};


static int
TraceSwap_init(TraceSwap *self, PyObject *args, PyObject *kwds_unused)
{
    if (!PyArg_ParseTuple(args, "Oi", &self->tracer, &self->disabling)) {
        return RET_ERROR;
    }
    Py_INCREF(self->tracer);
    return RET_OK;
}

static void
TraceSwap_dealloc(TraceSwap *self)
{
    Py_DECREF(self->tracer);
    Py_TYPE(self)->tp_free((PyObject*)self);
}

static PyObject *
TraceSwap__enter__(TraceSwap *self, PyObject *Py_UNUSED(ignored))
{
    // PyThreadState* thread_state = PyThreadState_Get();
    BOOL is_tracing = ((CTracer*)self->tracer)->enabled;
    BOOL noop = (self->disabling != is_tracing);
    self->noop = noop;
    if (! noop) {
        if (self->disabling)
        {
            // fprintf(stderr, "NoTracing enter\n");
            CTracer_stop((CTracer*)self->tracer, NULL);
        } else {
            // fprintf(stderr, "ResumedTracing enter\n");
            CTracer_start((CTracer*)self->tracer, NULL);
        }
    }
    Py_RETURN_NONE;
}

static PyObject *
TraceSwap__exit__(
    TraceSwap *self, PyObject **exc, int argct)
{
    if (!self->noop && exc[0] != PyExc_GeneratorExit)
    {
        if (self->disabling)
        {
            // fprintf(stderr, " NoTracing exit\n");
            CTracer_start((CTracer*)self->tracer, NULL);
        } else {
            CTracer_stop((CTracer*)self->tracer, NULL);
            // fprintf(stderr, " ResumedTracing exit\n");
        }
    }
    Py_RETURN_NONE;
}

static PyMemberDef
TraceSwap_members[] = {
    { NULL }
};


static PyMethodDef
TraceSwap_methods[] = {
    {"__enter__", (PyCFunction)TraceSwap__enter__,
        METH_NOARGS, "Enter tracing change"},
    {"__exit__", (PyCFunction)TraceSwap__exit__,
        METH_FASTCALL, "Exit tracing change"},
    { NULL }
};

PyTypeObject
TraceSwapType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "coverage.TraceSwap",      /*tp_name*/
    sizeof(TraceSwap),         /*tp_basicsize*/
    0,                         /*tp_itemsize*/
    (destructor)TraceSwap_dealloc, /*tp_dealloc*/
    0,                         /*tp_print*/
    0,                         /*tp_getattr*/
    0,                         /*tp_setattr*/
    0,                         /*tp_compare*/
    0,                         /*tp_repr*/
    0,                         /*tp_as_number*/
    0,                         /*tp_as_sequence*/
    0,                         /*tp_as_mapping*/
    0,                         /*tp_hash */
    0,                         /*tp_call*/
    0,                         /*tp_str*/
    0,                         /*tp_getattro*/
    0,                         /*tp_setattro*/
    0,                         /*tp_as_buffer*/
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, /*tp_flags*/
    "Change tracing state",    /* tp_doc */
    0,                         /* tp_traverse */
    0,                         /* tp_clear */
    0,                         /* tp_richcompare */
    0,                         /* tp_weaklistoffset */
    0,                         /* tp_iter */
    0,                         /* tp_iternext */
    TraceSwap_methods,         /* tp_methods */
    TraceSwap_members,         /* tp_members */
    0,                         /* tp_getset */
    0,                         /* tp_base */
    0,                         /* tp_dict */
    0,                         /* tp_descr_get */
    0,                         /* tp_descr_set */
    0,                         /* tp_dictoffset */
    (initproc)TraceSwap_init,  /* tp_init */
    0,                         /* tp_alloc */
    0,                         /* tp_new */
};


/* Module definition */


#define MODULE_DOC PyDoc_STR("CrossHair's intercepting tracer.")



static PyObject **crosshair_tracers_stack_lookup(PyFrameObject *frame, int index) {
#if PY_VERSION_HEX >= 0x030C0000
    // Python 3.12
    PyCodeObject* code = _PyFrame_GetCodeBorrow(frame);
    _PyInterpreterFrame* interpreterFrame = frame->f_frame;
    int64_t *stacks = _ch_get_stacks(code);
    int lasti = PyFrame_GetLasti(frame) / 2;
    int64_t stack_contents = stacks[lasti];
    if (stack_contents < 0) {
        return NULL;
    }
    int stacktop = stack_contents >> 1;
    return &(interpreterFrame->localsplus[code->co_nlocalsplus + stacktop + index]);
    // PyObject* ret = interpreterFrame->localsplus[stacktop + index];
#elif PY_VERSION_HEX >= 0x030B0000
    // Python 3.11
    _PyInterpreterFrame* interpreterFrame = frame->f_frame;
    int stacktop = interpreterFrame->stacktop;
    return &(interpreterFrame->localsplus[stacktop + index]);
#elif PY_VERSION_HEX >= 0x030A0000
    // Python 3.10
    return &(frame->f_valuestack[frame->f_stackdepth + index]);
#else
    // Python 3.8, 3.9
    return &(frame->f_stacktop[index]);
#endif
}

static PyObject *crosshair_tracers_stack_read(PyObject *self, PyObject *args)
{
    PyFrameObject *frame;
    int index;
    if (!PyArg_ParseTuple(args, "Oi", &frame, &index)) {
        return NULL;
    }
    PyObject **retaddr = crosshair_tracers_stack_lookup(frame, index);
    if (retaddr == NULL) {
        PyErr_SetString(PyExc_TypeError, "Stack computation overflow");
        return NULL;
    }
    PyObject *ret = *retaddr;
    if (ret == NULL) {
        PyErr_SetString(PyExc_ValueError, "No stack value is present");
        return NULL;
    } else {
        Py_INCREF(ret);
        return ret;
    }
}

static PyObject* crosshair_tracers_stack_write(PyObject *self, PyObject *args)
{
    PyFrameObject *frame;
    PyObject *val;
    int index;
    if (!PyArg_ParseTuple(args, "OiO", &frame, &index, &val)) {
        return NULL;
    }
    PyObject** stackval = crosshair_tracers_stack_lookup(frame, index);
    if (stackval == NULL) {
        PyErr_SetString(PyExc_TypeError, "Stack computation overflow");
        return NULL;
    }
    if (*stackval != NULL) {
        Py_DECREF(*stackval);
    }
    Py_INCREF(val);
    *stackval = val;
    Py_RETURN_NONE;
}

static PyObject* crosshair_tracers_code_stack_depths(PyObject *self, PyObject *args)
{
    PyCodeObject *code;
    if (!PyArg_ParseTuple(args, "O", &code)) {
        return NULL;
    }

#if PY_VERSION_HEX >= 0x030C0000
    // Python 3.12
    int64_t *stacks = _ch_get_stacks(code);
    int codelen = (int)Py_SIZE(code);
    PyObject* python_val = PyList_New(codelen);
    for (int i = 0; i < codelen; ++i)
    {
        int stackdepth = stacks[i];
        stackdepth = (stackdepth < 0) ? stackdepth : stackdepth >> 1;
        PyObject* python_int = Py_BuildValue("i", stackdepth);
        PyList_SetItem(python_val, i, python_int);
    }
    return python_val;
#else
    Py_RETURN_NONE;
#endif
}

static PyObject* crosshair_tracers_supported_opcodes(PyObject *self, PyObject *args)
{
    PyCodeObject *code;
    if (!PyArg_ParseTuple(args, "")) {
        return NULL;
    }
#if PY_VERSION_HEX >= 0x030C0000
    // Python 3.12
    PyObject* python_val = PyList_New(0);
    for (int i = 0; i < 256; i++) {
        if (_ch_TRACABLE_INSTRUCTIONS[i]) {
            PyObject* python_int = Py_BuildValue("i", i);
            PyList_Append(python_val, python_int);
        }
    }
    // We also designate the non-existant instruction (256) as "supported":
    PyObject* python_int = Py_BuildValue("i", 256);
    PyList_Append(python_val, python_int);
    return python_val;
#else
    Py_RETURN_NONE;
#endif
}

static PyMethodDef TracersMethods[] = {
    {"frame_stack_read",  crosshair_tracers_stack_read, METH_VARARGS, "Fetch a value from the interpreter stack."},
    {"frame_stack_write",  crosshair_tracers_stack_write, METH_VARARGS, "Overwrite a value on the interpreter stack."},
    {"code_stack_depths", crosshair_tracers_code_stack_depths, METH_VARARGS, "Find stack depths at various wordcode locations"},
    {"supported_opcodes", crosshair_tracers_supported_opcodes, METH_VARARGS, "Return opcodes that are supported by the C tracer"},
    {NULL, NULL, 0, NULL}        /* Sentinel */
};

static PyModuleDef
moduledef = {
    PyModuleDef_HEAD_INIT,
    "_crosshair_tracers",
    MODULE_DOC,
    -1,
    TracersMethods,
    NULL,
    NULL,       /* traverse */
    NULL,       /* clear */
    NULL
};


PyObject *
PyInit__crosshair_tracers(void)
{
    PyObject * mod = PyModule_Create(&moduledef);
    if (mod == NULL) {
        return NULL;
    }

    /* Initialize CTracer */
    CTracerType.tp_new = PyType_GenericNew;
    if (PyType_Ready(&CTracerType) < 0) {
        Py_DECREF(mod);
        return NULL;
    }

    Py_INCREF(&CTracerType);
    if (PyModule_AddObject(mod, "CTracer", (PyObject *)&CTracerType) < 0) {
        Py_DECREF(mod);
        Py_DECREF(&CTracerType);
        return NULL;
    }

    /* Initialize TraceSwap */
    TraceSwapType.tp_new = PyType_GenericNew;
    if (PyType_Ready(&TraceSwapType) < 0) {
        Py_DECREF(mod);
        Py_DECREF(&CTracerType);
        return NULL;
    }

    Py_INCREF(&TraceSwapType);
    if (PyModule_AddObject(mod, "TraceSwap", (PyObject *)&TraceSwapType) < 0) {
        Py_DECREF(mod);
        Py_DECREF(&CTracerType);
        Py_DECREF(&TraceSwapType);
        return NULL;
    }

    return mod;
}
