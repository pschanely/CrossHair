
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


#include <Python.h>
#include <opcode.h>

#include "_tracers_pycompat.h"
#include "_tracers.h"


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
            Py_DECREF(wanted_itr);
            return NULL;
        }
        Py_DECREF(wanted_item);
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
    Py_RETURN_NONE;
}

/*
 * Parts of the trace function.
 */

static int
CTracer_handle_opcode(CTracer *self, PyFrameObject *frame)
{
    int ret = RET_OK;
    PyObject * pCode = NULL;

    int lasti = PyFrame_GetLasti(frame);
    pCode = PyCode_GetCode(PyFrame_GetCode(frame));
    unsigned char * code_bytes = (unsigned char *)PyBytes_AS_STRING(pCode);

    // const char * funcname = PyUnicode_AsUTF8(PyFrame_GetCode(frame)->co_name);
    // printf("opcode %s @ %d op: %d\n", funcname, lasti, code_bytes[lasti]);
    self->handling = TRUE;

    FrameAndCallbackVec* vec = &self->postop_callbacks;
    int cb_count = vec->count;
    if (cb_count > 0)
    {
        FrameAndCallback fcb = vec->items[cb_count - 1];
        if (fcb.frame == (PyObject*)frame)
        {
            PyObject* cb = fcb.callback;
            PyObject* result = PyObject_CallObject(cb, NULL);
            if (result == NULL)
            {
                self->handling = FALSE;
                Py_XDECREF(pCode);
                return RET_ERROR;
            }
            Py_DECREF(result);
            vec->count--;
            Py_DECREF(cb);
        } else {
        }
    }

    int opcode = code_bytes[lasti];


    TableVec* tables = &self->handlers;
    int count = tables->count;
    HandlerTable* first_table = tables->items;
    PyObject *extra = NULL;
    extra = Py_None;
    Py_INCREF(extra);
    for(int table_idx = 0; table_idx < count; table_idx++) {
        PyObject* handler = first_table[table_idx].entries[opcode];
        if (handler == NULL) {
            continue;
        }

        PyObject * arglist = Py_BuildValue("OsiO", frame, "opcode", opcode, extra);
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
        if (result == Py_None)
        {
            Py_DECREF(result);
        } else {
            Py_DECREF(extra);
            extra = result;
        }
    }
    Py_DECREF(extra);
    self->handling = FALSE;
    Py_XDECREF(pCode);

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
    // printf("%x trace: f:%x @ %d\n", (int)self, (int)frame, PyFrame_GetLineNumber(frame));
    // struct timeval stop, start;
    // gettimeofday(&start, NULL);

    switch (what) {
    case PyTrace_CALL: {
        // const char * funcname = PyUnicode_AsUTF8(PyFrame_GetCode(frame)->co_name);
        // printf("func  { %s @ %s %d\n", funcname, filename, PyFrame_GetLineNumber(frame));

        // TODO: cache this result, perhaps?:
        const char * filename = PyUnicode_AsUTF8(PyFrame_GetCode(frame)->co_filename);
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
        if (CTracer_handle_opcode(self, frame) < 0) {
            return RET_ERROR;
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

    return RET_OK;
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
    Py_XINCREF(callback);
    FrameAndCallback fcb = {frame, callback};
    push_framecb(&self->postop_callbacks, fcb);
    Py_RETURN_NONE;
}

static PyObject *
CTracer_start(CTracer *self, PyObject *args_unused)
{
    PyEval_SetTrace((Py_tracefunc)CTracer_trace, (PyObject*)self);
    // Enable opcode tracing in all callers:
    PyFrameObject * frame = PyEval_GetFrame();
#if PY_VERSION_HEX < 0x03000000
    while(frame != NULL && frame->f_trace_opcodes != 1) {
        trace_frame(frame);
        frame = PyFrame_GetBack(frame);
    }
#else
    while(frame != NULL) {
        trace_frame(frame);
        frame = PyFrame_GetBack(frame);
    }
#endif
    self->enabled = TRUE;
    // printf(" -- -- trace start -- --\n");

    Py_RETURN_NONE;
}

static PyObject *
CTracer_stop(CTracer *self, PyObject *args_unused)
{
    PyEval_SetTrace(NULL, NULL);
    self->enabled = FALSE;

    // printf(" -- -- trace stop -- --\n");
    Py_RETURN_NONE;
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
    PyThreadState* thread_state = PyThreadState_Get();
    BOOL is_tracing = (
        thread_state->c_tracefunc == (Py_tracefunc)CTracer_trace &&
        thread_state->c_traceobj == self->tracer
    );
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


static PyMethodDef TracersMethods[] = {
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
