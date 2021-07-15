import dis
from crosshair.core import register_opcode_patch
from crosshair.tracers import TracingModule
from crosshair.util import debug

COMPARE_OP = dis.opmap["COMPARE_OP"]
CONTAINS_OP = dis.opmap.get("CONTAINS_OP", 118)


class OpcodeInterceptor(TracingModule):

    opcodes_wanted = frozenset(
        [
            # COMPARE_OP,
            CONTAINS_OP,
        ]
    )

    def trace_op(self, frame, codeobj, opcodenum):
        # TODO: implement string containment
        #print("I AM TRACING", opcodenum, codeobj.co_filename, codeobj.co_name)
        pass


def make_registrations():
    register_opcode_patch(OpcodeInterceptor())
