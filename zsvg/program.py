from zsvg.util import parse_step

from zsvg.registry import interpreter_registry
import zsvg.step_interpreters
import zsvg.view_interpreters
import zsvg.loc_interpreters_pred


class Program:

    def __init__(self, prog_str, init_state=None):
        self.prog_str = prog_str.rstrip()
        self.state = init_state if init_state is not None else dict()
        self.instructions = self.prog_str.split('\n')


class ProgramInterpreter:

    def __init__(self, loc='LOC_3D'):
        self.interpreters = interpreter_registry
        self.interpreters['LOC'] = self.interpreters[loc]

    def execute_step(self, prog_step, inspect):
        step_name = parse_step(prog_step.prog_str, partial=True)['step_name']
        return self.interpreters[step_name].execute(prog_step, inspect)

    def execute(self, prog, init_state, inspect=False):
        if isinstance(prog, str):
            prog = Program(prog, init_state)
        else:
            assert (isinstance(prog, Program))

        prog_steps = [Program(instruction, init_state=prog.state) for instruction in prog.instructions]

        html_str = '<hr>'
        for prog_step in prog_steps:
            if inspect:
                step_output, step_html = self.execute_step(prog_step, inspect)
                html_str += step_html + '<hr>'
            else:
                step_output = self.execute_step(prog_step, inspect)

        if inspect:
            return step_output, prog.state, html_str

        return step_output, prog.state
