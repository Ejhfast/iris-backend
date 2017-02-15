from .basic import StateMachine, Scope, AssignableMachine, Assign, DoAll, Print, ValueState, Value
from .advanced import AddToIrisEnv2
from . import types as t
from .model import IRIS_MODEL
from .middleware import Middleware, ExplainMiddleware
from .. import util
from .. import iris_objects
import copy
import sys

class NoneState:
    value = None

# the Function class returns a value, as well as a representation of the
# the program that produced that value
class FunctionReturn:
    def __init__(self, value, program):
        self.value = value
        self.program = program

def value_or_program(value):
    return ValueState(value) if not isinstance(value, FunctionReturn) else value.program

# here we are going to create a representation of each expression in the env
# as evaluation continues, we can use this later to learn from user commands
def compile_function(function, args):
    new_function = IrisCommand(index_command=False, compiled=True)
    new_function.command = function.command
    new_function.explanation = function.explanation
    new_function.command_args = function.command_args
    new_function.argument_types = {k:copy.copy(v) for k,v in function.argument_types.items()}
    new_function.title = "copy of " + function.title
    new_function.set_output()
    #new_function.output = ["Sure, I can call {}".format(function.title)]
    for key, value in args.items():
            new_function.binding_machine[key] = value_or_program(value)
    return new_function

# since variables are behaving more like references now, where the underlying
# value can change, and that should be reflected if e.g., a function is called again
# TODO: this is more general now, represents processing done on arguments before they
# are passed to function command code
def resolve_env_ref(iris, var):
    if isinstance(var, iris_objects.EnvReference):
        return iris.env[var.name]
    if isinstance(var, FunctionReturn):
        return var.value
    return var

# similar name helper for assignment names
def resolve_env_name(iris, var):
    if isinstance(var, iris_objects.EnvReference):
        return var.name
    return var

class CallMakeHoles(Scope, AssignableMachine):
    def __init__(self):
        super().__init__()
        self.accepts_input = False
        self.init_scope()
    def next_state_base(self, text):
        if not self.read_variable("for_holes") == None:
            get_func = self.read_variable("for_holes").function.function # Wrapper.Args
            return MakeHolesFunction(get_func).when_done(self.get_when_done_state())
        return Assign(self.gen_scope("for_holes"), FunctionSearch()).when_done(self)

# clean up
class MakeHolesFunction(Scope, AssignableMachine):
    def __init__(self, function):
        self.function = function
        super().__init__()
        self.accepts_input = False
        self.init_scope()
    def next_state_base(self, text):
        print(self.function, self.function.binding_machine)
        if all([self.read_variable(arg) != None for arg in self.function.binding_machine.keys()]):
            for arg in list(self.function.binding_machine.keys()):
                if self.read_variable(arg) == False:
                    del self.function.binding_machine[arg]
            return ValueState("HOLES").when_done(self.get_when_done_state())
        for arg in self.function.binding_machine.keys():
            if self.read_variable(arg) == None:
                to_print = Print(["I am inside {}".format(self.function.title)])
                if isinstance(self.function.binding_machine[arg], Function):
                    return Assign(self.gen_scope(arg), DoAll([
                        to_print,
                        MakeHolesFunction(self.function.binding_machine[arg])
                        ])).when_done(self)
                elif isinstance(self.function.binding_machine[arg], Block):
                    return Assign(self.gen_scope(arg), DoAll([
                        MakeHolesFunction(x) for x in self.function.binding_machine[arg].get_states()
                    ])).when_done(self)
                elif isinstance(self.function.binding_machine[arg], If):
                    if_stmt = self.function.binding_machine[arg]
                    return Assign(self.gen_scope(arg), DoAll([
                        MakeHolesFunction(if_stmt.condition),
                        MakeHolesFunction(if_stmt.true_exp)
                    ]))
                return Assign(self.gen_scope(arg), DoAll([
                    to_print,
                    Print(util.print_assignment(arg, None, self.function.binding_machine[arg].value)),
                    t.YesNo("Do you want to keep {}? If not, I will make it a variable".format(arg),
                        yes=True,
                        no=False)])).when_done(self)

class ArgMatch(AssignableMachine):
    def __init__(self, function, query, iris = IRIS_MODEL):
        self.iris = IRIS_MODEL
        super().__init__()
        self.function = function
        self.query = query
        self.accepts_input = False
    def next_state_base(self, text):
        matches, bindings = [], {}
        argument_types = self.function.argument_types
        for cmd in self.function.training_examples():
            succ, map_ = util.arg_match(self.query, cmd)
            if succ:
                convert_types = {k: argument_types[k].convert_type(v, doing_match=True) for k,v in map_.items()}
                if all([v[0] for k,v in convert_types.items()]):
                    for arg, vals in convert_types.items():
                        bindings[arg] = ValueState(vals[1], name=map_[arg])
        return BoundFunction(bindings, self.function).when_done(self.get_when_done_state())

class FunctionSearch(AssignableMachine):
    def __init__(self, question = ["What function would you like to retrieve?"], text = None, iris = IRIS_MODEL):
        self.iris = IRIS_MODEL
        super().__init__()
        self.question = question
        self.text = text
        self.output = question
        if text:
            self.output = []
            self.accepts_input = False
    def base_hint(self, text):
        predictions = self.iris.predict_commands(text, 3)
        return [x[0].title for x in predictions]
    def next_state_base(self, text):
        if self.text:
            text = self.text
        command, score = self.iris.predict_commands(text)[0]
        self.command = copy.copy(command)
        self.command.init_scope() # setup a new scope
        self.command.set_query(text)
        match_and_return = iris_objects.FunctionWrapper(ArgMatch(self.command, text), self.command.title)
        self.assign(match_and_return)
        return Value(match_and_return, self.context)

class ApplySearch(Scope, AssignableMachine):
    def __init__(self, question = ["What would you like to do?"], text = None):
        self.question = question
        self.function_generator = FunctionSearch(self.question, text=text)
        super().__init__()
        self.accepts_input = False
        self.init_scope()
    def reset(self):
        self.reset_context()
        return self
    def base_hint(self, text):
        return self.function_generator.hint(text)
    def next_state_base(self, text):
        if self.read_variable("function") == None:
            return Assign(self.gen_scope("function"), self.function_generator).when_done(self)
        command = self.read_variable("function").function # Wrapper
        self.command = command.function
        return command.when_done(self.get_when_done_state())

class IrisMiddleware(Middleware):
    def __init__(self, arg, iris = IRIS_MODEL):
        self.iris = IRIS_MODEL
        self.arg = arg
    def test(self, text):
        if text and len(text.split()) > 1 and text[0] != "\"":
            return True
        return False
    def hint(self, text):
        predictions = self.iris.predict_commands(text, 3)
        return [x[0].title for x in predictions]
    def transform(self, caller, state, text):
        state.clear_error()
        if isinstance(caller, WorkLoop) or isinstance(caller, FunctionSearch):
            return state
        return ApplySearch(text=text).when_done(caller.get_when_done_state()).set_arg_name(self.arg)

class Function(Scope, AssignableMachine):
    title = "Function title"
    argument_types = {}
    examples = []
    query = None
    def __init__(self, iris = IRIS_MODEL):
        self.command_args = self.command.__code__.co_varnames[:self.command.__code__.co_argcount][1:]
        self.argument_types = {**self.command.__annotations__, **self.argument_types}
        self.binding_machine = {}
        super().__init__()
        self.accepts_input = False
        self.init_scope()
        self.set_output()
        self.iris = IRIS_MODEL
    def set_query(self, text):
        self.query = text
    def set_output(self):
        self.output = ["Sure, I can run " + self.title.lower()]
    def training_examples(self):
        if hasattr(self, "class_index"):
            return self.iris.class2cmd[self.class_index]
        return [self.title] + self.examples
    def generate_name_bindings(self):
        out = {}
        for arg in self.command_args:
            scoped = self.gen_scope(arg)
            out[arg] = self.context["ASSIGNMENT_NAMES"][scoped]
        return out
    def next_state_base(self, text):
        self.output = []
        if all([self.read_variable(arg) != None for arg in self.command_args]):
            # so here we want to create a new function
            program_so_far = compile_function(self, {arg:self.read_variable(arg) for arg in self.command_args})
            args = [resolve_env_ref(self.iris, self.read_variable(arg)) for arg in self.command_args]
            try:
                result = self.command(*args)
            except:
                print(str(sys.exc_info()[1]))
                return StateException(str(sys.exc_info()[1])).when_done(self.get_when_done_state())
            # moved from IrisCommand because blocks need acces to that/etc.
            self.iris.add_to_env("__MEMORY__", result)
            self.iris.add_to_env("__MEMORY_FUNC__", program_so_far)
            return_val = FunctionReturn(result, program_so_far)
            self.assign(return_val, name="COMMAND VALUE")
            return return_val
        out = []
        for arg in self.command_args:
            if self.read_variable(arg) != None:
                assign_name = self.context["ASSIGNMENT_NAMES"][self.gen_scope(arg)]
                #out.append(Print(util.print_assignment(arg, assign_name, self.read_variable(arg))))
            else:
                # first choice is an expicit state machine we know to run and bind to
                # we would only sometimes have this
                if arg in self.binding_machine:
                    # we are using copy here, because if we re-run, do not want these
                    # states to be altered
                    type_machine = copy.copy(self.binding_machine[arg]).set_arg_name(arg)
                    print("BOUND", type_machine)
                # second choice is the logic for user argument extraction
                # we should always have this
                else:
                    type_machine = self.argument_types[arg].set_arg_name(arg).add_middleware(IrisMiddleware(arg))
                out.append(Assign(self.gen_scope(arg), type_machine))
                return DoAll(out).when_done(self)
    def command(self):
        pass

class IrisCommand(Function):
    argument_help = {}
    def __init__(self, index_command=True, compiled=False):
        super().__init__()
        self.compiled = compiled
        if index_command:
            self.class_index = self.iris.add_command(self)
        for arg in self.command_args:
            if not arg in self.argument_types:
                raise Exception("{} needs an argument type".format(arg))
        self.add_help()
    def next_state_base(self, text):
        next_state = super().next_state_base(text)
        if not isinstance(next_state, StateMachine):
            ret_val = next_state.value
            program = next_state.program
            if isinstance(ret_val, StateMachine):
                return ret_val
            print_out = self.wrap_explanation(ret_val)
            succ, learning = self.iris.learn(self, self.generate_name_bindings())
            if succ:
                print_out.append("I learned \"{}\"".format(learning))
            if not self.compiled:
                return DoAll([Print(print_out)]).when_done(self.get_when_done_state())
            return None
        return next_state
    def get_output(self):
        if self.compiled:
            return []
        return self.output
    def add_help(self):
        for arg, type in self.argument_types.items():
            if arg in self.argument_help:
                help_state = self.argument_help[arg]
            else:
                help_state = "No help available for this argument."
            self.argument_types[arg].add_middleware(ExplainMiddleware(help_state, arg))
    def explanation(self, result):
        return result
    def wrap_explanation(self, result):
        results = self.explanation(result)
        out = []
        for r in util.single_or_list(results):
            if isinstance(r, iris_objects.IrisImage):
                out.append({"type": "image", "value": r.value})
            elif isinstance(r, iris_objects.FunctionWrapper):
                out.append("<Bound function: {}>".format(r.name))
            elif util.is_data(r):
                out.append({"type": "data", "value": util.prettify_data(r)})
            else:
                out.append(str(r))
        return out

class BoundFunction(AssignableMachine):
    def __init__(self, bindings, function, iris = IRIS_MODEL):
        self.iris = iris
        self.bindings = bindings
        self.function = function
        super().__init__()
        self.accepts_input = False
    def set_query(self, query):
        self.function.query = query
    def next_state_base(self, text):
        self.function.binding_machine = {**self.bindings, **self.function.binding_machine} #binding_machine
        return self.function.when_done(self.get_when_done_state())

class PrintFunction(Scope, AssignableMachine):
    def __init__(self, function):
        self.function = function
        super().__init__()
        self.accepts_input = False
        self.init_scope()
    def next_state_base(self, text):
        if self.read_variable("result") == None:
            return Assign(self.gen_scope("result"), self.function).when_done(self)
        else:
            result = self.read_variable("result")
            self.assign(result)
            return Print(["{}".format(result)]).when_done(self.get_when_done_state())

class StateException(StateMachine):
    def __init__(self, exception_text):
        super().__init__()
        self.accepts_input = False
        self.exception_text = exception_text
    def next_state_base(self, text):
        return Print([
            "Sorry, something went wrong with the underlying command.",
            self.exception_text
        ]).when_done(self.get_when_done_state())

class Block(AssignableMachine):
    def __init__(self, states):
        super().__init__()
        self.states = []
        self.accepts_input = False
        for i in range(0, len(states)-1):
            self.states.append(Assign("junk", states[i]))
        self.states.append(states[-1])
    def next_state_base(self, text):
        return DoAll(self.states).when_done(self.get_when_done_state())
    def get_states(self):
        return [state.assign_state for state in self.states[:-1]] + [self.states[-1]]

class WorkLoop(AssignableMachine):
    def __init__(self):
        super().__init__()
        self.output = ["Entering workflow, what would you like to do?"]
    def hint(self, text):
        if text and "done" in text:
            return ["ends current workflow"]
        else:
            return ApplySearch().base_hint(text)
    def next_state_base(self, text):
        if not self.read_variable("last_command") == None:
            self.append_variable("programs", self.read_variable("last_command").program)
        if text and "done" in text:
            print("WORKLOOP", self.read_variable("programs"))
            wrapped_program = FunctionReturn(self.read_variable("last_command").value, Block(self.read_variable("programs")))
            return DoAll([
                Print(["Okay, done with workflow."]),
                ValueState(wrapped_program) #ValueState(self.read_variable("last_command"))
            ]).when_done(self.get_when_done_state())
        else:
            self.output = []#["Still in workflow. What would you like to do next?"]
            return Assign("last_command", ApplySearch(text=text)).when_done(self)


class While(AssignableMachine):
    def __init__(self, condition, true_exp):
        self.condition = condition
        self.true_exp = true_exp
        super().__init__()
        self.accepts_input = False
    def next_state_base(self, text):
        print("BACK IN WHILE")
        # this boilerplate needs to be extracted somewhere
        condition_copy = copy.copy(self.condition)
        condition_copy.set_output()
        condition_copy.init_scope()
        true_exp_copy = copy.copy(self.true_exp)
        true_exp_copy.set_output()
        true_exp_copy.init_scope()
        return If(condition_copy, true_exp_copy, continuation=self).when_done(self.get_when_done_state())

class WhileState(AssignableMachine):
    def __init__(self):
        super().__init__()
        self.accepts_input = False
    def next_state_base(self, text):
        if self.read_variable("condition") == None:
            return Assign("condition", ApplySearch(question = ["Condition?"])).when_done(self)
        if self.read_variable("true_exp") == None:
            return Assign("true_exp", ApplySearch(question = ["True exp?"])).when_done(self)
        program = While(self.read_variable("condition").program, self.read_variable("true_exp").program)
        if self.read_variable("condition").value == True:
            return DoAll([
                ValueState(FunctionReturn(self.read_variable("true_exp").value, program)),
                program
            ]).when_done(self.get_when_done_state())
        return ValueState(FunctionReturn(NoneState(), program)).when_done(self.get_when_done_state())

class If(Scope, AssignableMachine):
    def __init__(self, condition, true_exp, continuation=None):
        self.condition = condition
        self.true_exp = true_exp
        self.continuation = continuation
        super().__init__()
        self.init_scope()
        self.accepts_input = False
    def true_continutation(self):
        if self.continuation:
            return self.continuation
        else:
            return self.get_when_done_state()
    def next_state_base(self, text):
        if self.read_variable("condition") == None:
            return Assign(self.gen_scope("condition"), self.condition).when_done(self)
        elif self.read_variable("condition").value == True:
            return self.true_exp.when_done(self.true_continutation())
        return ValueState(NoneState()).when_done(self.get_when_done_state())

class IfState(AssignableMachine):
    def __init__(self):
        super().__init__()
        self.accepts_input = False
    def next_state_base(self, text):
        if self.read_variable("condition") == None:
            return Assign("condition", ApplySearch(question = ["Condition?"])).when_done(self)
        if self.read_variable("true_exp") == None:
            return Assign("true_exp", ApplySearch(question = ["True exp?"])).when_done(self)
        program = If(self.read_variable("condition").program, self.read_variable("true_exp").program)
        print("program", self.read_variable("condition").program, self.read_variable("true_exp").program)
        if self.read_variable("condition").value == True:
            return ValueState(FunctionReturn(self.read_variable("true_exp").value, program)).when_done(self.get_when_done_state())
        return ValueState(FunctionReturn(NoneState(), program)).when_done(self.get_when_done_state())
