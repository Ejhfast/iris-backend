from .basic import StateMachine, Scope, AssignableMachine, Assign, DoAll, Print, ValueState, Value
from . import types as t
from .model import IRIS_MODEL
from .middleware import Middleware, ExplainMiddleware
from .. import util
from .. import iris_objects
import copy
import sys

# Question for tomorrow: how should I poke holes in functions?

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
    new_function = Function()
    new_function.command = function.command
    new_function.command_args = function.command_args
    new_function.title = "copy of " + function.title
    for key, value in args.items():
        new_function.argument_types[key] = value_or_program(value)
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
                        # TODO: clean up, this should probably function like "bindings"
                        self.context["ASSIGNMENT_NAMES"][self.function.gen_scope(arg)] =  map_[arg]
                        bindings[arg] = vals[1]
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
        if text and len(text.split()) > 1:
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
        # Fix this, sort of broken
        self.output = ["Sure, I can " + self.title.lower()]
        self.iris = IRIS_MODEL
    def set_query(self, text):
        self.query = text
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
                return StateException(str(sys.exc_info()[1])).when_done(self.get_when_done_state())
            return_val = FunctionReturn(result, program_so_far)
            self.assign(return_val, name="COMMAND VALUE")
            return return_val
        out = []
        for arg in self.command_args:
            if self.read_variable(arg) != None:
                assign_name = self.context["ASSIGNMENT_NAMES"][self.gen_scope(arg)]
                out.append(Print(util.print_assignment(arg, assign_name, self.read_variable(arg))))
            else:
                if arg in self.binding_machine:
                    type_machine = self.binding_machine[arg].set_arg_name(arg)
                else:
                    type_machine = self.argument_types[arg].set_arg_name(arg).add_middleware(IrisMiddleware(arg))
                out.append(Assign(self.gen_scope(arg), type_machine))
                return DoAll(out).when_done(self)
    def command(self):
        pass

class BoundFunction(AssignableMachine):
    def __init__(self, bindings, function, iris = IRIS_MODEL):
        self.iris = iris
        self.bindings = bindings
        self.function = function
        super().__init__()
        self.accepts_input = False
    def set_query(self, query):
        self.function.query = query
    def bind_to_context(self, bindings, function):
        for name, type_ in function.argument_types.items():
            if not name in bindings: continue
            if isinstance(type_, Function):
                if not isinstance(bindings[name], dict):
                    raise Exception("Nested function bindings must be dictionary")
                self.bind_to_context(bindings[name], type_)
            else:
                self.context["ASSIGNMENT_NAMES"][self.function.gen_scope(name)] = resolve_env_name(self.iris, bindings[name])
                self.write_variable(function.gen_scope(name), bindings[name])
    def next_state_base(self, text):
        #self.bind_to_context(self.bindings, self.function)
        binding_machines = {k:value_or_program(value) for k, value in self.bindings.items()}
        self.function.binding_machines = binding_machines
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
        if text and "done" in text:
            return DoAll([
                Print(["Okay, done with workflow."]),
                ValueState(self.read_variable("last_command"))
            ]).when_done(self.get_when_done_state())
        else:
            self.output = []#["Still in workflow. What would you like to do next?"]
            return Assign("last_command", ApplySearch(text=text)).when_done(self)
