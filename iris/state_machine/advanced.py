from ..model import IRIS, IrisBase
from .core import StateMachine
from .basic import AssignableMachine, Assign, DoAll, Print, ValueState, Value
from . import types as t
import uuid

class AddToIrisEnv(StateMachine):
    def __init__(self, env_name, env_value, iris=IRIS):
        self.env_name = env_name
        self.env_value = env_value
        self.iris = iris
        super().__init__()
        self.accepts_input = False
    def get_output(self):
        return ["I saved the result as {}.".format(self.read_variable(self.env_name))]
    def next_state_base(self, text):
        self.iris.add_to_env(self.read_variable(self.env_name), self.read_variable(self.env_value))
        return Value(None, self.context)

class Function(AssignableMachine):
    title = "Function title"
    argument_types = {}
    def __init__(self):
        self.command_args = self.command.__code__.co_varnames[:self.command.__code__.co_argcount][1:]
        super().__init__()
        self.accepts_input = False
        self.reset()
        # read a variable from context
    def reset(self):
        self.output = ["Calling " + self.title.lower()]
        self.scope = str(uuid.uuid4()).upper()[0:10]
    def gen_scope(self, name):
        return self.scope + "_" + name
    def read_variable(self, varname):
        scope_var = self.gen_scope(varname)
        if scope_var in self.context["ASSIGNMENTS"]:
            return self.context["ASSIGNMENTS"][scope_var]
        return None
    def write_variable(self, varname, value):
        scope_var = self.gen_scope(varname)
        self.context["ASSIGNMENTS"][scope_var] = value
    def delete_variable(self, varname):
        scope_var = self.gen_scope(varname)
        if scope_var in self.context["ASSIGNMENTS"]:
            del self.context["ASSIGNMENTS"][scope_var]
    def next_state_base(self, text):
        self.output = []
        if all([self.read_variable(arg) for arg in self.command_args]):
            args = [self.read_variable(arg) for arg in self.command_args]
            result = self.command(*args)
            return ValueState(result).when_done(self.get_when_done_state())
        out = []
        for arg in self.command_args:
            if self.read_variable(arg):
                out.append(Print(["I am setting {} (inside \"{}\") as {}".format(arg, self.title.lower(), self.read_variable(arg))]))
            if not self.read_variable(arg):
                out.append(Assign(self.gen_scope(arg), self.argument_types[arg].set_arg_name(arg)))
                return DoAll(out).when_done(self)
    def command(self):
        pass

class BoundFunction(AssignableMachine):
    def __init__(self, bindings, function):
        self.bindings = bindings
        self.function = function
        super().__init__()
        self.accepts_input = False
    def bind_to_context(self, bindings, function):
        for name, type_ in function.argument_types.items():
            if not name in bindings: continue
            if isinstance(type_, Function):
                if not isinstance(bindings[name], dict):
                    raise Exception("Nested function bindings must be dictionary")
                self.bind_to_context(bindings[name], type_)
            else:
                self.write_variable(function.gen_scope(name), bindings[name])
    def next_state_base(self, text):
        self.bind_to_context(self.bindings, self.function)
        print("BEFORE EXECUTION", self.context)
        return self.function.when_done(self.get_when_done_state())

class ConfirmBinding(AssignableMachine):
    def __init__(self, binding):
        self.binding = binding
        super().__init__()
        self.accepts_input = False
    def next_state_base(self, text):
        if not self.read_variable("binding"):
            return Assign("binding", self.binding).when_done(self)
        else:
            binding = self.read_variable("binding")
            self.delete_variable("binding")
            if isinstance(binding, dict):
                return ValueState((binding, False)).when_done(self.get_when_done_state())
            else:
                confirm_binding = t.YesNo("Would you like to expose this argument as a variable?",
                          yes=(binding, True),
                          no=(binding, False))
                return confirm_binding.when_done(self.get_when_done_state())

class CreateBindings(AssignableMachine):
    def __init__(self, function, recursed=False):
        self.function = function
        super().__init__()
        self.accepts_input = False
        self.recursed_text = " recursed call to " if recursed else " "
    def next_state_base(self, text):
        command_args = self.function.command_args
        if all([self.read_variable(self.function.gen_scope(arg)) for arg in command_args]):
            bindings = {arg: self.read_variable(self.function.gen_scope(arg)) for arg in command_args}
            for arg in command_args:
                self.delete_variable(self.function.gen_scope(arg))
            return ValueState(bindings).when_done(self.get_when_done_state())
        for arg in command_args:
            if not self.read_variable(self.function.gen_scope(arg)):
                arg_type = self.function.argument_types[arg].set_arg_name(arg)
                if isinstance(arg_type, Function):
                    arg_type = CreateBindings(arg_type, recursed=True)
                return DoAll([
                    Print(["For{}{}".format(self.recursed_text, self.function.title.lower())]),
                    Assign(self.function.gen_scope(arg), ConfirmBinding(arg_type))
                ]).when_done(self)

class PrintFunction(StateMachine):
    def __init__(self, function):
        self.function = function
        super().__init__()
        self.accepts_input = False
    def next_state_base(self, text):
        if not self.read_variable("result"):
            return Assign("result", self.function).when_done(self)
        else:
            result = self.read_variable("result")
            self.delete_variable("result")
            return Print(["{}".format(result)]).when_done(self.get_when_done_state())

class CreatePartial(AssignableMachine):
    def __init__(self, function):
        self.function = function
        super().__init__()
        self.accepts_input = False
    def filter_bindings(self, bindings):
        out = {}
        for k,v in bindings.items():
            if not v[1]:
                out[k] = v[0] if not isinstance(v[0], dict) else self.filter_bindings(v[0])
        return out
    def all_bindings(self, bindings):
        out = {}
        for k,v in bindings.items():
            out[k] = v[0] if not isinstance(v[0], dict) else self.all_bindings(v[0])
        return out
    def next_state_base(self, text):
        if not self.read_variable("bindings"):
            return Assign("bindings", CreateBindings(self.function)).when_done(self)
        else:
            bindings = self.read_variable("bindings")
            filtered_bindings = self.filter_bindings(bindings)
            all_bindings = self.all_bindings(bindings)
            self.delete_variable("bindings")
            partial_func = BoundFunction(filtered_bindings, self.function)
            func_exec = BoundFunction(all_bindings, self.function)
            self.assign(partial_func)
            return PrintFunction(func_exec).when_done(self.get_when_done_state())
