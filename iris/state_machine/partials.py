from .advanced import *
from .expression import * #TODO: no
from .. import iris_objects

class ConfirmBinding(AssignableMachine):
    def __init__(self, binding):
        self.binding = binding
        super().__init__()
        self.accepts_input = False
    def next_state_base(self, text):
        if self.read_variable("binding") == None:
            return Assign("binding", self.binding).when_done(self)
        else:
            binding = self.read_variable("binding")
            self.delete_variable("binding")
            if isinstance(binding, dict):
                return ValueState((binding, False)).when_done(self.get_when_done_state())
            else:
                confirm_binding = t.YesNo("Would you like to expose this argument as a variable?",
                          yes=(binding, False),
                          no=(binding, True))
                return confirm_binding.when_done(self.get_when_done_state())

class CreateBindings(AssignableMachine):
    def __init__(self, function, recursed=False):
        self.function = function
        super().__init__()
        self.accepts_input = False
        self.recursed_text = " recursed call to " if recursed else " "
    def next_state_base(self, text):
        command_args = self.function.command_args
        if all([self.read_variable(self.function.gen_scope(arg)) != None for arg in command_args]):
            bindings = {arg: self.read_variable(self.function.gen_scope(arg)) for arg in command_args}
            for arg in command_args:
                self.delete_variable(self.function.gen_scope(arg))
            return ValueState(bindings).when_done(self.get_when_done_state())
        for arg in command_args:
            if not self.read_variable(self.function.gen_scope(arg)):
                arg_type = self.function.argument_types[arg].set_arg_name(arg).reset()
                if isinstance(arg_type, Function):
                    arg_type = CreateBindings(arg_type, recursed=True)
                return DoAll([
                    Print(["For{}{}".format(self.recursed_text, self.function.title.lower())]),
                    Assign(self.function.gen_scope(arg), ConfirmBinding(arg_type))
                ]).when_done(self)

class MakePartial(AssignableMachine):
    def __init__(self, function):
        self.function = function.function # because of ArgMatch
        super().__init__()
        self.accepts_input = False
    def filter_bindings(self, bindings, check=lambda x: x):
        out = {}
        for k,v in bindings.items():
            if check(v[1]):
                out[k] = v[0] if not isinstance(v[0], dict) else self.filter_bindings(v[0])
        return out
    def next_state_base(self, text):
        if self.read_variable("bindings") == None:
            return Assign("bindings", CreateBindings(self.function)).when_done(self)
        else:
            bindings = self.read_variable("bindings")
            filtered_bindings = self.filter_bindings(bindings)
            self.delete_variable("bindings")
            partial_func = iris_objects.FunctionWrapper(BoundFunction(filtered_bindings, self.function), self.function.title)
            self.assign(partial_func)
            return Value(partial_func, self.context)

class CreatePartial(AssignableMachine):
    def __init__(self):
        super().__init__()
        self.accepts_input = False
    def next_state_base(self, text):
        if self.read_variable("function") == None:
            return Assign("function", FunctionSearch()).when_done(self)
        else:
            function = self.read_variable("function").function # because of function wrapper
            return MakePartial(function).when_done(self.get_when_done_state())

# Probably don't need:

class UnwrapFunction(StateMachine):
    def __init__(self, function):
        self.function = function
        super().__init__()
        self.accepts_input = False
    def next_state_base(self, text):
        if self.read_variable("wrapped_func") == None:
            return Assign("wrapped_func", self.function).when_done(self)
        return Print(["{}".format(self.read_variable("wrapped_func").value)]).when_done(self.get_when_done_state())
    def set_query(self, text):
        self.function.set_query(text)
