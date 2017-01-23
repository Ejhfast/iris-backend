# this contains all the primitives for constructing state machines
import copy
from . import iris_objects
def print_state(state):
    return "{}".format(state)

class StateMachineRunner:
    def __init__(self, state_machine):
        self.original_state = state_machine
        self.current_state = state_machine
    # get current state machine output
    def current_output(self):
        print(self.current_state)
        if self.current_state.error:
            message_out = self.current_state.error
            self.current_state.error = None
            return message_out
        return self.current_state.get_output()
    # keep running the state machine until it needs user input
    def run_until_input_required(self):
        keep_going = True
        outputs = []
        while keep_going and (not self.current_state.get_accepts_input()):
            keep_going = self.next_state(None)
            if keep_going:
                outputs = outputs + self.current_output()
        return keep_going, outputs
    # reset machine
    def reset(self):
        self.current_state = self.original_state
        self.original_state.reset()
        return self
    # proceed to next state for machine
    def next_state(self, text):
        new_state = self.current_state.next_state(text)
        if isinstance(new_state, StateMachine):
            self.current_state = new_state(self.current_state.context)
            return True
        else:
            self.current_state = new_state # appropriate?
            return False

class StateMachine:
    def __init__(self):
        self.accepts_input = True
        self.error = None
        self.context = { "ASSIGNMENTS": {}, "ASSIGNMENT_NAMES": {}, "assign": [] }
        self.when_done_state = None
        self.output = []
        self.middleware = []
        self.previous_state = None
    # this allows us to pass context, and gives an execution hook after context is set
    def __call__(self, context):
        self.context = dict(context)
        return self
    # add middleware to the state, will process input text and potentially make different choices
    def add_middleware(self, middleware):
        if isinstance(middleware, list):
            for m in middleware:
                self.middleware.append(m)
        else:
            self.middleware.append(middleware)
        return self
    # does this machine support assignment
    def is_assignable(self):
        return False
    # any kind of reset
    def reset(self):
        return self
    # once one machine is "done" (no explicit next state), do we want to do
    # something else? useful for lists of things to do, or loops
    def when_done(self, new_state):
        self.when_done_state = new_state
        return self
    # getter for accepts_input (does this state need input from user)
    def get_accepts_input(self):
        return self.accepts_input
    # getter for when_done state
    def get_when_done_state(self):
        return self.when_done_state
    # read a variable from context
    def read_variable(self, varname):
        print(varname, self.context)
        if varname in self.context["ASSIGNMENTS"]:
            return self.context["ASSIGNMENTS"][varname]
        return None
    # write a variable to context
    def write_variable(self, varname, value):
        self.context["ASSIGNMENTS"][varname] = value
    # getter for context:
    def get_context(self):
        return self.context
    # getter on output, the source of output to user
    def get_output(self):
        return self.output
    # set error message state
    def set_error(self, data=None):
        self.error = data
        return self
    # clear error
    def clear_error(self):
        self.error = None
        return self
    # tells machine how to generate error message
    def error_message(self):
        return self.output
    # wrap the next_state_base function, allow for linked-list style machines
    def next_state(self, text):
        next_state = self.next_state_base(text)
        if (not isinstance(next_state, StateMachine)) and self.get_when_done_state():
                next_state = self.get_when_done_state()
        # peek at the future, then (if middleware), decide if we want to go there
        for middleware in self.middleware:
            if middleware.test(text):
                next_state = middleware.transform(self, next_state)
        return next_state
    # placeholder for move to next state
    def next_state_base(self, text):
        return self

# Middleware wraps the result of another machines next_state function
# can be used to abstract common design patterns, e.g., "if user input matches quit, exit
# instead of executing further"
class Middleware:
    def test(self, text):
        return True
    def transform(self):
        pass

class QuitMiddleware(Middleware):
    def test(self, text):
        if text:
            return "quit" in text
        return False
    def transform(self, caller, state):
        state.clear_error()
        return Jump("START")

class ExplainMiddleware(Middleware):
    def __init__(self, gen_state):
        self.gen_state = gen_state
    def test(self, text):
        if text:
            return any([x in text for x in ["explain", "help"]])
        return False
    def transform(self, caller, state):
        state.clear_error()
        return self.gen_state(caller)

class AssignableMachine(StateMachine):
    arg_name = None
    def assign(self, value, name=None):
        if len(self.context["assign"]) > 0:
            curr_assign = self.context["assign"].pop()
            print("ASSIGN", curr_assign, value, name)
            self.context["ASSIGNMENTS"][curr_assign] = value
            self.context["ASSIGNMENT_NAMES"][curr_assign] = name
    def is_assignable(self):
        return True
    def set_arg_name(self, name):
        self.arg_name = name
        return self
    def string_representation(self, value):
        return str(value)

class DoAll(AssignableMachine):
    def __init__(self, states, next_state_obj=None):
        self.states = states
        for i, _ in enumerate(self.states):
            if i+1 < len(self.states):
                self.states[i].when_done(self.states[i+1])
        if next_state_obj:
            self.states[-1].when_done(next_state_obj)
        super().__init__()
        self.accepts_input = False
    def next_state_base(self, text):
        return self.states[0]
    def when_done(self, state):
        self.states[-1].when_done(state)
        return self
    def set_arg_name(self, name):
        for state in self.states:
            if isinstance(state, AssignableMachine):
                state.set_arg_name(name)
        return self

class Label(StateMachine):
    def __init__(self, label, next_state_obj):
        self.label = label
        self.next_state_obj = next_state_obj
        super().__init__()
        self.accepts_input = False
    def next_state_base(self, text):
        self.context[self.label] = self.next_state_obj
        return self.next_state_obj

class Jump(StateMachine):
    def __init__(self, state_label):
        self.state_label = state_label
        super().__init__()
        self.accepts_input = False
    def next_state_base(self, text):
        return self.context[self.state_label]
    def when_done(self, new_state):
        self.context[self.state_label].when_done(new_state)
        return self

class Print(StateMachine):
    def __init__(self, output):
        super().__init__()
        self.output = output
        self.accepts_input = False
    def next_state_base(self, text):
        return Value(None, self.context)

class PrintF(StateMachine):
    def __init__(self, output_f):
        self.output_f = output_f
        super().__init__()
        self.accepts_input = False
    def get_output(self):
        return [self.output_f()]
    def next_state_base(self, text):
        return Value(None, self.context)

class Assign(StateMachine):
    def __init__(self, variable, assign_state):
        self.variable = variable
        # if not assign_state.is_assignable():
        #     raise Exception("{} is not assignable!".format(assign_state))
        self.assign_state = assign_state
        super().__init__()
        self.accepts_input = False
    def next_state_base(self, text):
        if isinstance(self.variable, Variable):
            self.context["assign"].append(self.variable.scope_name())
        else:
            self.context["assign"].append(self.variable)
        return self.assign_state
    def when_done(self, state):
        self.assign_state.when_done(state)
        return self

class Let(StateMachine):
    def __init__(self, variable, equal=None, then_do=None):
        super().__init__()
        self.next_state_obj = then_do
        self.variable = variable
        self.equal = equal
        self.accepts_input = False
    def next_state_base(self, text):
        self.context["ASSIGNMENTS"][self.variable.scope_name()]=self.equal
        return self.next_state_obj

class Variable(StateMachine):
    def __init__(self, name, scope=None):
        self.name = name
        self.scope = scope
        super().__init__()
        self.accepts_input = False
    def scope_name(self):
        if self.scope:
            return self.scope + "_" + self.name
        return self.name
    def next_state_base(self, text):
        print("VALUE", self.name, self.context)
        return ValueState(self.get_value()).when_done(self.get_when_done_state())
    def get_value(self):
        return self.context["ASSIGNMENTS"][self.scope_name()]
    def get_named_value(self):
        return self.context["ASSIGNMENT_NAMES"][self.scope_name()]

class SequentialMachine:
    def __init__(self):
        self.states = []
    def add(self, state):
        self.states.append(state)
    def compile(self):
        return DoAll(self.states)

class ValueState(AssignableMachine):
    def __init__(self, value):
        self.value = value
        super().__init__()
        self.accepts_input = False
    def next_state_base(self, text):
        if isinstance(self.value, iris_objects.IrisValue):
            self.assign(self.value, name=self.value.name)
        else:
            self.assign(self.value)
        return self.value

class Value:
    def __init__(self, result, context):
        self.result = result
        self.context = context

class PrintVar(StateMachine):
    def __init__(self, var, func):
        self.var = var
        self.func = func
        super().__init__()
        self.accepts_input = False
    def next_state_base(self, text):
        self.var(self.context)
        name, named_value, value = self.var.name, self.var.get_named_value(), self.var.get_value()
        return Print(self.func(name, named_value, value)).when_done(self.get_when_done_state())

# def state_wrapper(f):
#     def wrapper(*args):
#         class DummyState(StateMachine):
#             def next_state_base(self, text):
#                 new_args = [arg(self.context).get_value() for arg in args]
#                 result = f(*new_args)
#                 if isinstance(result, StateMachine):
#                     return True, result.when_done(self.get_when_done_state())
#                 return True, ValueState(result).when_done(self.get_when_done_state())
#         return DummyState()
#     return wrapper
