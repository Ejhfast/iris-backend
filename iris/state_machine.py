# this contains all the primitives for constructing state machines

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
        return self
    # proceed to next state for machine
    def next_state(self, text):
        keep_going, new_state = self.current_state.next_state(text)
        if isinstance(new_state, StateMachine):
            self.current_state = new_state(self.current_state.context)
        else:
            self.current_state = new_state # appropriate?
        return keep_going

class StateMachine:
    def __init__(self):
        self.accepts_input = True
        self.error = None
        self.context = { "ASSIGNMENTS": {}, "ASSIGNMENT_NAMES": {} }
        self.when_done_state = None
        self.output = []
        self.middleware = []
        self.middleware_test = None
    # this allows us to pass context, and gives an execution hook after context is set
    def __call__(self, context):
        self.context = context
        return self
    def add_middleware(self, middleware):
        if isinstance(middleware, list):
            for m in middleware:
                self.middleware.append(m)
        else:
            self.middleware.append(middleware)
        return self
    def is_assignable(self):
        return False
    # once one machine is "done" (no explicit next state), do we want to do
    # something else? useful for lists of things to do, or loops
    def when_done(self, new_state):
        self.when_done_state = new_state
        return self
    def get_accepts_input(self):
        return self.accepts_input
    # getter, because we need override later
    def get_when_done_state(self):
        return self.when_done_state
    # getter context:
    def get_context(self):
        return self.context
    # thin wrapper on question, the source of output to user
    def get_output(self):
        return self.output
    # set error message state
    def set_error(self, data=None):
        print("setting error", data)
        self.error = data
        return self
    # clear error
    def clear_error(self):
        self.error = None
        return self
    # how to compose error message
    def error_message(self):
        return self.output
    # wrap the next_state_base function, allow for linked-list style machines
    def next_state(self, text):
        out_tuple = self.next_state_base(text)
        keep_going, state = out_tuple
        if (not keep_going) and self.get_when_done_state():
                out_tuple = True, self.get_when_done_state()
        # peek at the future, then (if middleware), decide if we want to go there
        new_out_tuple = out_tuple
        for middleware in self.middleware:
            if middleware.test(text):
                new_out_tuple = middleware.transform(self, new_out_tuple)
        print(self, new_out_tuple)
        print(self.context)
        return new_out_tuple
    # placeholder for move to next state
    def next_state_base(self, text):
        return True, self

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
        return "quit" in text
    def transform(self, caller, state_tuple):
        keep_going, state = state_tuple
        state.clear_error()
        return True, Jump("START")

class ExplainMiddleware(Middleware):
    def __init__(self, gen_state):
        self.gen_state = gen_state
    def test(self, text):
        return any([x in text for x in ["explain", "help"]])
    def transform(self, caller, state_tuple):
        keep_going, state = state_tuple
        state.clear_error()
        return True, self.gen_state(caller)
        return keep_going, state

class AssignableMachine(StateMachine):
    arg_name = None
    def assign(self, value, name=None):
        if not name:
            name = self.string_representation(value)
        if "assign" in self.context:
            self.context["ASSIGNMENTS"][self.context["assign"]] = value
            self.context["ASSIGNMENT_NAMES"][self.context["assign"]] = name
            del self.context['assign']
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
        return True, self.states[0]
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
        return True, self.next_state_obj

class Jump(StateMachine):
    def __init__(self, state_label):
        self.state_label = state_label
        super().__init__()
    def next_state_base(self, text):
        return True, self.context[self.state_label]
    def when_done(self, new_state):
        self.context[self.state_label].when_done(new_state)
        return self

class Print(StateMachine):
    def __init__(self, output):
        super().__init__()
        self.output = output
        self.accepts_input = False
    def next_state_base(self, text):
        return False, Value(None, self.context)

class PrintF(StateMachine):
    def __init__(self, output_f):
        self.output_f = output_f
        super().__init__()
        self.accepts_input = False
    def get_output(self):
        return [self.output_f()]
    def next_state_base(self, text):
        return False, Value(None, self.context)

class Assign(StateMachine):
    def __init__(self, variable, assign_state):
        self.variable = variable
        # if not assign_state.is_assignable():
        #     raise Exception("{} is not assignable!".format(assign_state))
        self.assign_state = assign_state
        super().__init__()
        self.accepts_input = False
    def next_state_base(self, text):
        self.context["assign"] = self.variable.name
        return True, self.assign_state
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
        self.context["ASSIGNMENTS"][self.variable.name]=self.equal
        return True, self.next_state_obj

class Variable(StateMachine):
    def __init__(self, name):
        self.name = name
        super().__init__()
        self.accepts_input = False
    def next_state_base(self, text):
        return False, self.context["ASSIGNMENTS"][self.name]

class SequentialMachine:
    def __init__(self):
        self.states = []
    def add(self, state):
        self.states.append(state)
    def compile(self):
        return DoAll(self.states)

class RestartMiddleware(StateMachine):
    def __init__(self, test, next_state_obj, label):
        self.test = test
        self.next_state_obj = next_state_obj
        super().__init__()
        self.output = next_state_obj(self.context).get_output()

    def next_state_base(self, text):
        print("if", self.test(text), text, "quit" in text)
        if self.test(text):
            return self.next_state_obj.next_state_base(text)
        return self.label(self.context).next_state_base(text)

class Continue(StateMachine):
    def next_state_base(self, text):
        return False, None

Continue = None

class ValueState(AssignableMachine):
    def __init__(self, value):
        self.value = value
        super().__init__()
        self.accepts_input = False
    def next_state_base(self, text):
        self.assign(self.value)
        return False, self.value

class Value:
    def __init__(self, result, context):
        self.result = result
        self.context = context
