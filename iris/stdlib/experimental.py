from .. import IrisCommand
from .. import state_types as t
from .. import state_machine as sm
from .. import util as util


# If I want to collect and stich together functions, need
# 1: retrieved functions should be storable state machines
# 2: when user paramaterizes them, should ask what are paramaters of workflow and what are constants
# 3: should be able to generate partial function based on user response, and store that
# 4: then stitch together partial functions and ask for metadata

# Notes, I can get this (below) to work, just fancier version of jump

class TestWorkLoop(IrisCommand):
    title = "test workloop"
    argument_types = {
        "test": sm.WorkLoop()
    }
    def command(self, test):
        return test

testWorkLoop = TestWorkLoop()

class TestGetFunction(IrisCommand):
    title = "get a function"
    argument_types = {
        "test": sm.FunctionSearch()
    }
    def command(self, test):
        return test

testGetFunction = TestGetFunction()

class MakePartial(IrisCommand):
    title = "make a partial function"
    argument_types = {
        "test": sm.CreatePartial(),
    }
    def command(self, test):
        return test

makePartial = MakePartial()

class CallFunction(IrisCommand):
    title = "call a function"
    argument_types = {
        "func": t.Function("What saved function do you want to call?")
    }
    def command(self, func):
        to_call = func.function
        to_call.set_query(None)
        # to_call.init_scope() # new scope
        return to_call.when_done(self.get_when_done_state())

callFunction = CallFunction()

class MakeCommand(IrisCommand):
    title = "save last block as iris command"
    argument_types = {
        "name": t.String("What would you like to call this new command?"),
    }
    def command(self, name):
        new_func = self.iris.env["__MEMORY_FUNC__"]
        new_func.title = name
        new_func.__class__ = IrisCommand
        new_func.class_index = self.iris.add_command(new_func)
        self.iris.train_model()
        return name
    def explanation(self, result):
        return "I created a new command called \"{}\"".format(result)

makeCommand = MakeCommand()

# class Add(sm.Function):
#     title = "add {x} and {y}"
#     argument_types = {
#         "x": t.Int("x"),
#         "y": t.Int("y")
#     }
#     def command(self, x, y):
#         return x + y
#
# add = Add()
#
# class Subtract(sm.Function):
#     title = "subtract {x} and {y}"
#     argument_types = {
#         "x": t.Int(),
#         "y": t.Int()
#     }
#     def command(self, x, y):
#         return x - y
#
# subtract = Subtract()

# subtract_with_closure = sm.BoundFunction({
#     "x": {"y": 3},
#     "y": 50
# }, Subtract())
#
# class Int(sm.Function):
#     title = "get int"
#     def __init__(self):
#         super().__init__()
#         self.accepts_input = False
#     def next_state_base(self, text):
#         return t.Int("Please enter an integer value:").when_done(self.get_when_done_state())
#
# class GrabFunc(sm.Function):
#     title = "grab func"
#     def __init__(self):
#         super().__init__()
#         self.accepts_input = False
#     argument_types = {
#         "func": sm.FunctionSearch()
#     }
#     def command(self, func):
#         return func
#
# grabFunc = GrabFunc()

# class DoPartial(sm.Function):
#     def __init__(self):
#         super().__init__()
#         self.accepts_input = False
#     argument_types = { "function": sm.FunctionSearch() }
#     def


# sm.IRIS_MODEL.train_model()

# class TestWhile(IrisCommand):
#     title = "test while loop"
#     examples = []
#     argument_types = {
#         "test": sm.DoAll([
#             sm.Assign("x", ValueState(10))
#             sm.While(sm.GreaterThan("x", 0),
#                 sm.DoAll([
#                     sm.Print(["Hi!"])
#                     sm.Assign("x", sm.Minus("x",1))
#                 ])
#             ),
#         ]
#     }
#     def command(self, test):
#         return test
#
# testWhile = TestWhile()
