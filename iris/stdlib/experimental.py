from .. import IRIS, IrisCommand
from .. import state_types as t
from .. import state_machine as sm
from .. import util as util
from ..event_loop import WorkLoop

# If I want to collect and stich together functions, need
# 1: retrieved functions should be storable state machines
# 2: when user paramaterizes them, should ask what are paramaters of workflow and what are constants
# 3: should be able to generate partial function based on user response, and store that
# 4: then stitch together partial functions and ask for metadata

# Notes, I can get this (below) to work, just fancier version of jump

class Workflow(IrisCommand):
    title = "workflow"
    examples = []
    def command(self, test : WorkLoop()):
        return test

workflow = Workflow()

class Add(sm.Function):
    title = "Add two numbers"
    argument_types = {
        "x": t.Int(),
        "y": t.Int()
    }
    def command(self, x, y):
        return x + y

class Subtract(sm.Function):
    title = "Subtract two numbers"
    argument_types = {
        "x": Add(),
        "y": t.Int()
    }
    def command(self, x, y):
        return x - y

subtract_with_closure = sm.BoundFunction({
    "x": {"y": 3},
    "y": 50
}, Subtract())

class TestAdd(IrisCommand):
    title = "test nested function closure"
    argument_types = {
        "test": sm.CreatePartial(Subtract())
    }
    def command(self, test):
        return (test,)

testAdd = TestAdd()

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
