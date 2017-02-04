from .. import IrisCommand
from .. import state_types as t
from .. import state_machine as sm
from .. import util as util

class GetArrayLength(IrisCommand):
    title = "get length of array {arr}"
    examples = ["array {} length"]
    def command(self, arr: t.Array("What array to get length of?")):
        return arr.shape[0]

getArrayLength = GetArrayLength()


class GenerateNumber(IrisCommand):
    title = "generate a random number"
    examples = [ "generate number" ]
    def command(self):
        import random
        return random.randint(0,100)

generateNumber = GenerateNumber()

class StatsTest(IrisCommand):
    title = "stats test"
    examples = []
    argument_types = {
        "response": t.StatisticalState("Would you like to continue?",
            { "yes": {
                    "examples": ["yes", "yeah", "okay"],
                    "state": "Great, done." },
              "no": {
                    "examples": ["nope", "no thanks", "no"],
                    "state": "Bye." }
            }
        )
    }
    def command(self, response):
        return response

statsTest = StatsTest()

class GenerateArray(IrisCommand):
    title = "generate a random array of {n} numbers"
    examples = [ "generate numpy array of size {n}"]
    def command(self, n : t.Int("Please enter size of array:")):
        import numpy
        return numpy.random.randint(100, size=n)

generateArray = GenerateArray()

class AddTwoNumbers(IrisCommand):
    title = "add two numbers: {x} and {y}"
    examples = [ "add {x} and {y}",
                 "add {x} {y}" ]
    argument_types = {
        "x": t.Int("Please enter a number for x:"),
        "y": t.Int("Please enter a number for y:")
    }
    help_text = [
        "This command performs addition on two numbers, e.g., 'add 3 and 2' will return 5"
    ]
    def command(self, x, y):
        return x + y

addTwoNumbers = AddTwoNumbers()

class PrintValue(IrisCommand):
    title = "print {value}"
    examples = [ "display {value}", "{}"]
    help_text = [
        "This command will display the underlying data for environment variables."
    ]
    def command(self, value : t.EnvVar()):
        return value

printValue = PrintValue()
