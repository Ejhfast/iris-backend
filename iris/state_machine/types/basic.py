from ... import util
from ..model import IRIS_MODEL
from ... import state_machine as sm
from ... import iris_objects
import numpy as np

IRIS = IRIS_MODEL

def OR(tuple_list):
    for tuple in tuple_list:
        if tuple[0] == True:
            return tuple
    return False, None

def primitive_or_question(object, text, doing_match):
    if isinstance(object, sm.StateMachine):
        return object.convert_type(text, doing_match)
    return object == text, text

def is_pronoun(text):
    if text in ["this", "that", "these", "those", "it"]:
        return True
    return False

class EnvVar(sm.AssignableMachine):
    def __init__(self, question="Please give me a value for {}:", iris=IRIS):
        self.iris = iris
        self.question = question
        super().__init__()

    def string_representation(self, value):
        if isinstance(value, iris_objects.IrisValue):
            return value.name
        return str(value)

    def get_output(self):
        return [self.question.format(self.arg_name)]

    def error_message(self, text):
        return ["I could not find '{}' in the environment".format(text)]

    def is_type(self, x):
        return True

    def type_from_string(self, x):
        return False, None

    def convert_type(self, text, doing_match=False):
        if is_pronoun(text):
            if not doing_match: self.assign(iris_objects.EnvReference("__MEMORY__"), name=text)
            return True, iris_objects.EnvReference("__MEMORY__")
        elif text in self.iris.env and self.is_type(self.iris.env[text]):
            if not doing_match: self.assign(iris_objects.EnvReference(text), name=text)#self.iris.env[text], name=text)
            return True, iris_objects.EnvReference(text)#self.iris.env[text]
        else:
            success, result = self.type_from_string(text)
            if success:
                if not doing_match: self.assign(result, name=text)
                return True, result
            return False, self.error_message(text)

    def base_hint(self, text):
        success, _ = self.convert_type(text, doing_match=True)
        if success:
            return ["'{}' works as an arg".format(text)]
        else:
            return []

    def next_state_base(self, text):
        success, result = self.convert_type(text)
        if success: return result
        return self.set_error(result)

class Int(EnvVar):
    def is_type(self, x):
        if isinstance(x, int): return True
        return False

    def error_message(self, text):
        return ["I could not find '{}' in the environment or convert it to an integer. Please try again:".format(text)]

    def type_from_string(self, x):
        try:
            result = int(x)
            return True, result
        except:
            return False, None

class String(EnvVar):
    def is_type(self, x):
        if isinstance(x, str): return True
        return False

    def error_message(self, text):
        return ["I could not find '{}' in the environment or convert it to an string. Please try again:".format(text)]

    def type_from_string(self, x):
        return True, x.replace("\"","") # sketch

class Array(EnvVar):
    def is_type(self, x):
        if isinstance(x, np.ndarray): return True
        return False

    def error_message(self, text):
        return ["I could not find '{}' in the environment or convert it to an Array. Please try again:".format(text)]

class Function(EnvVar):
    def is_type(self, x):
        if isinstance(x, iris_objects.FunctionWrapper): return True
        return False

    def error_message(self, text):
        return ["I could not find '{}' in the environment. Please try again:".format(text)]

class ArgList(EnvVar):
    def is_type(self, x):
        return True, x

    def string_representation(self, value):
        return 'LIST OF {}'.format(self.arg_name)

    def error_message(self, text):
        return ["I couldn't parse that. Please try again:".format(text)]

    def convert_type(self, text):
        elements = [x.strip() for x in text.split(",")]
        if all([e in self.iris.env and self.is_type(self.iris.env[e]) for e in elements]):
            self.assign([self.iris.env[e] for e in elements])
            return True, [self.iris.env[e] for e in elements]
        return False, self.error_message(text)

class File(EnvVar):
    def is_type(self, x):
        try:
            f = open(x, "r")
            f.close()
        except:
            return False
        return True

    def get_content(self, x):
        with open(x, "r") as f:
            return f.read()#.decode('utf-8')

    def error_message(self, text):
        return ["I couldn't find {} in the environment. Please try again.".format(text)]

    def convert_type(self, text, doing_match=False):
        if self.is_type(text):
            content = self.get_content(text)
            obj = iris_objects.IrisFile(text, content)
            self.assign(obj, name=text)
            return True, obj#)#.when_done(self.get_when_done_state())
        return False, self.error_message(text)

class VarName(sm.AssignableMachine):
    global_id = 0
    def __init__(self, question="Please give me a variable name"):
        super().__init__()
        self.output = [question]
    def convert_type(self, text, doing_match=False):
        return True, iris_objects.IrisId(text, VarName.global_id)
    def next_state_base(self, text):
        success, result = self.convert_type(text)
        self.assign(result, text)
        VarName.global_id += 1
        return result
