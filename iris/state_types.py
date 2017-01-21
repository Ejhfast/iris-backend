from . import util
from .core import IRIS
from . import state_machine as sm
from . import iris_objects
import numpy as np

def OR(tuple_list):
    for tuple in tuple_list:
        if tuple[0] == True:
            return tuple
    return False, None

def primitive_or_question(object, text, doing_match):
    if isinstance(object, sm.StateMachine):
        return object.convert_type(text, doing_match)
    return object == text, text

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
        if text in self.iris.env and self.is_type(self.iris.env[text]):
            if not doing_match: self.assign(self.iris.env[text], name=text)
            return True, self.iris.env[text]
        else:
            success, result = self.type_from_string(text)
            if success:
                if not doing_match: self.assign(result, name=text)
                return True, result
            return False, self.error_message(text)

    def next_state_base(self, text):
        success, result = self.convert_type(text)
        if success: return False, result
        return True, self.set_error(result)

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
        return True, x

class Array(EnvVar):
    def is_type(self, x):
        if isinstance(x, np.ndarray): return True
        return False

    def error_message(self, text):
        return ["I could not find '{}' in the environment or convert it to an Array. Please try again:".format(text)]

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
            if not doing_match: self.assign(content, name=text)
            return True, content
        return False, self.error_message(text)

class VarName(sm.AssignableMachine):
    global_id = 0
    def __init__(self, question="Please give me a variable name"):
        super().__init__()
        self.output = [question]
    def convert_type(self, text):
        return True, text
    def next_state_base(self, text):
        success, result = self.convert_type(text)
        result = iris_objects.IrisId(result, VarName.global_id)
        self.assign(result)
        VarName.global_id += 1
        return False, result

class YesNo(sm.AssignableMachine):
    def __init__(self, question, yes=None, no=None):
        self.yes = yes
        self.no = no
        super().__init__()
        if isinstance(question, list):
            self.output = question
        else:
            self.output = [question]

    def string_representation(self, value):
        if isinstance(value, str) or isinstance(value, int):
            return str(value)
        return "CHOICE FOR {}".format(self.arg_name)

    def convert_type(self, text):
        return OR([
            primitive_or_question(self.yes, text),
            primitive_or_question(self.no, text)
        ])

    def next_state_base(self, text):
        new_state = self
        if util.verify_response(text): new_state = self.yes
        else: new_state = self.no
        if isinstance(new_state, sm.StateMachine):
            return True, new_state
        else:
            self.assign(new_state)
            return False, new_state

    def when_done(self, state):
        if isinstance(self.yes, sm.StateMachine):
            self.yes.when_done(state)
        if isinstance(self.no, sm.StateMachine):
            self.no.when_done(state)
        self.when_done_state = state
        return self

class Select(sm.AssignableMachine):
    def __init__(self, options={}, option_info={}, default=None):
        super().__init__()
        self.default = default
        self.id2option = {}
        option_keys = sorted(options.keys())
        question_text = []
        #question_text.append()
        for i,k in enumerate(option_keys):
            self.id2option[i] = options[k]
            question_text.append("{}: {}".format(i,k))
            if options[k] in option_info:
                for m in option_info[options[k]]:
                    question_text.append({"type":"explain", "value":m})
        question_text.append("Would you like any of these?")
        self.output = question_text

    def string_representation(self, value):
        if isinstance(value, str):
            return value
        return "CHOICE FOR {}".format(self.arg_name)

    def get_output(self):
        if self.arg_name != None:
            message = "Please choose from one of the following for {}:".format(self.arg_name)
            return [message] + self.output
        return ["Please choose from one of the following:"] + self.output

    def error_message(self, text):
        return ["{} is not a valid option".format(text)]

    def convert_type(self, text, doing_match=False):
        return OR([primitive_or_question(value, text, doing_match) for _, value in self.id2option.items()])

    def next_state_base(self, text):
        new_state = self
        success, choice = util.extract_number(text)
        if success:
            if choice in self.id2option:
                new_state = self.id2option[choice]
                if isinstance(new_state, sm.StateMachine):
                    return True, new_state
                else:
                    self.assign(new_state)
                    return False, new_state
        return True, self.set_error(self.error_message(text))

    def when_done(self, next_state):
        for id, state in self.id2option.items():
            if isinstance(state, sm.StateMachine):
                state.when_done(next_state)
        self.when_done_state = next_state
        return self
