import numpy as np

class IrisType:
    name = "Root"
    question_txt = "What is the value of {}?"

    def __init__(self, question=None):
        self.class_ = type(self).__name__
        if question:
            self.question_txt = question

    def question(self, x):
        return [self.question_txt.format(x)]

    def is_type(self, value):
        return True

    def fail_check(self, value, env):
        return False, "I couldn't find \"{}\" of type {} in the current program environment".format(value, self.class_), value

    def convert_type(self, value, env):
        value = value.lower()
        if value in env and self.is_type(env[value]):
            return True, env[value], value
        else:
            return self.fail_check(value, env)

class Int(IrisType):
    name = "Int"
    question_txt = "Please provide an integer value for {}."

    def is_type(self, value):
        return isinstance(value, int)

    def fail_check(self, value, env):
        try:
            return True, int(value), value
        except:
            return False, "I want an {} type, but couldn't find one for \"{}\"".format(self.class_, value), value

class String(IrisType):
    name = "String"
    question_txt = "Please provide an string value for {}."

    def is_type(self, value):
        return isinstance(value, str)

    def fail_check(self, value, env):
        try:
            return True, str(value), value
        except:
            return False, "I want an {} type, but couldn't find one for \"{}\"".format(self.class_, value), value

class List(IrisType):
    name = "List"
    question_txt = "Please enter the name of a list variable in the environment for {}."

    def is_type(self, value):
        return isinstance(value,list)

class Array(IrisType):
    name = "Array"
    question_txt = "Please enter the name of an array variable in the environment for {}."

    def is_type(self, value):
        return isinstance(value,np.ndarray)

class Any(IrisType):
    name = "Any"
    def is_type(self, value): return True

class ArgList(IrisType):
    name = "ArgList"
    question_txt = "What is the value of {}? (If multiple values, please seperate by commas.)"
    def is_type(self, value): return True

    def fail_check(self, value, env):
        return False, "I want an {} type, but couldn't find one for \"{}\"".format(self.class_, value), value

    def convert_type(self, value, env):
        elements = [x.strip() for x in value.split(",")]
        if all([e in env and self.is_type(env[e]) for e in elements]):
            return True, [env[e] for e in elements], value
        else:
            self.fail_check(value, env)

class StoreName(String):
    name = "Name"
    question_txt = "What name should I use to store the result of this computation?"
    global_id = 0

    def __init__(self, name=None, question=None):
        self.class_ = self.class_ = type(self).__name__ # annoying duplication. super?
        if name:
            self.name = name
        if question:
            self.question_txt = question

    def is_type(self, value): return True

    def fail_check(self, value, env):
        try:
            StoreName.global_id += 1 # sketchy
            return True, IrisId(str(value),StoreName.global_id), value
        except:
            return False, "I want an {} type, but couldn't find one for \"{}\"".format(self.class_, value), value

class Select(IrisType):
    name = "Option"

    def __init__(self, options, default=None):
        id2option, id2data, data2id = {}, {}, {}
        self.default = None
        for i,d in enumerate(options.items()):
            option_text, option_data = d
            id2option[i] = option_text
            id2data[i] = option_data
            data2id[option_data] = i
            if option_data == default:
                self.default = i
        self.id2option = id2option
        self.id2data = id2data
        self.data2id = data2id

    def question(self, x):
        begin = ["Please select one of the following for \"{}\":".format(x)]
        default = []
        options = ["{}: {}".format(i,o) for i,o in self.id2option.items()]
        if self.default:
            default.append("The default for \"{}\" is option {}, {}".format(x, self.default, self.id2data[self.default]))
        return begin + default + options

    def is_type(self, value): return value in self.id2option.keys()

    def fail_check(self, value, env):
        try:
            intv = int(value)
            if intv in self.id2option:
                return True, self.id2data[intv], value
        except:
            if value in self.data2id:
                return True, value, value
        return False, "\"{}\" was not a valid option".format(value), value

class IrisValue:
    def __init__(self, value, name=None):
        self.value = value
        self.name = name

class IrisValues(IrisValue):
    def __init__(self, values, names):
        self.values = values
        self.names = names

class IrisId(IrisValue):
    def __init__(self, value, id, name=None):
        self.value = value
        self.id = id
        if not name:
            self.name = value
        else:
            self.name = name

class IrisImage(IrisId):
    type="Image"

class IrisModel(IrisValue):
    type="Model"
    def __init__(self, model, X, y):
        self.X = X
        self.y = y
        self.model = model

class IrisData(IrisValue):
    type="Data"
    def __init__(self, xvals, yvals):
        self.X = xvals
        self.y = yvals
