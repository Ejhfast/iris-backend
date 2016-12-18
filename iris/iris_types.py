

class IrisType:
    name = "Root"
    question = "What is the value of {}?"
    @classmethod
    def convert_type(cls, value, env):
        if value in env and cls.is_type(env[value]):
            return True, env[value], value
        return False, "I couldn't find \"{}\" of type {} in the current program environment".format(value, cls.name), value

class Int(IrisType):
    name = "Int"
    def is_type(value):
        return isinstance(value, int)
    @classmethod
    def convert_type(cls, value, env):
        if value in env and Int.is_type(env[value]):
            return True, env[value], value
        else:
            try:
                return True, int(value), value
            except:
                return False, "I want an {} type, but couldn't find one for \"{}\"".format(cls.name, value), value

class String(IrisType):
    name = "String"
    def is_type(value):
        return isinstance(value, str)
    @classmethod
    def convert_type(cls, value, env):
        if value in env and String.is_type(env[value]):
            return True, env[value], value
        else:
            try:
                return True, str(value), value
            except:
                return False, "I want an {} type, but couldn't find one for \"{}\"".format(cls.name, value), value

class List(IrisType):
    name = "List"
    def is_type(value):
        return isinstance(value,list)

class Any(IrisType):
    name = "Any"
    def is_type(value): return True

class ArgList(IrisType):
    name = "ArgList"
    question = "What is the value of {}? (If multiple values, please seperate by commas.)"
    def is_type(value): return True
    @classmethod
    def convert_type(cls, value, env):
        elements = [x.strip() for x in value.split(",")]
        if all([e in env and cls.is_type(env[e]) for e in elements]):
            return True, [env[e] for e in elements], value
        else:
            return False, "I want an {} type, but couldn't find one for \"{}\"".format(cls.name, value), value

class Name(String):
    name = "ArgList"
    question = "What name should I use to store the result?"
    id = 0 # we need to move to type instances
    def is_type(value): return True
    @classmethod
    def convert_type(cls, value, env):
        try:
            return True, str(value), value
        except:
            return False, "I want an {} type, but couldn't convert \"{}\"".format(cls.name, value), value

class IrisValue:

    def __init__(self, value, name=None):
        self.value = value
        self.name = name

class IrisImage(IrisValue):
    type="Image"

class IrisModel(IrisValue):
    type="Model"
    def __init__(self, model, X, y, name):
        self.X = X
        self.y = y
        self.model = model
        self.name = name
        self.value = self
