

class IrisType:
    name = "Root"
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
        if value in env and Int.is_type(env[value]):
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

class IrisValue:

    def __init__(self, value, name=None):
        self.value = value
        self.name = name
