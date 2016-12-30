from collections import defaultdict
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
import sys
import os
from .iris_types import IrisValue, IrisImage, Int, IrisType, Any, List, String, ArgList, Name, IrisModel, IrisId, Array, Select, IrisValues, IrisData

class IrisBase:

    def __init__(self):
        self.mappings = {}
        self.cmd2class = {}
        self.class2cmd = defaultdict(list)
        self.class_functions = {}
        self.model = LogisticRegression()
        self.vectorizer = CountVectorizer()
        self.env = {}
        self.env_order = {}
        self.class2title = {}
        self.class2format = {}

    def iris(self):
        return self

    def train_model(self):
        x_docs, y = zip(*[(k, v) for k,v in self.cmd2class.items()])
        x = self.vectorizer.fit_transform(x_docs)
        self.model.fit(x,y)

    def predict_input(self, query):
        return self.model.predict_proba(self.vectorizer.transform([query]))

    def learn_from_example(self, cls_idx, query_string, arg_triple):
        succs = [x[2] for x in arg_triple]
        if not all(succs):
            return False, None
        else:
            arg_map = {}
            for name,val,_ in arg_triple: arg_map[val] = name
            transform = []
            query_words = query_string.lower().split()
            for w in query_words:
                if w in arg_map:
                    transform.append("{"+arg_map[w]+"}")
                else:
                    transform.append(w)
            command_string = " ".join(transform)
            if command_string in self.cmd2class: return False, None
            self.cmd2class[command_string] = cls_idx
            self.class2cmd[cls_idx].append(command_string)
            self.train_model()
            return True, command_string

    def get_predictions(self, text, n=1):
        predictions = self.predict_input(text)[0].tolist()
        sorted_predictions = sorted([(i,self.class2cmd[i],x) for i,x in enumerate(predictions)],key=lambda x: x[-1], reverse=True)
        return sorted_predictions[:n]


    # placeholder for something that needs to convert string input into a python value
    def magic_type_convert(self, x, type_):
        return type_.convert_type(x, self.env)

IRIS = IrisBase()

class IrisWatcher(type):

    def __init__(cls, name, bases, clsdict):
        # only trigger if this is third in inheritance chain
        # e.g., AddCommand, IrisCommand, IrisWatcher
        # this prevents triggering on definition of IrisCommand

        iris = IRIS

        if len(cls.mro()) > 2:
            title, examples, explanation, func = cls.title, cls.examples, cls.explanation, cls.command
            inner_examples = [title]
            if len(examples) > 0:
                inner_examples = examples
            # sketchy hack to get function arg names CPython
            f_args = func.__code__.co_varnames[:func.__code__.co_argcount]
            if any(func.__annotations__):
                f_types = func.__annotations__
            elif cls.argument_types:
                f_types = cls.argument_types
            else:
                raise Exception("Need annotations on command types")
            print(f_types)
            # the unique index for this function
            new_index = len(iris.class_functions)
            iris.class_functions[new_index] = {"function":cls.wrap_command, "args":f_args, "types":f_types}
            iris.class2title[new_index] = title
            iris.class2format[new_index] = explanation
            for command_string in inner_examples:
                lower_command = command_string.lower()
                print(new_index, lower_command)
                iris.mappings[lower_command] = {"function":cls.wrap_command, "args":f_args}
                iris.cmd2class[lower_command] = new_index
                iris.class2cmd[new_index].append(lower_command)
            # here i'll put the initializations that were formerly done by iris.register
        super(IrisWatcher, cls).__init__(name, bases, clsdict)


# Base class for iris command, which new commands will extend
class IrisCommand(metaclass = IrisWatcher):
    title = None
    examples = None
    argument_types = None

    # the logic of a given command
    def command(*args):
        pass

    # how the present the result of a command to the user
    # by default we will return a string representation
    def explanation(*args):
        if len(args) == 1:
            return str(args[0])
        else:
            return str(args)

    # how iris should store the results of this command
    # by default return the original value (do nothing)
    def memory(*args):
        if len(args) == 1:
            return args[0]
        elif len(args) == 0:
            return None
        else:
            return args

    @classmethod
    def wrap_command(self, *args, **kwargs):
        result = self.command(*args, **kwargs)
        memory_value = self.memory(*result)
        print(memory_value)
        if isinstance(memory_value, IrisValue):
            iris = IRIS
            # for id values, we are keeping the iris object
            if isinstance(memory_value, IrisId):
                iris.env[memory_value.name] = memory_value
                iris.env_order[memory_value.name] = len(iris.env_order)
            elif isinstance(memory_value, IrisValues):
                for name, value in zip(memory_value.names, memory_value.values):
                    iris.env[name] = value
                    iris.env_order[name] = len(iris.env_order)
            else:
                iris.env[memory_value.name] = memory_value.value
                iris.env_order[memory_value.name] = len(iris.env_order)
        return memory_value
