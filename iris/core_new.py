from collections import defaultdict
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
import sys
import os
from . import iris_types as t
from . import util


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

# Base class for iris command, which new commands will extend
class IrisCommand:
    title = None
    examples = None
    argument_types = None
    store_result = None
    result_questions = None

    # next step, wrap the function here with extra types specified by the storage option
    def __init__(self, iris=IRIS):
        self.iris = iris
        # command, etc. are static class-level commands, don't want a reference to instance object
        class_ = self.__class__
        title, examples, explanation, func = self.title, self.examples, class_.explanation, class_.command
        inner_examples = [title]
        if len(examples) > 0:
            inner_examples = examples
        # sketchy hack to get function arg names CPython
        f_args = func.__code__.co_varnames[:func.__code__.co_argcount]
        self.num_args = len(f_args)
        if any(func.__annotations__):
            f_types = func.__annotations__
        elif self.argument_types:
            f_types = self.argument_types
        else:
            raise Exception("Need annotations on command types")
        if self.store_result:
            for i, store_val in enumerate(util.single_or_list(self.store_result)):
                if store_val:
                    name_ = "name{}".format(i)
                    f_types[name_] = store_val
                    f_args = f_args + (name_,)
        # the unique index for this function
        new_index = len(iris.class_functions)
        iris.class_functions[new_index] = {"function":self.wrap_command, "args":f_args, "types":f_types}
        iris.class2title[new_index] = title
        iris.class2format[new_index] = explanation
        for command_string in inner_examples:
            lower_command = command_string.lower()
            print(new_index, lower_command)
            iris.mappings[lower_command] = {"function":self.wrap_command, "args":f_args}
            iris.cmd2class[lower_command] = new_index
            iris.class2cmd[new_index].append(lower_command)

    # make class instances behave like functions
    def __call__(self, *args):
        class_ = self.__class__
        return class_.command(*args)

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

    def wrap_command(self, *args, **kwargs):
        name = None
        if self.store_result:
            names = args[self.num_args:]
            args = args[:self.num_args]
        results = self(*args, **kwargs)
        if self.store_result:
            if isinstance(results, tuple):
                if len(names) != len(results):
                    raise Exception("Expected {} return values, got {}".format(len(names), len(results)))
            else:
                if len(names) > 1:
                    raise Exception("Expected {} return values, got only 1".format(len(names)))
            for name, result in zip(names, util.single_or_list(results)):
                self.iris.env[name.name] = result
                self.iris.env_order[name.name] = len(self.iris.env_order)
        return results

        #
        # class_ = self.__class__
        # result = class_.command(*args, **kwargs)
        # if isinstance(result, tuple):
        #     memory_value = class_.memory(*result)
        # else:
        #     memory_value = class_.memory(result)
        # print(memory_value)
        # if isinstance(memory_value, IrisValue):
        #     iris = self.iris
        #     # for id values, we are keeping the iris object
        #     if isinstance(memory_value, IrisId):
        #         iris.env[memory_value.name] = memory_value
        #         iris.env_order[memory_value.name] = len(iris.env_order)
        #     elif isinstance(memory_value, IrisValues):
        #         for name, value in zip(memory_value.names, memory_value.values):
        #             iris.env[name] = value
        #             iris.env_order[name] = len(iris.env_order)
        #     else:
        #         iris.env[memory_value.name] = memory_value.value
        #         iris.env_order[memory_value.name] = len(iris.env_order)
        # return memory_value
