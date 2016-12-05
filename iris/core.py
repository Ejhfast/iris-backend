import shlex
import random
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from collections import defaultdict
import numpy as np
from . import util
from .iris_types import IrisValue, IrisImage, Int, IrisType, Any, List, String, ArgList, Name, IrisModel

class Iris:

    def __init__(self):
        self.mappings = {}
        self.cmd2class = {}
        self.class2cmd = defaultdict(list)
        self.class_functions = {}
        self.model = LogisticRegression()
        self.vectorizer = CountVectorizer()
        self.env = {}

    def train_model(self):
        x_docs, y = zip(*[(k, v) for k,v in self.cmd2class.items()])
        x = self.vectorizer.fit_transform(x_docs)
        self.model.fit(x,y)

    def predict_input(self, query):
        return self.model.predict_proba(self.vectorizer.transform([query]))

    def gen_example(self, cls_idx, query_string, arg_triple):
        succs = [x[2] for x in arg_triple]
        if not all(succs):
            print("won't generate an example")
            return False, None
        else:
            print("in else")
            arg_map = {}
            for name,val,_ in arg_triple: arg_map[val] = name
            transform = []
            query_words = query_string.lower().split()
            print(arg_map)
            for w in query_words:
                if w in arg_map:
                    transform.append("{"+arg_map[w]+"}")
                else:
                    transform.append(w)
            command_string = " ".join(transform)
            if command_string in self.cmd2class: return False, None
            self.cmd2class[command_string] = cls_idx
            self.class2cmd[cls_idx].append(command_string)
            print(command_string)
            self.train_model()
            return True, command_string

    # Thanksgiving

    def get_predictions(self, text, n=1):
        predictions = self.predict_input(text)[0].tolist()
        sorted_predictions = sorted([(i,self.class2cmd[i],x) for i,x in enumerate(predictions)],key=lambda x: x[-1], reverse=True)
        return sorted_predictions[:n]

    # placeholder for something that needs to convert string input into a python value
    def magic_type_convert(self, x, type_):
        return type_.convert_type(x, self.env)

    # is this word an argument?
    def is_arg(self, s):
        if len(s)>2 and s[0] == "{" and s[-1] == "}": return True
        return False

    # attempt to match query string to command and return mappings
    def arg_match(self, query_string, command_string):#, types):
        maps = {}
        labels = []
        query_words, cmd_words = [shlex.split(x) for x in [query_string, command_string]]
        if len(query_words) != len(cmd_words): return False, {}
        for qw, cw in zip(query_words, cmd_words):
            if self.is_arg(cw):
                word_ = cw[1:-1]
                maps[word_] = qw #self.magic_type_convert(qw, types[word_])
            else:
                if qw != cw: return False, {}

        return True, maps

    def ctx_wrap(self, func):
        def inner(*args, **kwargs):
            result = func(*args, **kwargs)
            if isinstance(result, IrisValue):
                self.env[result.name] = result.value
            # else:
            #     self.env["results"].append(result)
            return result
        return inner

    def register(self, command_string):
        def inner(func):
            # sketchy hack to get function arg names CPython
            f_args = func.__code__.co_varnames[:func.__code__.co_argcount]
            f_types = func.__annotations__
            self.mappings[command_string] = {"function":self.ctx_wrap(func), "args":f_args}
            new_index = len(self.cmd2class)
            self.cmd2class[command_string] = new_index
            self.class2cmd[new_index].append(command_string)
            self.class_functions[new_index] = {"function":self.ctx_wrap(func), "args":f_args, "types":f_types}
            return self.ctx_wrap(func)
        return inner
