from collections import defaultdict
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
import sys
import os
from . import iris_types as t
from . import util


class IrisBase:

    def __init__(self):
        self.cmd2class = {}
        self.class2cmd = defaultdict(list)
        self.class_functions = {}
        self.model = LogisticRegression()
        self.vectorizer = CountVectorizer()
        self.env = {}
        self.env_order = {}

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
        sorted_predictions = sorted([(i,self.class2cmd[i],x) for i,x in enumerate(predictions)], key=lambda x: x[-1], reverse=True)
        return sorted_predictions[:n]

    # placeholder for something that needs to convert string input into a python value
    def magic_type_convert(self, x, type_):
        return type_.convert_type(x)+(x,)

IRIS = IrisBase()

class IrisCommand:
    title = None
    examples = []
    argument_types = {}
    store_result = None
    help_text = None
    context = {}

    def __init__(self, iris=IRIS):
        self.iris = iris
        # hack to get function arg names CPython (:1 to remove "self")
        self.command_args = self.command.__code__.co_varnames[:self.command.__code__.co_argcount][1:]
        self.all_args = self.command_args
        if any(self.command.__annotations__):
            self.argument_types = {**self.command.__annotations__, **self.argument_types}
        if not self.argument_types and len(self.command_args) > 0:
            raise Exception("Need annotations on command types")
        if self.store_result:
            for i, store_val in enumerate(util.single_or_list(self.store_result)):
                if store_val:
                    name_ = "name{}".format(i)
                    self.argument_types[name_] = store_val
                    self.all_args = self.all_args + (name_,)
        # the unique index for this function
        new_index = len(iris.class_functions)
        iris.class_functions[new_index] = self
        for command_string in self.training_examples():
            lower_command = command_string.lower()
            iris.cmd2class[lower_command] = new_index
            iris.class2cmd[new_index].append(lower_command)

    # make class instances behave like functions
    def __call__(self, *args):
        return self.command(*args)

    def num_command_args(self):
        return len(self.command_args)

    def training_examples(self):
        return [self.title] + self.examples

    # the logic of a given command
    def command(*args):
        pass

    def wrap_explanation(self, *args):
        results = []
        if self.store_result:
            names = [n.name for n in self.context["names"]]
            results.append("Stored data in " + " and ".join(names))
        for r in util.single_or_list(self.explanation(*args)):
            results.append(r)
        return results

    # how the present the result of a command to the user
    # by default we will return a string representation
    def explanation(self, *args):
        results = []
        for r in args:
            if isinstance(r, t.IrisImage):
                results.append({"type": "image", "value": r.value})
            elif util.is_data(r):
                results.append({"type": "data", "value": util.prettify_data(r)})
            else:
                results.append(str(r))
        return results

    # borrow context (e.g., list of storage names) before executing
    def with_context(self, context):
        def wrap(*args):
            old_context = self.context
            for k,v in context.items():
                self.context[k] = v
            result = self(*args)
            self.context = old_context
            return result
        return wrap

    def wrap_command(self, *args, **kwargs):
        name = None
        if self.store_result:
            names = args[self.num_command_args():]
            # store for possible use by command
            # note: if used by command, command is no longer stand-alone
            self.context["names"] = names
            args = args[:self.num_command_args()]
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
