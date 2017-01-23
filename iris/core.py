from collections import defaultdict
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
import sys
import os
from . import iris_objects
from . import util
from . import state_machine as sm

class IrisBase:

    def __init__(self):
        self.cmd2class = {}
        self.class2cmd = defaultdict(list)
        self.class_functions = {}
        self.model = LogisticRegression()
        self.vectorizer = CountVectorizer()
        self.env = {}
        self.env_order = {}
        self.history = {"history": [], "currentConvo": { 'messages': [], 'title': None, 'hidden': False, 'id': 0, 'args': {} }}

    def add_to_env(self, name, result):
        self.env[name] = result
        self.env_order[name] = len(self.env_order)

    def set_history(self, request):
        self.history = request["conversation"]

    def serialize_state(self):
        return {"env":self.env, "env_order":self.env_order, "history":self.history}

    def load_state(self, data):
        self.env = data["env"]
        self.env_order = data["env_order"]
        self.history = data["history"]

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
        print("learning")
        print(succs)
        if not all(succs):
            return False, None
        else:
            arg_map = {}
            for name,val,_ in arg_triple: arg_map[str(val)] = name
            transform = []
            query_words = query_string.lower().split()
            print(arg_map)
            for w in query_words:
                print(w, w in arg_map)
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

# omg these closures...
def gen_print_caller(help_state):
    return lambda caller: sm.Print([help_state]).when_done(caller)

# this one is more tricky, need to pass along when_done to the alternative help state machine
def gen_state_caller(help_state):
    def anon(caller):
        if caller.get_when_done_state():
            return help_state.add_middleware(sm.QuitMiddleware()).when_done(caller.get_when_done_state())
        return help_state
    return anon

class IrisCommand:
    title = None
    examples = []
    argument_types = {}
    argument_help = {}
    store_result = None
    help_text = [ "This command has no help text." ]
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
        for arg, type in self.argument_types.items():
            if arg in self.argument_help:
                help_state = self.argument_help[arg]
            else:
                help_state = "No help available for this argument."
            mw = []
            if isinstance(help_state, str):
                mw.append(sm.ExplainMiddleware(gen_print_caller(help_state)))
                self.argument_types[arg].add_middleware(mw)
            else:
                mw.append(sm.ExplainMiddleware(gen_state_caller(help_state.set_arg_name(arg))))
                self.argument_types[arg].add_middleware(mw)
        # the unique index for this function
        self.class_index = len(iris.class_functions)
        self.iris.class_functions[self.class_index] = self
        for command_string in self.training_examples():
            lower_command = command_string.lower()
            self.iris.cmd2class[lower_command] = self.class_index
            self.iris.class2cmd[self.class_index].append(lower_command)

    # make class instances behave like functions
    def __call__(self, *args):
        return self.command(*args)

    def get_class_index(self):
        return self.class_index

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
            if isinstance(r, iris_objects.IrisImage):
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
        if len(args) > 0:
            results = self(*args, **kwargs)
        else:
            results = self()
        if self.store_result:
            if isinstance(results, tuple):
                if len(names) == len(results):
                    for name, result in zip(names, util.single_or_list(results)):
                        self.iris.add_to_env(name.name, result)
                elif len(names) == 1:
                    self.iris.add_to_env(names[0].name, results)
                else:
                    raise Exception("Expected {} return values, got {}".format(len(names), len(results)))
            else:
                if len(names) > 1:
                    raise Exception("Expected {} return values, got only 1".format(len(names)))
                self.iris.add_to_env(names[0].name, results)
        self.iris.add_to_env("__MEMORY__", results)
        return results

    def state_machine_output(self, arg_map, arg_names, result, query):
        results = []
        for_ex = [(arg, arg_names[arg], True) for arg in self.all_args]
        args = [arg_map[arg] for arg in self.all_args]
        learn, lcmd = self.iris.learn_from_example(self.get_class_index(), query, for_ex)
        if learn:
            results.append("(I learned how to \"{}\")".format(lcmd))
        self.iris.train_model()
        explanations = self.wrap_explanation(result)
        if isinstance(explanations, list):
            for explanation in explanations:
                results.append(explanation)
        else:
            results.append(explanations)
        return results
