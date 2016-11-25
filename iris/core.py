import shlex
import random
import dill
from .shell import IrisShell
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from collections import defaultdict
import numpy as np


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

class Iris:

    def __init__(self):
        self.mappings = {}
        self.cmd2class = {}
        self.class2cmd = defaultdict(list)
        self.class_functions = {}
        self.model = LogisticRegression()
        self.vectorizer = CountVectorizer()
        self.env = {"results":[]}

    def train_model(self):
        x_docs, y = zip(*[(k, v) for k,v in self.cmd2class.items()])
        x = self.vectorizer.fit_transform(x_docs)
        self.model.fit(x,y)

    def predict_input(self, query):
        return self.model.predict_log_proba(self.vectorizer.transform([query]))

    def make_labels(self,query,cmd,succ):
        label = 0
        labels = []
        cw_words = cmd.split()
        for i,w in enumerate(query.split()):
            if i < len(cw_words) and self.is_arg(cw_words[i]) and succ:
                label += 1
                labels.append({"text":w,"index":i,"label":label})
            else:
                labels.append({"text":w,"index":i,"label":0})
        return labels

    def gen_example(self, cls_idx, query_string, arg_triple):
        print("doing gen")
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

    def verify_response(self, text):
        if text.lower() == "yes":
            return True
        else:
            return False

    # this is for parsing args out of iris conversation
    def parse_args(self, messages):
        fail_indexes = [i for i,x in enumerate(messages) if x["origin"] == "iris" and x["state"] == "RESOLVE_ARGS" ]
        args = {}
        for i in fail_indexes:
            iris_ask = messages[i]["text"]
            var = iris_ask.split()[-1][:-1]
            args[var] = messages[i+1]["text"]
        return args

    def process_succ_failure(self, cls_idx, s_args, arg_names, query):
        succs = [x[0] for x in s_args]
        args = [x[1] for x in s_args]
        values = [x[2] for x in s_args]
        triples = list(zip(arg_names,args,succs))
        for_ex = list(zip(arg_names,values,succs))
        if all(succs):
            learn, lcmd = self.gen_example(cls_idx, query, for_ex)
            result = self.class_functions[cls_idx]["function"](*args)
            if learn:
                result = ["(I learned how to \"{}\")".format(lcmd),str(result)]
            else:
                result = [str(result)]
            return result
        else:
            # kind of obnoxious that I am doing string construction here...
            assump = lambda x,y: "I think {} is \"{}\"".format(x,y)
            problem = lambda x,y: "For {}, {}".format(x,y)
            arg_assumptions = [assump(x[0],x[1]) for x in triples if x[2] == True]
            arg_problems = [problem(x[0],x[1]) for x in triples if x[2] == False]
            return ["I ran into a problem:"]+arg_assumptions+arg_problems

    # may want some distinction between state returned and input state
    def state_machine(self, data):
        state = data["state"]
        messages = data["messages"]
        if state == "START":
            # since this is starting an interaction, we only need to look at the first/last message
            text = messages[-1]["text"]
            class_id, class_text, _ = self.get_predictions(text)[0] #None, None # do classification(text)
            # keep track of class_id so we can use it later
            return { "state": "CLASSIFICATION", "text": ["I think you want to \"{}\". Is that correct?".format(class_text[0])], "id": class_id }
        elif state == "CLASSIFICATION":
            # the last message will hold the user yes/no response
            text = messages[-1]["text"]
            verify = self.verify_response(text)
            # since user accepted this task we will try to resolve args now
            if verify:
                return self.state_machine({"state": "RESOLVE_ARGS", "messages":messages, "id": data["id"], "arg_map": {} })
            # user doesn't want this, so start over
            else:
                return {"state": "START", "text": ["Okay, what would you like to do?"] }
        elif state == "RESOLVE_ARGS":
            # the first message is the one we will be using for argument extraction from base command
            text = messages[0]["text"]
            function_data = self.class_functions[data["id"]] # this gets types, args, and function executable
            cmd_names = self.class2cmd[data["id"]] # list of possible command strings
            message_args = self.parse_args(messages)
            # case 1: all args are in arg_map derived from messages
            if all([arg in message_args for arg in function_data["args"]]):
                arg_list = [self.magic_type_convert(message_args[arg], function_data["types"][arg]) for arg in function_data["args"]]
                return self.state_machine({"state": "EXECUTE", "id": data["id"], "messages": messages, "text": text,
                                    "arg_list": arg_list, "arg_names": function_data["args"]})
            # case 2: all args can be inferred from matches with one of the command strings
            for cmd in cmd_names:
                succ, map_ = self.arg_match(text, cmd, function_data["types"])
                print(succ, map_)
                if succ:
                    arg_list = [map_[arg] for arg in function_data["args"]]
                    return self.state_machine({"state": "EXECUTE", "id": data["id"], "messages": messages, "text": text,
                                        "arg_list": arg_list, "arg_names": function_data["args"]})
            # case 3: ask user for some number of args
            for arg in function_data["args"]:
                if (not arg in message_args):
                    return {"state": "RESOLVE_ARGS", "text": ["What is the value of {}?".format(arg)], "id":data["id"] }
            return None
            # arg resolution logic, confusing
        elif state == "EXECUTE":
            return {"state":"START", "text": self.process_succ_failure(data["id"], data["arg_list"], data["arg_names"], data["text"])}
            # execute logic

    # returns "Ask", "Success", or "Failure" + message
    def loop(self, query_string, arg_map):
        # helper for aggregating over args
        def process_succ_failure(cls_idx, s_args, arg_names):
            succs = [x[0] for x in s_args]
            args = [x[1] for x in s_args]
            values = [x[2] for x in s_args]
            triples = list(zip(arg_names,args,succs))
            for_ex = list(zip(arg_names,values,succs))
            if all(succs):
                learn, lcmd = self.gen_example(cls_idx, query_string, for_ex)
                result = to_execute["function"](*args)
                if learn:
                    result = ["(I learned how to \"{}\")".format(lcmd),str(result)]
                else:
                    result = [str(result)]
                return "Success", result
            else:
                # kind of obnoxious that I am doing string construction here...
                assump = lambda x,y: "I think {} is \"{}\"".format(x,y)
                problem = lambda x,y: "For {}, {}".format(x,y)
                arg_assumptions = [assump(x[0],x[1]) for x in triples if x[2] == True]
                arg_problems = [problem(x[0],x[1]) for x in triples if x[2] == False]
                return "Failure", ["I ran into a problem:"]+arg_assumptions+arg_problems
        # first get best prediction
        predictions = self.predict_input(query_string)[0].tolist()
        sorted_predictions = sorted([(i,self.class2cmd[i],x) for i,x in enumerate(predictions)],key=lambda x: x[-1], reverse=True)
        best_class = sorted_predictions[0][0]
        to_execute = self.class_functions[best_class]
        # now check arg_map
        if all([arg_name in arg_map for arg_name in to_execute["args"]]):
            # we were given all the arguments, now do type conversion
            s_args = [self.magic_type_convert(arg_map[arg_name], to_execute["types"][arg_name]) for arg_name in to_execute["args"]]
            return process_succ_failure(best_class, s_args, to_execute["args"])
        # else we don't have all the args, so try to match
        for cmd in self.class2cmd[best_class]:
            succ, map = self.arg_match(query_string, cmd, to_execute["types"])
            if succ:
                s_args = [map[arg_name] for arg_name in to_execute["args"]]
                return process_succ_failure(best_class, s_args, to_execute["args"])
        # if all that fails we need to start asking for args
        for arg_name in to_execute["args"]:
            if not (arg_name in arg_map):
                if len(arg_map) == 0: # implies this is the first argument asked for
                    return "Ask", ["I think you want to \"{}\". What is the value of {}?".format(cmd,arg_name)]
                return "Ask", ["What is the value of {}?".format(arg_name)]

    def best_n(self, query, n=1):
        probs = self.model.predict_log_proba(self.vectorizer.transform([query]))[0]
        results = []
        for i,p in sorted(enumerate(probs),key=lambda x: x[1],reverse=True):
            for cmd in self.class2cmd[i]:
                succ, map = self.arg_match(query, cmd)
                if succ: break
            print(query,cmd,succ)
            labels = self.make_labels(query,cmd,succ)
            results.append({
                "class":i,
                "prob":p,
                "cmds": self.class2cmd[i],
                "args": len(self.class_functions[i]["args"]),
                "labels": labels
            })
        return results[0] # just returning one for now

    def call_class(self, class_, args):
        to_execute = self.class_functions[class_]
        num_args = max([x["label"] for x in args])
        extracted_args = []
        cmd_words = [x for x in self.class2cmd[class_][0].split() if self.is_arg(x)]
        for i,ref in zip(range(1,num_args+1),cmd_words):
            ref_type =  ref[1:-1].split(":")[1].strip()
            argi = " ".join([x["text"] for x in args if x["label"] == i])
            argt = self.magic_type_convert(argi,ref_type)
            extracted_args.append(argt)
        print(extracted_args)
        res =  to_execute["function"](*extracted_args)
        print(res)
        return res

    # placeholder for something that needs to convert string input into a python value
    def magic_type_convert(self, x, type_):
        return type_.convert_type(x,self.env)

    # is this word an argument?
    def is_arg(self, s):
        if len(s)>2 and s[0] == "{" and s[-1] == "}": return True
        return False

    def get_user_args(self, cmd):
        maps = {}
        for w in shlex.split(cmd):
            if self.is_arg(w):
                word_, type_ = w[1:-1].split(":")
                print("\tWhat is {} in the command: \"{}\"".format(w,cmd))
                data = input()
                maps[word_] = self.magic_type_convert(data.strip(),type_)
        return maps

    # attempt to match query string to command and return mappings
    def arg_match(self, query_string, command_string, types):
        maps = {}
        labels = []
        query_words, cmd_words = [shlex.split(x) for x in [query_string, command_string]]
        if len(query_words) != len(cmd_words): return False, {}
        for qw, cw in zip(query_words, cmd_words):
            if self.is_arg(cw):
                word_ = cw[1:-1]
                maps[word_] = self.magic_type_convert(qw, types[word_])
            else:
                if qw != cw: return False, {}
        return True, maps

    def ctx_wrap(self, func):
        def inner(*args, **kwargs):
            result = func(*args, **kwargs)
            if isinstance(result, IrisValue):
                self.env[result.name] = result.value
            else:
                self.env["results"].append(result)
            return result
        return inner

    def execute(self, query_string):
        predictions = self.predict_input(query_string)[0].tolist()
        sorted_predictions = sorted([(i,self.class2cmd[i],x) for i,x in enumerate(predictions)],key=lambda x: x[-1], reverse=True)
        best_class = sorted_predictions[0][0]
        to_execute = self.class_functions[best_class]
        for cmd in self.class2cmd[best_class]:
            succ, map = self.arg_match(query_string, cmd)
            if succ:
                args = [map[arg_name] for arg_name in to_execute["args"]]
                return True, to_execute["function"](*args)
        args_map = self.get_user_args(cmd)
        if args_map:
            args = [args_map[arg_name] for arg_name in to_execute["args"]]
            return True, to_execute["function"](*args)
        # for cmd in self.mappings.keys():
        #     succ, map = self.arg_match(query_string, cmd)
        #     if succ:
        #         to_execute = self.mappings[cmd]
        #         args = [map[arg_name] for arg_name in to_execute["args"]]
        #         return True, to_execute["function"](*args)
        return False, None

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

    def env_loop(self):
        def main_loop(x):
            try:
                succ, r = self.execute(x)
                if succ:
                    if isinstance(r, IrisValue):
                        print("\t{}".format(r.value))
                        print("\t(stored in {})".format(r.name))
                    else:
                        print("\t{}".format(r))
                else:
                    print("\tDid not match any existing command")
            except Exception as e:
                print("\tSomething went wrong: {}".format(e))
        shell = IrisShell(main_loop)
        shell.cmdloop()
