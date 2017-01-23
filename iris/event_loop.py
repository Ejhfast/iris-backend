from . import state_machine as sm
from . import state_types as st
from .core import IRIS
from . import util
import sys
import uuid

class IrisMachine(sm.AssignableMachine):
    def __init__(self, iris = IRIS, output = None, recursed = False):
        self.iris = IRIS
        self.recursed = recursed
        super().__init__()
        # this marks variables with their scope
        self.uniq = str(uuid.uuid4()).upper()[0:10]
        if output:
            self.output = output
        else:
            self.output = ["Okay, what would you like to do?"]

    def reset(self):
        # need to reset..., possibly better way
        self.context["ASSIGNMENTS"] = {}
        self.context["ASSIGNMENT_NAMES"] = {}
        # create new scope
        self.uniq = str(uuid.uuid4()).upper()[0:10]
        return self

    # create option selection for the predicted command
    def compose_options(self, text, n):
        predictions = self.iris.get_predictions(text, n=n)
        cmd_objects = [self.iris.class_functions[x[0]] for x in predictions]
        option_dict = {cmd_o.title: cmd_o.get_class_index() for cmd_o in cmd_objects}
        option_info = {cmd_o.get_class_index(): cmd_o.help_text for cmd_o in cmd_objects}
        normal_options = st.Select(options=option_dict).add_middleware(sm.QuitMiddleware())
        explained_options = st.Select(options=option_dict, option_info=option_info).add_middleware(sm.QuitMiddleware())
        return normal_options.add_middleware(sm.ExplainMiddleware(lambda caller: explained_options))

    def next_state_base(self, text):
        # add a label, we might jump back here
        self.context["START"] = self
        # get initial prediction
        class_id, class_text, pred = self.iris.get_predictions(text)[0]
        cmd_object = self.iris.class_functions[class_id]
        explain_cmd = sm.ExplainMiddleware(lambda caller: sm.Print(cmd_object.help_text).when_done(caller))
        command_title = class_text[0]
        self.context[self.uniq + "_" + "query"] = text
        # create a variable to hold the predicted command class
        user_class = sm.Variable("user_class", scope=self.uniq)
        # this machine will resolve the args on that command
        resolve_args = ResolveArgs(user_class, uniq=self.uniq, recursed=self.recursed).when_done(self.get_when_done_state())
        # four alternative commands to present to user, if we don't want the one selected
        select_alternative = self.compose_options(text, 4)
        # ask whether we want these options
        options = st.YesNo("Would you like more options?",
                    # if we want something else, reassign that to the selected class
                    yes=sm.Assign(user_class, select_alternative).when_done(resolve_args),
                    no=sm.Jump("START"))
        # confirm whether we want to continue
        confirm = st.YesNo([
                        "Do you want to {}?".format(command_title),
                        "(My certainty is {})".format(round(pred, 5))],
                    yes=resolve_args,
                    no=options).add_middleware([explain_cmd, sm.QuitMiddleware()])
        # bind user_class to class_id, then run
        return True, sm.Let(user_class, equal=class_id, then_do=confirm)

    def when_done(self, new_state):
        self.when_done_state = new_state
        return self

class IrisMiddleware(sm.Middleware):
    def __init__(self, output):
        self.output = output
    def test(self, text):
        if text:
            return "command" in text
    def transform(self, caller, state_tuple):
        keep_going, state = state_tuple
        state.clear_error()
        return True, IrisMachine(output = self.output, recursed=True).when_done(caller.get_when_done_state())

class AskForArg(sm.StateMachine):
    def __init__(self, arg, uniq=""):
        super().__init__()
        self.accepts_input = False
        self.arg = arg
        self.uniq = uniq
        self.cmd_object = cmd_object
        self.done_state_list = done_state_list
    def next_state_base(self, text):
        iris_middle = IrisMiddleware(["Sure, we can run another function to generate {}.".format(arg),
                                      "What would you like to run?"])
        type_machine =  cmd_object.argument_types[arg].set_arg_name(arg).add_middleware(iris_middle).reset()
        arg_var = sm.Variable(arg, scope=self.uniq)
        verify_arg = sm.PrintVar(arg_var, util.print_assignment)
        assign_var = sm.Assign(arg_var, type_machine.add_middleware(sm.QuitMiddleware()))
        return True, sm.DoAll(done_state_list + [assign_var, verify_arg])

class ResolveArgs(sm.StateMachine):
    def __init__(self, class_id_var, iris = IRIS, uniq = "", recursed = False):
        self.iris = IRIS
        self.recursed = recursed
        self.class_id_var = class_id_var
        super().__init__()
        self.accepts_input = False
        self.uniq = uniq

    def next_state_base(self, text):
        self.class_id = self.context["ASSIGNMENTS"][self.class_id_var.scope_name()]
        cmd_object = self.iris.class_functions[self.class_id]
        cmd_names = self.iris.class2cmd[self.class_id]
        done_state_list = []
        for cmd in cmd_names:
            succ, map_ = util.arg_match(self.context[self.uniq + "_" + "query"], cmd)
            if succ:
                convert_values = {k: cmd_object.argument_types[k].convert_type(v, doing_match=True) for k,v in map_.items()}
                if all([success[0] for k, success in convert_values.items()]):
                    for arg, _ in convert_values.items():
                        self.context["ASSIGNMENTS"][self.uniq + "_" + arg] = convert_values[arg][1]
                        self.context["ASSIGNMENT_NAMES"][self.uniq + "_" + arg] = map_[arg]
                        done_state_list.append(sm.Print(["I am using {} for {}.".format(map_[arg], arg)]))
                    if all([arg in map_ for arg in cmd_object.all_args]):
                        to_exe = Execute(cmd_object, self.uniq, self.context["ASSIGNMENTS"], self.context["ASSIGNMENT_NAMES"], self.context[self.uniq + "_" + "query"], self.recursed)
                        return True, sm.DoAll(done_state_list + [to_exe]).when_done(self.get_when_done_state())
                    for arg in cmd_object.all_args:
                        cmd_object.argument_types[arg].clear_error()
        if all([(self.uniq + "_" + arg) in self.context["ASSIGNMENTS"] for arg in cmd_object.all_args]):
            to_exe = Execute(cmd_object, self.uniq, self.context["ASSIGNMENTS"], self.context["ASSIGNMENT_NAMES"], self.context[self.uniq + "_" + "query"], self.recursed)
            return True, to_exe.when_done(self.get_when_done_state())
        for arg in cmd_object.all_args:
            if (not (self.uniq + "_" + arg) in self.context["ASSIGNMENTS"]):
                iris_middle = IrisMiddleware(["Sure, we can run another function to generate {}.".format(arg),
                                              "What would you like to run?"])
                type_machine =  cmd_object.argument_types[arg].set_arg_name(arg).add_middleware(iris_middle).reset()
                arg_var = sm.Variable(arg, scope=self.uniq)
                verify_arg = sm.PrintVar(arg_var, util.print_assignment)
                assign_var = sm.Assign(arg_var, type_machine.add_middleware(sm.QuitMiddleware()))
                return True, sm.DoAll(done_state_list + [assign_var, verify_arg]).when_done(self)

def strip_key(key, dictionary):
    return {"_".join(k.split("_")[1:]):v for k,v in dictionary.items() if key == k.split("_")[0]}

class Execute(sm.AssignableMachine):
    def __init__(self, cmd_object, uniq, assignments, arg_names, query, recursed, iris = IRIS):
        self.iris = IRIS
        super().__init__()
        self.accepts_input = False
        orig_assignments = strip_key(uniq, assignments)
        orig_names = strip_key(uniq, arg_names)
        self.raw_output = None
        self.uniq = uniq
        try:
            self.raw_output = cmd_object.wrap_command(*[orig_assignments[x] for x in cmd_object.all_args])
            self.output = cmd_object.state_machine_output(orig_assignments, orig_names, self.raw_output, query)
            print("RECURSED", recursed)
            if recursed:
                self.output = []
        except:
            self.output = ["Sorry, something went wrong with the underlying command.", str(sys.exc_info()[1])]

    def next_state_base(self, text):
        self.assign(self.raw_output, name="COMMAND VALUE")
        if isinstance(self.raw_output, sm.StateMachine):
            return True, self.raw_output
        return False, sm.Value(None, self.context)

class EventLoop:
    def __init__(self, iris = IRIS):
        self.machine = sm.StateMachineRunner(IrisMachine())
        self.iris = iris
    def end(self, outputs):
        top_level_scope = self.machine.original_state.uniq
        class_id = self.machine.current_state.context["ASSIGNMENTS"][top_level_scope+"_"+"user_class"]
        cmd_object = self.iris.class_functions[class_id]
        label = cmd_object.title.upper()
        arg_map = {arg: strip_key(top_level_scope, self.machine.current_state.context["ASSIGNMENT_NAMES"])[arg] for arg in cmd_object.all_args}
        self.machine.reset()
        return {"state":"START", "text": outputs, "label":label, "arg_map": arg_map, "history": self.iris.history["history"] }
    def state_machine(self, data):
        outputs = []
        text = util.get_last_message(data["messages"])
        self.machine.next_state(text)
        for o in self.machine.current_output():
            outputs.append(o)
        keep_going, more_outputs = self.machine.run_until_input_required()
        if not keep_going:
            return self.end(outputs + more_outputs)
        return {"state": "RECURSE", "text": outputs + more_outputs}
