from . import state_machine as sm
from . import state_types as st
from .core import IRIS
from . import util
import sys

class IrisMachine(sm.StateMachine):
    def __init__(self, iris = IRIS):
        self.iris = IRIS
        super().__init__()
        self.output = ["Okay, what would you like to do?"]

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
        # need to reset..., possibly better way
        self.context["ASSIGNMENTS"] = {}
        self.context["ASSIGNMENT_NAMES"] = {}
        # get initial prediction
        class_id, class_text, pred = self.iris.get_predictions(text)[0]
        cmd_object = self.iris.class_functions[class_id]
        explain_cmd = sm.ExplainMiddleware(lambda caller: sm.Print(cmd_object.help_text).when_done(caller))
        command_title = class_text[0]
        self.context["query"] = text
        # create a variable to hold the predicted command class
        user_class = sm.Variable("user_class")
        # this machine will resolve the args on that command
        resolve_args = ResolveArgs(user_class)
        # four alternative commands to present to user, if we don't want the one selected
        select_alternative = self.compose_options(text, 4)
        # ask whether we want these options
        options = st.YesNo("Would you like more options?",
                    # if we want something else, reassign that to the selected class
                    yes=sm.Assign(user_class, select_alternative).when_done(resolve_args),
                    no=sm.Jump("START"))
        # confirm whether we want to continue
        confirm = st.YesNo("Would you like to {}?".format(command_title),
                    yes=resolve_args,
                    no=options).add_middleware([explain_cmd, sm.QuitMiddleware()])
        # bind user_class to class_id, then run
        return True, sm.Let(user_class, equal=class_id, then_do=confirm)

class ResolveArgs(sm.StateMachine):
    def __init__(self, class_id_var, iris = IRIS):
        self.iris = IRIS
        self.class_id_var = class_id_var
        super().__init__()
        self.accepts_input = False

    def next_state_base(self, text):
        print("RESOLVE", self.context)
        self.class_id = self.context["ASSIGNMENTS"][self.class_id_var.name]
        cmd_object = self.iris.class_functions[self.class_id]
        cmd_names = self.iris.class2cmd[self.class_id]
        for cmd in cmd_names:
            succ, map_ = util.arg_match(self.context["query"], cmd)
            if succ:
                convert_values = {k: cmd_object.argument_types[k].convert_type(v) for k,v in map_.items()}
                if all([success[0] for k, success in convert_values.items()]):
                    for arg, _ in convert_values.items():
                        self.context["ASSIGNMENTS"][arg] = convert_values[arg][1]
                        self.context["ASSIGNMENT_NAMES"][arg] = map_[arg]
                    if all([arg in map_ for arg in cmd_object.all_args]):
                        return True, Execute(cmd_object, self.context["ASSIGNMENTS"], self.context["ASSIGNMENT_NAMES"], self.context["query"])#(self.context)
                else:
                    for arg in cmd_object.all_args:
                        cmd_object.argument_types[arg].clear_error()
        if all([arg in self.context["ASSIGNMENTS"] for arg in cmd_object.all_args]):
            return True, Execute(cmd_object, self.context["ASSIGNMENTS"], self.context["ASSIGNMENT_NAMES"], self.context["query"])
        for arg in cmd_object.all_args:
            if (not arg in self.context["ASSIGNMENTS"]):
                type_machine =  cmd_object.argument_types[arg].set_arg_name(arg)
                return True, sm.Assign(sm.Variable(arg), type_machine.add_middleware(sm.QuitMiddleware())).when_done(self)

class Execute(sm.StateMachine):
    def __init__(self, cmd_object, assignments, arg_names, query, iris = IRIS,):
        self.iris = IRIS
        super().__init__()
        self.accepts_input = False
        try:
            self.output = cmd_object.state_machine_output(assignments, arg_names, query)
        except:
            self.output = ["Sorry, something went wrong with the underlying command.", str(sys.exc_info()[1])]

    def next_state_base(self, text):
        return False, sm.Value(None, self.context)

class EventLoop:
    def __init__(self, iris = IRIS):
        self.machine = sm.StateMachineRunner(IrisMachine())
        self.iris = iris
    def end(self, outputs):
        class_id = self.machine.current_state.context["ASSIGNMENTS"]["user_class"]
        cmd_object = self.iris.class_functions[class_id]
        label = cmd_object.title.upper()
        arg_map = {arg: self.machine.current_state.context["ASSIGNMENT_NAMES"][arg] for arg in cmd_object.all_args}
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
