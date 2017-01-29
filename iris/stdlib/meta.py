from .. import IRIS, IrisCommand
from .. import state_types as t
from .. import state_machine as sm
from .. import util as util

class SaveEnv(IrisCommand):
    title = "save environment to {name}"
    examples = [ "save environment {name}",
                 "save env to {name}" ]
    help_text = [
        "This command saves the current environment (all data in the left pane).",
        "This data can be loaded later using the 'load environment' command."
    ]
    def command(self, name : t.String(question="What filename to save under?")):
        import dill as pickle
        with open(name, 'wb') as f:
            print(self.iris.env)
            pickle.dump(self.iris.serialize_state(), f)
            return "Saved to {}.".format(name)

saveEnv = SaveEnv()

class LoadEnv(IrisCommand):
    title = "load environment from {name}"
    examples = [ "load environment {name}",
                 "load env from {name}" ]
    help_text = [
        "This command loads an enviornment previously saved by Iris."
    ]
    def command(self, name : t.String(question="What filename to load?")):
        import dill as pickle
        with open(name, 'rb') as f:
            data = pickle.load(f)
            self.iris.load_state(data)
            return "Loaded environment from \"{}\".".format(name)

loadEnv = LoadEnv()

class StoreCommand(IrisCommand):
    title = "save value of last command"
    examples = [ "save last as {name0}" ]
    store_result = t.VarName(question="Where would you like to save this result?")
    def command(self, cmd_val : t.Memory()):
        return cmd_val
    def explanation(self, result):
        return []

storeCommand = StoreCommand()