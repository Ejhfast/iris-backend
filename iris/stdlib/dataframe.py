from .. import IRIS, IrisCommand
from .. import state_types as t
from .. import state_machine as sm
from .. import util as util

class LoadCSVData(IrisCommand):
    title = "load csv data from {file}"
    examples = ["load csv {file}"]
    argument_types = {
        "file": t.File("What file would you like to load?"),
    }
    def command(self, file):
        from .fileupload import file_state
        # file_state is a custom state machine, built using Iris low-level library
        # allows for injection of user input into code logic
        return file_state(file)
    def explanation(self, result):
        return []

loadCSVData = LoadCSVData()

class ListDataframeNames(IrisCommand):
    title = "list columns from {dataframe}"
    examples = [ "list columns {dataframe}" ]
    help_text = [
        "This command lists the column names for a Dataframe object."
    ]
    def command(self, dataframe : t.EnvVar("What dataframe?")):
        return dataframe.column_names
    def explanation(self, results):
        return [
            "The column names are:",
            {"type":"data", "value":util.prettify_data(results)}
        ]

listDataframeNames = ListDataframeNames()

class GetDFColumn(IrisCommand):
    title = "get {column} from {dataframe}"
    examples = [ "get {column} {dataframe}" ]
    argument_types = {
        "dataframe": t.Dataframe("What dataframe to extract the column from?"),
        "column": t.String("What is the name of the column?")
    }
    help_text = [
        "This command pull a column from a dataframe into the main environment."
    ]
    def command(self, dataframe, column):
        return dataframe.get_column(column)

getDFColumn = GetDFColumn()

class SelectorTest(IrisCommand):
    title = "selector test"
    examples = []
    def command(self, selector : t.DataframeSelector("Give me dataframe")):
        return selector

selectorTest = SelectorTest()
