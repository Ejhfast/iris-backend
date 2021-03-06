import numpy as np
import io
import base64
from . import util
import copy
from collections import defaultdict

class IrisValue:
    def __init__(self, value, name=None):
        self.value = value
        self.name = name

class IrisValues(IrisValue):
    def __init__(self, values, names):
        self.values = values
        self.names = names

class IrisId(IrisValue):
    def __init__(self, value, id, name=None):
        self.value = value
        self.id = id
        if not name:
            self.name = value
        else:
            self.name = name

class IrisImage(IrisId):
    type="Image"
    def __init__(self, plt, name):
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        self.value = base64.b64encode(buf.read()).decode('utf-8')
        self.name = name

class IrisModel(IrisValue):
    type="Model"
    def __init__(self, model, X, y, name=None):
        self.dataframe_X = X
        self.dataframe_y = y
        self.X = X.to_matrix()
        self.y = y.to_matrix()
        print("X shape", self.X.shape)
        print("y shape", self.y.shape)
        self.y = self.y.reshape(self.y.shape[0])
        self.model = model
        self.name = name
    def fit(self):
        self.model.fit(self.X, self.y)

class IrisData(IrisValue):
    type="Data"
    def __init__(self, xvals, yvals):
        self.X = xvals
        self.y = yvals

class IrisFile(IrisValue):
    type="File"
    def __init__(self, name, content):
        self.name = name
        self.content = content

class IrisDataframe(IrisValue):
    type="DataFrame"
    def __init__(self, name=None, column_names=[], column_types=[], data=[], do_conversion=True):
        self.name = name
        self.column_names = column_names
        self.column_types = column_types
        if do_conversion:
            self.data = self.convert_data(data)
        else:
            self.data = data

    def get_column(self, name):
        indexes = {name:i for i, name in enumerate(self.column_names)}
        return np.array([row[indexes[name]] for row in self.data])

    def to_matrix(self):
        return np.array(self.data)#.T #np.array([self.get_column(name) for name in self.column_names]).T

    def copy_frame(self, columns, conditions=[]):
        new_frame = copy.copy(self)
        new_frame.column_names = list(columns)
        new_frame.data = []
        new_frame.cat2index = {}
        indexes = {name:i for i, name in enumerate(self.column_names)}
        for i in range(0, len(self.data)):
            if all([f(self.data[i]) for f in conditions]):
                new_frame.data.append(list([self.data[i][indexes[c]] for c in new_frame.column_names]))
        for i,name in enumerate(new_frame.column_names):
            new_frame.cat2index[i] = dict(self.cat2index[indexes[name]])
            new_frame.column_types[i] = str(self.column_types[indexes[name]])
        return new_frame

    def change_type(self, name, type_):
        cat2index = {}
        def convert_type(value, type_):
            if type_ == "String":
                return str(value)
            elif type_ == "Number":
                return float(value)
            elif type_ == "Categorical":
                if not value in cat2index:
                    cat2index[value] = len(cat2index)
                return cat2index[value]
        indexes = {name:i for i, name in enumerate(self.column_names)}
        for row in self.data:
            row[indexes[name]] = convert_type(row[indexes[name]], type_)
        self.column_types[indexes[name]] = type_
        self.cat2index[indexes[name]] = cat2index
        return self

    def remove_column(self, name):
        new_frame = copy.copy(self)
        new_frame.column_names = list(self.column_names)
        new_frame.data = []
        new_frame.cat2index = {}
        indexes = {name:i for i, name in enumerate(self.column_names)}
        for i in range(0, len(self.data)):
            new_frame.data.append(list(self.data[i]))
            del new_frame.data[i][indexes[name]]
        new_frame.column_names.remove(name)
        for i,name in enumerate(new_frame.column_names):
            new_frame.cat2index[i] = dict(self.cat2index[indexes[name]])
            new_frame.column_types[i] = str(self.column_types[indexes[name]])
        return new_frame

    def select_data(self, column, operation):
        indexes = {name:i for i, name in enumerate(self.column_names)}
        col_i = indexes[column]
        def selector(row):
            if(operation(row[col_i])):
                return True
            return False
        return self.copy_frame(self.column_names, conditions=[selector])

    # TODO: add error handling
    def convert_data(self, data):
        new_data = []
        cat2index = defaultdict(dict)
        for j,line in enumerate(data):
            if line == '': continue # empty last line after new line
            old_row = util.split_line(line)
            new_row = []
            for i,value in enumerate(old_row):
                if self.column_types[i] == "Number":
                    new_row.append(float(value))
                elif self.column_types[i] == "Categorical":
                    if not value in cat2index[i]:
                        cat2index[i][value] = len(cat2index[i])
                    new_row.append(cat2index[i][value])
                else:
                    new_row.append(value)
            new_data.append(new_row)
        self.cat2index = cat2index
        return new_data

class FunctionWrapper:
    def __init__(self, function, name="anonymous func"):
        self.function = function
        self.name = name

class EnvReference:
    def __init__(self, name):
        self.name = name
    def get_value(self, iris):
        return iris.env[self.name]
