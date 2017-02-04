import numpy as np
import io
import base64

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
        self.X = X
        self.y = y
        self.model = model
        self.name = name

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
    def __init__(self, name, column_names, column_types, data):
        self.name = name
        self.column_names = column_names
        self.column_types = column_types
        self.data = self.convert_data(data)

    def get_column(self, name):
        indexes = {name:i for i, name in enumerate(self.column_names)}
        return np.array([row[indexes[name]] for row in self.data])

    def convert_data(self, data):
        new_data = []
        cat2index = {}
        for j,line in enumerate(data):
            old_row = line.replace("\"","").split(",")
            new_row = []
            for i,value in enumerate(old_row):
                if self.column_types[i] == "Number":
                    new_row.append(float(value))
                elif self.column_types[i] == "Categorical":
                    if not value in cat2index:
                        cat2index[value] = len(cat2index)
                    new_row.append(cat2index[value])
                else:
                    new_row.append(value)
            new_data.append(new_row)
        return new_data

class FunctionWrapper:
    def __init__(self, function, name="anonymous func"):
        self.function = function
        self.name = name

class EnvReference:
    def __init__(self, name):
        self.name = name
