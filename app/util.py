from collections import defaultdict
import numpy as np
from sklearn.linear_model import LogisticRegression

# crappy data type inference function
def detect_data_type(data):
    try:
        return "Number", float(data)
    except:
        if len(data.split()) > 1:
            return "Text", data
        else:
            return "Categorical", data

# crappy function to figure out csv type and header
# TODO: deal with no header
# TODO: deal with quoted strings
def rows_and_types(data):
    title = data[0]
    cols = title.split(",")
    sample_line = data[1].split(",")
    return list(zip(cols,[detect_data_type(x) for x in sample_line]))

def process_data(meta, content):
    env = defaultdict(list)
    cat2index = {}
    for j,line in enumerate(content):
        if j == 0: continue
        cols = line.split(",")
        for i,col in enumerate(cols):
            val = None
            if meta[i]["type"] == "number":
                val = float(col)
            elif meta[i]["type"] == "categorical":
                if col in cat2index:
                    val = cat2index[col]
                else:
                    val = len(cat2index)
                    cat2index[col] = val
            else:
                val = col
            env[meta[i]["name"]].append(val)
    return env

def make_xy(meta, env):
    X, y = None, None
    feature_index = {"X":{}, "y":{}}
    def init_or_append(X,data):
        npx = np.array(data)
        npx = npx.reshape(npx.shape[0],1)
        if X != None: return np.concatenate((X,npx),axis=1)
        return npx
    for k,vs in meta.items():
        if "x_value" in vs:
            X = init_or_append(X, env[vs["name"]])
            feature_index["X"][X.shape[1]-1] = vs["name"]
        if "y_value" in vs:
            y = init_or_append(y, env[vs["name"]])
            feature_index["y"][y.shape[1]-1] = vs["name"]
    if y.shape[1] == 1:
        y = y.reshape(y.shape[0])
    from sklearn.cross_validation import cross_val_score
    print(cross_val_score(LogisticRegression(), X, y, scoring="f1_macro", cv=5))
    return X,y,feature_index
