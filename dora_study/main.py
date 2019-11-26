import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.feature_extraction import DictVectorizer
from functools import wraps
import inspect
from functools import wraps

def saveLast(func):
  @wraps(func)
  def with_logging(self,*args, **kwargs):
      self._last=self._data.copy()
      self._lastlogs=self._logs.copy()

      self._lastlast=self._last.copy()
      self._lastlastlogs=self._lastlogs.copy()

      force=kwargs.pop("force",None)

      rep=func(self,*args, **kwargs)

      argss= inspect.getcallargs(func,self, *args, **kwargs)
      del argss["self"]
      argss=["{}={}".format(i,"\""+j+"\"" if isinstance(j,str) else j) for i,j in argss.items()]
      self._log( "self.{}({})".format( func.__name__, ", ".join(argss) ) ,force=force)

      return rep
  return with_logging
  
def addCustomFunc2(self,func):
  @wraps(func)
  def with_logging(*args, **kwargs):

      return saveLast(func)(*args,**kwargs)
      # self._last=self._data.copy()
      # self._lastlogs=self._logs.copy()
      
      # self._lastlast=self._last.copy()
      # self._lastlastlogs=self._lastlogs.copy()

      # rep=func(self,*args, **kwargs)
      # argss= inspect.getcallargs(func,self, *args, **kwargs)
      # del argss["self"]
      # argss=["{}={}".format(i,"\""+j+"\"" if isinstance(j,str) else j) for i,j in argss.items()]
      # self._log( "self.{}({})".format( func.__name__, ", ".join(argss) ) )
      # return rep
  return with_logging

class Dora:
  _CUSTOMS={}
  
    
  def __init__(self, data = None, output = None):
    self.init(data = data, output = output)

  def init(self,data,output):
    self._snapshots = {}
    self._logs = []
    self._last=None
    self._lastlogs=None
    self._lastlast=None
    self._lastlastlogs=None

    self.configure(data = data, output = output)

  def configure(self, data = None, output = None):
    if (type(output) is str or type(output) is int):
      self._output = output
    if (type(data) is str):
      self._initial_data = pd.read_csv(data)
      self._data = self._initial_data.copy()
      self._logs = []
    if (type(data) is pd.DataFrame):
      self._initial_data = data
      self._data = self._initial_data.copy()
      self._logs = []


#____________PLOT______________

  def plot_feature(self, feature_name):
    x = self._data[feature_name]
    y = self._data[self._output]
    fit = np.polyfit(x, y, deg = 1)
    fig, ax = plt.subplots()
    ax.plot(x, fit[1] + fit[0] * x)
    ax.scatter(x, y)
    ax.set_title("{0} vs. {1}".format(feature_name, self._output))
    fig.show()

  def explore(self):
    features = self._input_columns()
    row_count = math.floor(math.sqrt(len(features)))
    col_count = math.ceil(len(features) / row_count)
    figure = plt.figure(1)

    for index, feature in enumerate(features):
      figure.add_subplot(row_count, col_count, index + 1)
      x = self._data[feature]
      y = self._data[self.output]
      fit = np.polyfit(x, y, deg = 1)
      plt.plot(x, fit[0] * x + fit[1])
      plt.scatter(x, y)
      plt.title("{0} vs. {1}".format(feature, self._output))
    plt.show()


#____________USEFUL______________
  def input_columns(self):
    column_names = list(self._data.columns)
    column_names.remove(self._output)
    return column_names

  def set_training_and_validation(self):
    training_rows = np.random.rand(len(self._data)) < 0.8
    self.training_data = self._data[training_rows]
    self.validation_data = self._data[~training_rows]


  def _log(self, string,force=False):
    if string in self._logs and not force:
      raise Exception("""
            _log: {string} already in logs, if you want to force, add force=True""")
    self._logs.append(string)


#_______________SNAP_____________________
  def snapshot(self, name):
    snapshot = {
      "data": self._data.copy(),
      "logs": self._logs.copy()
    }
    self._snapshots[name] = snapshot

  def use_snapshot(self, name):
    self._data = self._snapshots[name]["data"]
    self._logs = self._snapshots[name]["logs"]

#________________back_____________________
  def back_initial_data(self):
    self.init(self._initial_data,self._output)

  def back_one(self):
    self._data=self._last.copy()
    self._logs=self._lastlogs.copy()
    self._last=self._lastlast.copy()
    self._lastlogs=self._lastlastlogs.copy()


#____________________PREP_____________________
  @saveLast
  def remove_feature(self, feature_name):
    del self.data[feature_name]
    # self._log("self.remove_feature('{0}')".format(feature_name))

  @saveLast
  def extract_feature(self, old_feat, new_feat, mapper):
    new_feature_column = map(mapper, self.data_[old_feat])
    self._data[new_feat] = list(new_feature_column)
    # self._log("self.extract_feature({0}, {1}, {2})".format(old_feat, new_feat, mapper))

  @saveLast
  def impute_missing_values(self):
    column_names = self._input_columns()
    imp = preprocessing.Imputer()
    imp.fit(self._data[column_names])
    self._data[column_names] = imp.transform(self._data[column_names])
    # self._log("self.impute_missing_values()")

  @saveLast
  def scale_input_values(self):
    column_names = self.i_nput_columns()
    self._data[column_names] = preprocessing.scale(self._data[column_names])
    # self._log("self.scale_input_values()")

  @saveLast
  def extract_ordinal_feature(self, feature_name):
    feature = self._data[feature_name]
    feature_dictionaries = map(
      lambda x: { str(feature_name): str(x) },
      feature
    )
    vec = DictVectorizer()
    one_hot_matrix = vec.fit_transform(feature_dictionaries).toarray()
    one_hot_matrix = pd.DataFrame(one_hot_matrix)
    one_hot_matrix.columns = vec.get_feature_names()
    self._data = pd.concat(
      [
        self._data,
        one_hot_matrix
      ],
      axis = 1
    )
    del self._data[feature_name]
    # self._log("self.extract_ordinal_feature('{0}')".format(feature_name))


# _______CLASSMETHOD__________
  @classmethod
  def addCustomFunction(cls,func,fn=None):
    fn = func.__name__ if fn is None else fn
    cls._CUSTOMS[fn]=saveLast(func)


#______JUPYTER NOTEBOOK__SPECIAL_FUNC___
  def _ipython_display_(self, **kwargs):
    from IPython.display import HTML,display
    display(HTML(self._data._repr_html_()))


#_______CLASS_______________________________
  def __dir__(self):
    return list(self._CUSTOMS.keys())+[i for i in super().__dir__() if not i.startswith("_")]+[i for i in super().__dir__() if  i.startswith("_")]

  def __getattr__(self,g):
    if g in self._CUSTOMS:
      return addCustomFunc2(self,self._CUSTOMS[g])
    return object.__getattribute__(self,g)
  
