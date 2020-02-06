import h5py
import pandas as pd
import numpy as np

def reshape_database(database, design_vars, obj_vars):
  """This function will take in an H5 database, convert it to a pandas database, and return a list of coordiantes and objectives"""
  db = h5py.File(database, 'r+')
  # Create a dict of reactor name : reactor attributes to fill the data frame
  reactorDict = {}
  for r in db.keys():
    reactor = db[r]
    dataDict = {}
    for k, v in reactor.items():
      if k[0] != 'F':
        dataDict[k] = list(v)
    reactorDict[r] = dataDict

  reactorData = pd.DataFrame(reactorDict)
  reactorData = reactorData.T
  pu_content_dict = {}
  for num, enr in reactorData['enrichment'].items():
    if enr[0] == '15Pu12U10Zr':
      pu_content_dict[num] = 0.55555555
    elif enr[0] == '27U10Zr':
      pu_content_dict[num] = 0.0
    elif enr[0] == '27Pu0U10Zr':
      pu_content_dict[num] = 1.0
    elif enr[0] == '7Pu20U10Zr':
      pu_content_dict[num] = 0.25
    elif enr[0] == '20Pu7U10Zr':
      pu_content_dict[num] = 0.75
    else:
      print(num, enr)

  reactorData['pu_content'] = pu_content_dict.values()
  reactorData['pu_content'] = reactorData['pu_content'].apply(lambda x : [x])

  # Create a a coordinate system via the variables for interpolator
  var_array = []
  for var in design_vars:
    var_list = []
    for data_point in reactorData[var]:
      var_list.append(data_point[0])
    var_array.append(var_list)
  coordinates = list(zip(*var_array))

  # Create a list of objectives to solve for, and a list of known values
  # for the objectives
  obj_array = []
  for obj in obj_vars:
    obj_list = []
    for data_point in reactorData[obj]:
      try:
        obj_list.append(data_point[0])
      except IndexError:
        obj_list.append(data_point)
    obj_array.append(obj_list)
  objectives = list(zip(*obj_array))

  return coordinates, objectives

def create_dataframe(database):
  
  db = h5py.File(database, 'r+')     
  reactorDict = {}
  for r in db.keys():
    reactor = db[r]
    dataDict = {}
    for k, v in reactor.items():
      if k[0] != 'F':
        dataDict[k] = list(v)
    reactorDict[r] = dataDict

  reactorData = pd.DataFrame(reactorDict)
  reactorData = reactorData.T
  pu_content_dict = {}
  for num, enr in reactorData['enrichment'].items():
    if enr[0] == '15Pu12U10Zr':
      pu_content_dict[num] = 0.55555555
    elif enr[0] == '27U10Zr':
      pu_content_dict[num] = 0.0
    elif enr[0] == '27Pu0U10Zr':
      pu_content_dict[num] = 1.0
    elif enr[0] == '7Pu20U10Zr':
      pu_content_dict[num] = 0.25
    elif enr[0] == '20Pu7U10Zr':
      pu_content_dict[num] = 0.75
    else:
      print(num, enr)

  reactorData['pu_content'] = pu_content_dict.values()
  reactorData['pu_content'] = reactorData['pu_content'].apply(lambda x : [x])
  return reactorData
