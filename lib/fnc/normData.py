#######################################################################################################################
#######################################################################################################################
# Title: Baseline NILM Architecture
# Topic: Non-intrusive load monitoring utilising machine learning, pattern matching and source separation
# File: normData
# Date: 23.10.2021
# Author: Dr. Pascal A. Schirmer
# Version: V.0.0
# Copyright: Pascal A. Schirmer
#######################################################################################################################
#######################################################################################################################


#######################################################################################################################
# Import external libs
#######################################################################################################################


#######################################################################################################################
# Norm
#######################################################################################################################
def normData(X, Y, setup_Data):
    ####################################################################################################################
    # init
    ####################################################################################################################
    if setup_Data['normData'] == 1:
        maxX = setup_Data['meanX']
        maxY = setup_Data['meanY']
    elif setup_Data['normData'] == 2:
        maxXY = setup_Data['meanX']
    elif setup_Data['normData'] == 3:
        stdX = setup_Data['stdX']
        stdY = setup_Data['stdY']
        meanX = setup_Data['meanX']
        meanY = setup_Data['meanY']
    else:
        stdX = 1
        stdY = 1
        meanX = 0
        meanY = 0

    ####################################################################################################################
    # Norm
    ####################################################################################################################
    if setup_Data['normData'] == 1:
        X = X/maxX
        Y = Y/maxY
    elif setup_Data['normData'] == 2:
        X = X/maxXY
        Y = Y/maxXY
    elif setup_Data['normData'] == 3:
        X = (X - meanX)/stdX
        Y = (Y - meanY) / stdY

    return [X, Y]


#######################################################################################################################
# Inv Norm
#######################################################################################################################
def invNormData(X, Y, setup_Data):
    ####################################################################################################################
    # init
    ####################################################################################################################
    if setup_Data['normData'] == 1:
        maxX = setup_Data['meanX']
        maxY = setup_Data['meanY']
    elif setup_Data['normData'] == 2:
        maxXY = setup_Data['meanX']
    elif setup_Data['normData'] == 3:
        stdX = setup_Data['stdX']
        stdY = setup_Data['stdY']
        meanX = setup_Data['meanX']
        meanY = setup_Data['meanY']
    else:
        stdX = 1
        stdY = 1
        meanX = 0
        meanY = 0

    ####################################################################################################################
    # Norm
    ####################################################################################################################
    if setup_Data['normData'] == 1:
        X = X*maxX
        Y = Y*maxY
    if setup_Data['normData'] == 2:
        X = X*maxXY
        Y = Y*maxXY
    elif setup_Data['normData'] == 3:
        X = X*stdX + meanX
        Y = Y*stdY + meanY

    return [X, Y]
