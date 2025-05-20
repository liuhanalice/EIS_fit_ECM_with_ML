import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import constants
import matplotlib.ticker as ticker
from matplotlib.ticker import EngFormatter
import scipy.io

from utils import *

if __name__ == "__main__":
    np.random.seed(0)
    np.random.seed(1)

    # # For Classification C1-C5
    # all_param=[]
    # all_param.append(Circuit0_param.tolist())
    # all_param.append(Circuit1_param.tolist())
    # all_param.append(Circuit2_param.tolist())
    # all_param.append(Circuit3_param.tolist())
    # all_param.append(Circuit4_param.tolist())
    # df = pd.DataFrame(all_param)
    # df.transpose()
    # k=df.stack()
    # k.to_csv('paramc1-c5_test.csv', index=False)

    # data_ver="v2"
    # if size_number >= 1000: data_num_n = str("%.0f%s" % (size_number/1000.0, 'k'))
    # else : data_num_n = str(size_number)

    # # File_name="xy_data_"+data_num_n+"_"+str(number_of_circuit)+"circuit_"+data_ver+".mat"
    # File_name="xy_data_"+data_num_n+"_"+str(number_of_circuit)+"circuit_"+data_ver+"_test_set.mat"


    # x_data,y_data = export_data(Circuit_spec,size_number,number_of_point,numc=number_of_circuit)
    # mdic={"x_data":x_data,"y_data":y_data}
    # scipy.io.savemat(File_name, mdic)



    

    # # For Regression Ci
    # data_ver = "v2"
    # circuit_param = [Circuit0_param, Circuit1_param, Circuit2_param, Circuit3_param, Circuit4_param]
    # for i in range(5):
    #     all_param = []
    #     all_param.append(circuit_param[i].tolist())
    #     df = pd.DataFrame(all_param)
    #     df.transpose()
    #     k = df.stack()
    #     k.to_csv('paramc' + str(i+1) + '.csv', index=False)

    #     if size_number >= 1000: data_num_n = str("%.0f%s" % (size_number/1000.0, 'k'))
    #     else : data_num_n = str(size_number)

    #     File_name="xy_data_"+data_num_n+"_regC"+str(i+1)+"_"+data_ver+".mat"

    #     x_data,_ = arrange_data(Circuit_spec[i],(i),size_number, number_of_point)
    #     y_data = circuit_param[i]
    #     print(y_data.shape)
    #     mdic={"x_data":x_data,"y_data":y_data}
    #     scipy.io.savemat(File_name, mdic)
        
    #     # Split data into train and test sets
    #     test_size_number = int(0.2 * size_number)
    #     if test_size_number >= 1000: test_data_num_n = str("%.0f%s" % (test_size_number/1000.0, 'k'))
    #     else : test_data_num_n = str(test_size_number)

    #     test_file_name = "xy_data_"+test_data_num_n+"_regC"+str(i+1)+"_"+data_ver+"_test.mat"
    #     x_test_data = x_data[-test_size_number:]
    #     y_test_data = y_data[-test_size_number:]
    #     mdic={"x_data":x_test_data,"y_data":y_test_data}
    #     scipy.io.savemat(test_file_name, mdic)


    # # For Regression C6
    # data_ver = "v2"
    # # circuit_param = [Circuit0_param, Circuit1_param, Circuit2_param, Circuit3_param, Circuit4_param]
    
    # all_param = []
    # all_param.append(Circuit5_param.tolist())
    # df = pd.DataFrame(all_param)
    # df.transpose()
    # k = df.stack()
    # k.to_csv('paramc6.csv', index=False)

    # if size_number >= 1000: data_num_n = str("%.0f%s" % (size_number/1000.0, 'k'))
    # else : data_num_n = str(size_number)

    # File_name="xy_data_"+data_num_n+"_regC6_"+data_ver+".mat"

    # x_data,_ = arrange_data(Circuit_spec[5],5,size_number, number_of_point)
    # y_data = Circuit5_param
    # print(y_data.shape)
    # mdic={"x_data":x_data,"y_data":y_data}
    # scipy.io.savemat(File_name, mdic)
    
    # # Split data into train and test sets
    # test_size_number = int(0.2 * size_number)
    # if test_size_number >= 1000: test_data_num_n = str("%.0f%s" % (test_size_number/1000.0, 'k'))
    # else : test_data_num_n = str(test_size_number)

    # test_file_name = "xy_data_"+test_data_num_n+"_regC6_"+data_ver+"_test.mat"
    # x_test_data = x_data[-test_size_number:]
    # y_test_data = y_data[-test_size_number:]
    # mdic={"x_data":x_test_data,"y_data":y_test_data}
    # scipy.io.savemat(test_file_name, mdic)

    # For Regression C7
    data_ver = "v2"
    
    all_param = []
    all_param.append(Circuit5_param.tolist())
    df = pd.DataFrame(all_param)
    df.transpose()
    k = df.stack()
    k.to_csv('paramc7.csv', index=False)

    if size_number >= 1000: data_num_n = str("%.0f%s" % (size_number/1000.0, 'k'))
    else : data_num_n = str(size_number)

    File_name="xy_data_"+data_num_n+"_regC7_"+data_ver+".mat"

    x_data,_ = arrange_data(Circuit_spec[6],6,size_number, number_of_point)
    y_data = Circuit6_param
    print(y_data.shape)
    mdic={"x_data":x_data,"y_data":y_data}
    scipy.io.savemat(File_name, mdic)
    
    # Split data into train and test sets
    test_size_number = int(0.2 * size_number)
    if test_size_number >= 1000: test_data_num_n = str("%.0f%s" % (test_size_number/1000.0, 'k'))
    else : test_data_num_n = str(test_size_number)

    test_file_name = "xy_data_"+test_data_num_n+"_regC7_"+data_ver+"_test.mat"
    x_test_data = x_data[-test_size_number:]
    y_test_data = y_data[-test_size_number:]
    mdic={"x_data":x_test_data,"y_data":y_test_data}
    scipy.io.savemat(test_file_name, mdic)


        
    
