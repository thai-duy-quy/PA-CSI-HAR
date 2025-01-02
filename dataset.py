import numpy as np
import glob
import scipy.io as sio
import torch
import pandas as pd
import csv
import math 
import os
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import normalize

class CSI_Dataset(Dataset):
    """CSI dataset."""

    def __init__(self, root_dir, modal='CSIamp', transform=None, few_shot=False, k=5, single_trace=True):
        """
        Args:
            root_dir (string): Directory with all the images.
            modal (CSIamp/CSIphase): CSI data modal
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.modal=modal
        self.transform = transform
        self.data_list = glob.glob(root_dir+'/*/*.mat')
        self.folder = glob.glob(root_dir+'/*/')
        self.category = {self.folder[i].split('/')[-2]:i for i in range(len(self.folder))}

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        sample_dir = self.data_list[idx]
        y = self.category[sample_dir.split('/')[-2]]
        x = sio.loadmat(sample_dir)[self.modal]
        
        # normalize
        x = (x - 42.3199)/4.9802
        
        # sampling: 2000 -> 500
        x = x[:,::4]
        x = x.reshape(3, 114, 500)
        
        if self.transform:
            x = self.transform(x)
        
        x = torch.FloatTensor(x)

        return x,y


class Widar_Dataset(Dataset):
    def __init__(self,root_dir):
        self.root_dir = root_dir
        self.data_list = glob.glob(root_dir+'/*/*.csv')
        self.folder = glob.glob(root_dir+'/*/')
        self.category = {self.folder[i].split('/')[-2]:i for i in range(len(self.folder))}
        
    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        sample_dir = self.data_list[idx]
        y = self.category[sample_dir.split('/')[-2]]
        x = np.genfromtxt(sample_dir, delimiter=',')
        
        # normalize
        x = (x - 0.0025)/0.0119
        
        # reshape: 22,400 -> 22,20,20
        x = x.reshape(22,20,20)
        # interpolate from 20x20 to 32x32
        # x = self.reshape(x)
        x = torch.FloatTensor(x)

        return x,y

def UT_HAR_dataset(root_dir):
    data_list = glob.glob(root_dir+"\\UT_HAR\\data\\*.csv")
    label_list = glob.glob(root_dir+"\\UT_HAR\\label\\*.csv")
    WiFi_data = {}
    for data_dir in data_list:
        data_name = data_dir.split("\\")[-1].split('.')[0]
        with open(data_dir, 'rb') as f:
            data = np.load(f)
            #print(data.shape)
            #exit()
            data = data.reshape(len(data),1,250,90)
            data_norm = (data - np.min(data)) / (np.max(data) - np.min(data))
        #print(data_name)
        #exit()
        #break
        WiFi_data[data_name] = torch.Tensor(data_norm)
    for label_dir in label_list:
        label_name = label_dir.split("\\")[-1].split('.')[0]
        with open(label_dir, 'rb') as f:
            label = np.load(f)
            #print(label[0])
            #exit()
        WiFi_data[label_name] = torch.Tensor(label)
        #print(label_name)
        #exit()
        #break
    return WiFi_data

def magnitude_math(z):
    return math.sqrt(pow(z.real, 2) + pow(z.imag, 2))

def get_complex_number(complex_number):
    complex_number = complex_number.replace("+-","-")
    complex_number = complex_number.replace("i","j")
    complex_number = complex(complex_number)
    imag = np.imag(complex_number)
    real = np.real(complex_number)
    angle = np.angle(complex_number)
    deg = math.degrees(angle)
    return complex_number, imag, real, angle, deg, magnitude_math(complex_number)  

def make_subcarriers_title(Nrx,Ntx,NSub):
    sub_arr = []
    for i in range(Nrx):
        for j in range(Ntx):
            for k in range(NSub):
                str_ = "csi_" + str(i+1)+"_"+str(j+1) + "_" + str(k+1)
                sub_arr.append(str_)
    return sub_arr

def normalize_data(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def average_list(d_list):
    sum = [0.0 for _ in range(len(d_list[0]))]
    for j in range(len(d_list[0])):
        for i in range(len(d_list)):
            sum[j] += d_list[i][j]
        sum[j] /= len(d_list)
    return sum

def merge_timestamp(data, time_stamp, max_len):
    intervel = (time_stamp[len(time_stamp)-1] - time_stamp[0]) / max_len
    cur_range = time_stamp[0] + intervel
    temp_list = []
    new_data = []
    for i in range(len(time_stamp)):
        if time_stamp[i] > cur_range:
            if len(temp_list) != 0:
                new_data.append(average_list(temp_list))
            else:
                new_data.append(data[i])
            temp_list = []
            cur_range = cur_range + intervel
            #print(cur_range)
        temp_list.append(data[i])
    if len(temp_list) != 0:
        new_data.append(average_list(temp_list))
    #print(len(new_data))
    if len(new_data) < max_len:
        for i in range(max_len-len(new_data)):
            new_data.append(data[len(time_stamp)-(i+1)])
    #    print("!!!!")
    return new_data[:max_len]


def Baha_et_al(root_dir):
    '''
    E - Environment: {1, 2, 3}
    S - Subject: {1, 2, 3, …, 30}
    C - Experiment Class: {1, 2, 3, 4, 5} 
    A - Activity: {1, 2, 3, …, 12}
    T - Trial: {1, 2, 3, …, 20}
    '''
    count = 0
    sub_arr = make_subcarriers_title(1,3,30)
    data = []
    data_angle = []
    data_deg = []
    label = []
    subject_list = os.listdir(root_dir)
    file_path = root_dir + "\\"
    count = 0
    for subject in subject_list:
        # if subject != 'Subject 1':
        #    continue
        # print(subject)
        # if count > 2:
        #    continue
        file_path_1 = file_path + "\\" + subject

        data_list = os.listdir(file_path_1)
        
        count1 = 0
        for f in data_list:
            # if count1 > 3:
            #    continue
            #if f != 'E1_S03_C01_A01_T02.csv':
            #    continue
            file = file_path_1 +"\\" + f
            print(file)
            data_name = file.split('\\')[-1].split('_')[-2]
            csi_data = pd.read_csv(file)
            #print(csi_data.shape)
            #exit()
            arr = []
            arr_angle = []
            arr_deg = []
            time_stamp = []
            count = 0
            for i in range(len(csi_data)):
                arr_row = []
                arr_row_angle = []
                arr_row_deg = []
                # print(count)
                # print(csi_data['timestamp_low'][i])
                time_stamp.append(csi_data['timestamp_low'][i])
                for item in sub_arr:
                    csi = csi_data[item][i]
                    _,_,_,angle,deg,magnitude = get_complex_number(csi)
                    arr_row.append(magnitude)
                    arr_row_angle.append(angle)
                    arr_row_deg.append(deg)
                count += 1

                #arr_row = normalize(arr_row)
                arr.append(arr_row)
                arr_angle.append(arr_row_angle) 
                arr_deg.append(arr_row_deg)
            #exit() 
            #arr_1 = np.array(arr)
            #print(arr_1.shape)
            arr = np.array(arr)
            time_stamp = np.array(time_stamp)
            print(arr.shape)
            #print(time_stamp.shape)
            exit()
            arr = merge_timestamp(arr, time_stamp,750)
            # arr = np.array(arr)
            # print(arr.shape)
            # exit() 
            data.append(arr)
            
            # arr_angle = merge_timestamp(arr_angle, time_stamp,850)
            # data_angle.append(np.array(arr_angle))

            # arr_deg = merge_timestamp(arr_deg, time_stamp,850)
            # data_deg.append(np.array(arr_deg))

            #activity = data_name.replace("A","")
            if data_name == 'A01' or data_name == 'A04' or data_name == 'A03':
                class_name = 0
            elif data_name == 'A02' or data_name == 'A05':
                class_name = 1
            elif data_name == 'A06' or data_name == 'A08':
                class_name = 2
            elif data_name == 'A10' or data_name == 'A11':
                class_name = 3
            elif data_name == 'A07' or data_name == 'A09':
                class_name = 4
            elif data_name == 'A12':
                class_name = 5
            #label.append(int(activity)-1)
            label.append(class_name)
            count1 += 1
            # print(count1)
            # if count1 == 50:
            #     break
        count += 1

    data = np.array(data)
    data_angle = np.array(data_angle)
    data_deg = np.array(data_deg)
    label = np.array(label)
    print(data.shape)
    #print(data_deg.shape)
    print(label.shape)
    print(label)
    # exit()
    #Baha_data['data'] = data 
    
    #Baha_data['label'] = label
    if root_dir == 'data\Baha_et_al\Environment 1':
        np.save('data_6c_1_hang_750.npy', data)
        # np.save('data_angle_6c_1.npy', data_angle)
        # np.save('data_deg_6c_1.npy', data_deg)
        np.save('label_6c_1_hang_750.npy', label)

    if root_dir == 'data\Baha_et_al\Environment 2':
        np.save('data_6c_2_hang_750.npy', data)
        # np.save('data_angle_6c_2.npy', data_angle)
        # np.save('data_deg_6c_2.npy', data_deg)
        np.save('label_6c_2_hang_750.npy', label)

    if root_dir == 'data\Baha_et_al\Environment 3':
        np.save('data_6c_3_hang_750.npy', data)
        # np.save('data_angle_6c_3.npy', data_angle)
        # np.save('data_deg_6c_3.npy', data_deg)
        np.save('label_6c_3_hang_750.npy', label)
    return data,data_angle, label

#Baha_et_al('data\Baha_et_al\Environment 1')
# Baha_et_al('data\Baha_et_al\Environment 2')
# Baha_et_al('data\Baha_et_al\Environment 3')



def time_alignment(data, time_stamp, num_alignment):
    intervel = (time_stamp[len(time_stamp)-1] - time_stamp[0]) / num_alignment
    cur_range = time_stamp[0] + intervel
    temp_list = []
    new_data = []
    for i in range(len(time_stamp)):
        if time_stamp[i] > cur_range:
            if len(temp_list) != 0:
                new_data.append(average_list(temp_list))
            else:
                new_data.append(data[i])
            temp_list = []
            cur_range = cur_range + intervel
        temp_list.append(data[i])
    if len(temp_list) != 0:
        new_data.append(average_list(temp_list))
    #print(len(new_data))
    if len(new_data) < num_alignment:
        new_data.append(data[len(time_stamp)-1])
        print("!!!!")
    return new_data[:num_alignment]

def load_StanWiFi_data(root,train_test):
    root = root + '/'
    file_list = os.listdir(root)
    label = []
    data = []
    data_phase = []
    aclist = ['bed', 'fall', 'run', 'sitdown', 'standup', 'walk']
    count = 1
    count1 = 0
    for file in file_list:
        print(count,root + file)
        with open(root + file, encoding='utf-8') as f:
            reader = csv.reader(f)
            record = []
            record_phase = []
            time_stamp = []
            for r in reader:
                record.append([float(str_d) for str_d in r[1:91]])
                record_phase.append([float(str_d_1) for str_d_1 in r[91:181]])
                time_stamp.append(float(r[0]))

            record = np.array(record)
            record_phase = np.array(record_phase)
            time_stamp = np.array(time_stamp)
            record = time_alignment(record, time_stamp, 15000)
            record_phase = time_alignment(record_phase, time_stamp,15000)
            data.append(record)
            data_phase.append(record_phase)
            for j in range(len(aclist)):
                if file.find(aclist[j]) != -1:
                    label.append(j)
                    break
        if count % 1 == 0:
            count1 += 1
            print('Save data:')
            data = np.array(data)
            data_phase = np.array(data_phase)
            label = np.array(label)
            np.save('backup/new_data_15000/'+train_test+'/amp/data_amp_'+str(count1)+'.npy', data)
            np.save('backup/new_data_15000/'+train_test+'/phase/data_phase_'+str(count1)+'.npy', data_phase)
            np.save('backup/new_data_15000/'+train_test+'/label/label_'+str(count1)+'.npy', label)
            #print(label)

            label = []
            data = []
            data_phase = []
        count += 1

        #if count == 10:
        #    break
    #data = np.array(data)
    #data_phase = np.array(data_phase)
    #label = np.array(label)
    #print(data_phase.shape)
    #print(data.shape)
    #print(label.shape)
    #exit()
    # count1 += 1
    # np.save('backup/new_data_4000/amp/data_amp_'+str(count1)+'.npy', data)
    # np.save('backup/new_data_4000/phase/data_phase_'+str(count1)+'.npy', data_phase)
    # np.save('backup/new_data_4000/label/label_'+str(count1)+'.npy', label)
    return data, data_phase, label


def load_StanWiFi_data_1(root):
    root = root + '/'
    file_list = os.listdir(root)
    label = []
    data_amp = []
    data_phase = []
    aclist = ['bed', 'fall', 'run', 'sitdown', 'standup', 'walk']
    count = 1
    for file in file_list:
        print(count,root + file)
        with open(root + file, encoding='utf-8') as f:
            reader = csv.reader(f)
            record_amp = []
            record_phase = []
            time_stamp = []
            count1 = 0
            for r in reader:
                count1 += 1
                #if count1 >= 2500 and count1 < 10000:
                if True:
                    record_amp.append([float(str_d) for str_d in r[1:91]])
                    record_phase.append([float(str_d_1) for str_d_1 in r[91:181]])
                    time_stamp.append(float(r[0]))      
            record_amp = np.array(record_amp)
            record_phase = np.array(record_phase)
            print(record_amp.shape)
            exit()
            time_stamp = np.array(time_stamp)
            
            record_amp = time_alignment(record_amp, time_stamp, 2000)

            record_phase = time_alignment(record_phase, time_stamp,2000)
            data_amp.append(record_amp)
            data_phase.append(record_phase)
            for j in range(len(aclist)):
                if file.find(aclist[j]) != -1:
                    label.append(j)
                    break
            print('label:', label)
        count += 1
        # if count == 5:
        #     break
    data_amp = np.array(data_amp)
    data_phase = np.array(data_phase)
    label = np.array(label)
    # print(data_amp.shape)
    # print(data_phase.shape)
    # print(label.shape)
    # exit()
    np.save('data/StanWiFi_2000/data_amp_5000_new_1.npy', data_amp)
    np.save('data/StanWiFi_2000/data_phase_5000_new_1.npy', data_phase)
    np.save('data/StanWiFi_2000/label_5000_new_1.npy', label)
    return data_amp, data_phase, label


load_StanWiFi_data_1("data/StanWifi_/train")
# load_StanWiFi_data("data/StanWifi_/test",'val')

#load_StanWiFi_data_1("data/StanWifi")


def read_data(reader,start,offset):
    record_amp = []
    record_phase = []
    time_stamp = []
    count1 = 0
    for r in reader:
        if count1 >= start and count1 < start + offset:
            record_amp.append([float(str_d) for str_d in r[1:91]])
            record_phase.append([float(str_d_1) for str_d_1 in r[91:181]])
            time_stamp.append(float(r[0])) 
        count1 += 1

    record_amp = np.array(record_amp)
    record_phase = np.array(record_phase)
    time_stamp = np.array(time_stamp)

    return record_amp, record_phase, time_stamp

def load_StanWiFi_data_overlap(root):
    root = root + '/'
    file_list = os.listdir(root)
    label = []
    data_amp = []
    data_phase = []
    aclist = ['bed', 'fall', 'run', 'sitdown', 'standup', 'walk']
    count = 1
    for file in file_list:
        print(count,root + file)

        for i in range(5):
            with open(root + file, encoding='utf-8') as f:
                start = 4000 + i * 200
                reader = csv.reader(f)
                record_amp,record_phase,time_stamp = read_data(reader,start,5000)
                record_amp = time_alignment(record_amp, time_stamp, 2000)
                record_phase = time_alignment(record_phase, time_stamp,2000)
                data_amp.append(record_amp)
                data_phase.append(record_phase)
                for j in range(len(aclist)):
                    if file.find(aclist[j]) != -1:
                        label.append(j)
                        break
        #data_amp = np.array(data_amp)
        #data_phase = np.array(data_phase)
        #label = np.array(label)
        # print(data_amp.shape)
        # print(data_phase.shape)
        # print(label.shape)
        # exit()
        # record_amp = []
        # record_phase = []
        # time_stamp = []
        # count1 = 0
        # for r in reader:
        #     count1 += 1
        #     if count1 >= 5000 and count1 < 10000:
        #         record_amp.append([float(str_d) for str_d in r[1:91]])
        #         record_phase.append([float(str_d_1) for str_d_1 in r[91:181]])
        #         time_stamp.append(float(r[0])) 

        # record_amp = np.array(record_amp)
        # record_phase = np.array(record_phase)
        # time_stamp = np.array(time_stamp)

        #print(record_amp.shape)
        #exit()
        #record_amp = time_alignment(record_amp, time_stamp, 5000)
        #record_phase = time_alignment(record_phase, time_stamp,5000)
        # data_amp.append(record_amp)
        # data_phase.append(record_phase)
        # for j in range(len(aclist)):
        #     if file.find(aclist[j]) != -1:
        #         label.append(j)
        #         break
        print('label:', label)
        count += 1
        # if count == 5:
        #     break
    data_amp = np.array(data_amp)
    data_phase = np.array(data_phase)
    label = np.array(label)
    print(data_amp.shape)
    print(data_phase.shape)
    print(label.shape)
    np.save('data/StanWiFi_2000/data_amp_2000.npy', data_amp)
    np.save('data/StanWiFi_2000/data_phase_2000.npy', data_phase)
    np.save('data/StanWiFi_2000/label_2000.npy', label)
    #exit()
    return data_amp, data_phase, label

#load_StanWiFi_data_overlap("data/StanWifi")

def load_StanWiFi_data_2(root,train_test):
    root = root + '/'
    file_list = os.listdir(root)
    label = []
    data = []
    data_phase = []
    aclist = ['bed', 'fall', 'run', 'sitdown', 'standup', 'walk']
    count = 1
    count1 = 0
    for file in file_list:
        print(count,root + file)
        with open(root + file, encoding='utf-8') as f:
            reader = csv.reader(f)
            record = []
            record_phase = []
            time_stamp = []
            for r in reader:
                record.append([float(str_d) for str_d in r[1:91]])
                record_phase.append([float(str_d_1) for str_d_1 in r[91:181]])
                #time_stamp.append(float(r[0]))

            record = np.array(record)
            record = record[:15000]
            record_phase = np.array(record_phase)
            record_phase = record_phase[:15000]
            #time_stamp = np.array(time_stamp)
            #record = time_alignment(record, time_stamp, 15000)
            #record_phase = time_alignment(record_phase, time_stamp,15000)
            data.append(record)
            data = np.array(data)
            data_phase.append(record_phase)
            data_phase = np.array(data_phase)
            #print(data.shape)
            #print(data_phase.shape)
            
            for j in range(len(aclist)):
                if file.find(aclist[j]) != -1:
                    label.append(j)
                    break
            ##label = np.array(label)
            #print(label.shape)
            #exit()
        if count % 1 == 0:
            count1 += 1
            print('Save data:')
            data = np.array(data)
            data_phase = np.array(data_phase)
            label = np.array(label)
            np.save('backup/new_data_15000_01/'+train_test+'/amp/data_amp_'+str(count1)+'.npy', data)
            np.save('backup/new_data_15000_01/'+train_test+'/phase/data_phase_'+str(count1)+'.npy', data_phase)
            np.save('backup/new_data_15000_01/'+train_test+'/label/label_'+str(count1)+'.npy', label)
            #print(label)

            label = []
            data = []
            data_phase = []
        count += 1

        #if count == 10:
        #    break
    #data = np.array(data)
    #data_phase = np.array(data_phase)
    #label = np.array(label)
    #print(data_phase.shape)
    #print(data.shape)
    #print(label.shape)
    #exit()
    # count1 += 1
    # np.save('backup/new_data_4000/amp/data_amp_'+str(count1)+'.npy', data)
    # np.save('backup/new_data_4000/phase/data_phase_'+str(count1)+'.npy', data_phase)
    # np.save('backup/new_data_4000/label/label_'+str(count1)+'.npy', label)
    return data, data_phase, label
# load_StanWiFi_data_2("data/StanWifi_/train",'train')
# load_StanWiFi_data_2("data/StanWifi_/test",'test')