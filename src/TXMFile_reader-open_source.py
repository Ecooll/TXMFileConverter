
"""
TXM File Converter
Author: Evan(https://github.com/Ecooll)

license: 'Mozilla Public License 2.0',
version: v0.0.1
last edit time: 2022/09/06

"""

import os
import olefile as of
import numpy as np
import array
from matplotlib import pyplot as plt
import struct
import re
from skimage.transform import resize
from gooey import Gooey, GooeyParser
from scipy.interpolate import RegularGridInterpolator
import traceback
import psutil
import sys

dir2 = None


sys.tracebacklimit = -1


class OleReader:

    def __init__(self, inputFilename_i, outputFilename_i, previewMode_i, disableTrimming_i, trimmingSize_i,
                 targetSize_i):

        self.inputFileName = inputFilename_i
        self.outputFileName = outputFilename_i
        self.previewMode = previewMode_i
        self.disableTrimming = disableTrimming_i
        self.trimmingSize = trimmingSize_i
        self.targetSize = targetSize_i
        self.ole = None
        self.rawWidth = None
        self.rawHeight = None
        self.oriSize = None
        self.img_dir = []
        self.dataIn3D = None

    def ReadTxm(self):
        os.system('cls')

        try:
            self.ole = of.OleFileIO(self.inputFileName)
        except:
            print(r'Can not open txm file')
            exit(1)

        if self.previewMode == False:
            try:
                if self.outputFileName[-4:] != ('.csv'):
                    self.outputFileName = self.outputFileName + '.csv'
                test_f = open(self.outputFileName, 'w')
                test_f.close()
                print('Input File：' + str(self.inputFileName))
                print('Output File：' + str(self.outputFileName))
            except:
                print(r'Can not create target file')
                exit(1)

        dir1 = self.ole.listdir(self.ole)
        self.rawWidth = self.ole.openstream(['ImageInfo', 'ImageWidth'])
        self.rawWidth = self.rawWidth.read()
        self.rawWidth = struct.unpack('i', self.rawWidth)[0]
        self.rawHeight = self.ole.openstream(['ImageInfo', 'ImageHeight'])
        self.rawHeight = self.rawHeight.read()
        self.rawHeight = struct.unpack('i', self.rawHeight)[0]
        dir2 = np.array(dir1, dtype=object)
        for i in range(dir2.shape[0]):
            if re.match('ImageData', dir2[i][0]):
                self.img_dir.append(dir2[i])

        if self.rawHeight % 2 != 0 and self.rawHeight > 0:
            self.rawHeight = self.rawHeight - 1

        if self.rawWidth % 2 != 0 and self.rawWidth > 0:
            self.rawWidth = self.rawWidth - 1

        if not self.disableTrimming:
            rawSizeHeight = (self.trimmingSize[1] - self.trimmingSize[0])
            rawSizeWidth = (self.trimmingSize[3] - self.trimmingSize[2]) * 2
        else:
            rawSizeHeight = self.rawHeight
            rawSizeWidth = self.rawWidth

        self.oriSize = [len(self.img_dir), rawSizeHeight, rawSizeWidth, 0, 0]
        x1 = len(self.img_dir) * rawSizeHeight * rawSizeWidth

        x2 = int(psutil.virtual_memory()[4] / 4)

        x3 = x1 / x2
        if x3 > 0.9:
            self.oriSize[3] = int(x3) + 1
        else:
            self.oriSize[3] = 1

        if self.previewMode == True:
            try:
                print('\nStarting Preview')
                self.PreviewMode()
                print('All Done!')
                return 0
            except:
                print('\nPreview Failed\nTry to re-set the Trimming settings')
                print('Error code:', traceback.format_exc().split(',')[1][6:])
                return -1

        if not self.previewMode:
            try:
                self.dataIn3D = np.zeros(
                    np.arange(0, len(self.img_dir), self.oriSize[3], dtype=int).shape[0] * rawSizeHeight * rawSizeWidth,
                    dtype=np.float32)
            except:
                print('\nSystem memory access error')
                return -1

            try:
                print('\nStarting Convert Raw Data')
                self.ReadTXMFile()
            except:
                print('\nRead Data Failed, please contact developer')
                print('Error code:', traceback.format_exc().split(',')[1][6:])
                return -1

            try:
                print('\nStart Resizing Model')
                self.ModelResize()
                print('\nResizing Model End')
            except:
                print('\nResizing Model Failed, please contact developer')
                print('Error code:', traceback.format_exc().split(',')[1][6:])
                return -1

            try:
                if self.previewMode == False and self.dataIn3D.shape[1] > 1:
                    try:
                        print('\nStart Writing CSV File')
                        self.ExportCSVFile()
                        print('\nAll done!')
                    except:
                        print('Write CSV Data Failed, please contact developer')
                        print('Error code:', traceback.format_exc().split(',')[1][6:])
                        return 1
            except:
                raise ('Output data error')


    def PreviewMode(self):
        print('=========##################=========')
        print('Raw file size:')
        print('Depth:', len(self.img_dir),
              '\tWidth:', struct.unpack('i', self.ole.openstream(['ImageInfo', 'ImageWidth']).read())[0],
              '\tHeight:', struct.unpack('i', self.ole.openstream(['ImageInfo', 'ImageHeight']).read())[0]
              )
        print('=========##################=========')

        ole = self.ole.openstream(self.img_dir[int(len(self.img_dir) / 2)])
        img = ole.read()
        img_arr = array.array('i', img)
        img_list = img_arr.tolist()

        if (int(self.rawHeight / 2) * int(self.rawWidth / 2)) < len(img_list):
            img_list = img_list[:(int(self.rawHeight / 2) * int(self.rawWidth / 2))]

        img_list = np.reshape(img_list, (int(self.rawHeight / 2), int(self.rawWidth / 2)))
        img_right = img_list[:, :int(self.rawWidth / 4)]
        img_left = img_list[:, int(self.rawWidth / 4):]
        img_list = img_right + img_left
        if not self.disableTrimming:
            img_list = img_list[self.trimmingSize[0]:self.trimmingSize[1], self.trimmingSize[2]:self.trimmingSize[3]]
        img_list = resize(img_list, (img_list.shape[0], img_list.shape[1] * 2))
        plt.imshow(img_list)
        plt.show()

    def ReadTXMFile(self):
        if not self.disableTrimming:
            rawSizeHeight = (self.trimmingSize[1] - self.trimmingSize[0])
            rawSizeWidth = (self.trimmingSize[3] - self.trimmingSize[2]) * 2
        else:
            rawSizeHeight = self.rawHeight
            rawSizeWidth = self.rawWidth
        counterFlag = -1
        conter_img = 0

        raw_seq = np.arange(0, len(self.img_dir), self.oriSize[3], dtype=int)
        self.oriSize[4] = raw_seq.shape[0]

        for i in raw_seq:
            ole = self.ole.openstream(self.img_dir[i])
            img = ole.read()
            img_arr = array.array('i', img)
            img_list = img_arr.tolist()
            if (int(self.rawHeight / 2) * int(self.rawWidth / 2)) < len(img_list):
                img_list = img_list[:(int(self.rawHeight / 2) * int(self.rawWidth / 2))]
            img_list = np.reshape(img_list, (int(self.rawHeight / 2), int(self.rawWidth / 2)))
            img_right = img_list[:, :int(self.rawWidth / 4)]
            img_left = img_list[:, int(self.rawWidth / 4):]
            img_list = img_right + img_left
            del img_right, img_left, img_arr, img, ole
            if not self.disableTrimming:
                img_list = img_list[self.trimmingSize[0]:self.trimmingSize[1],
                           self.trimmingSize[2]:self.trimmingSize[3]]
                img_list = resize(img_list, (img_list.shape[0], img_list.shape[1] * 2))
            else:
                img_list = resize(img_list, (self.rawHeight, self.rawWidth))

            img_list = np.array(img_list, dtype=np.float32).flatten()

            baseAddress = conter_img * (rawSizeHeight * rawSizeWidth)
            self.dataIn3D[baseAddress:(baseAddress + (rawSizeWidth * rawSizeHeight))] = img_list
            del img_list

            conter_img = conter_img + 1
            counter = int(conter_img / raw_seq.shape[0] * 100)
            if counter % 10 == 0 and counterFlag != counter:
                counterFlag = counter
                print(str(conter_img) + ' / ' + str(raw_seq.shape[0]), end=' / ')
                print(counter, '%')
        print('Read raw data complete')
        return 0

    def ExportCSVFile(self):
        np.savetxt(self.outputFileName, self.dataIn3D, delimiter=",", fmt='%s')

    def ModelResize(self):
        self.oriSize[0] = self.oriSize[4]
        originalSize = self.oriSize[0:3]

        oriData_pos = np.zeros((self.targetSize[0] * self.targetSize[1] * self.targetSize[2], 3), dtype=np.float32)
        for x in np.arange(0, self.targetSize[0], 1):  # 1/10， （1/10）/10
            counter_point = 0
            baseAddress = x * (self.targetSize[1] * self.targetSize[2])
            for y in np.arange(0, self.targetSize[1], 1):
                for z in np.arange(0, self.targetSize[2], 1):
                    oriData_pos[baseAddress + counter_point] = np.float32([x, y, z])
                    counter_point = counter_point + 1

        # 还原数组
        self.dataIn3D = self.dataIn3D.reshape(originalSize)

        temp_lx = np.linspace(0, self.targetSize[0], originalSize[0])
        temp_ly = np.linspace(0, self.targetSize[1], originalSize[1])
        temp_lz = np.linspace(0, self.targetSize[2], originalSize[2])

        interpolatingFunction = RegularGridInterpolator((temp_lx, temp_ly, temp_lz), self.dataIn3D)
        resizedPointMatrix = interpolatingFunction((oriData_pos[:, 0], oriData_pos[:, 1], oriData_pos[:, 2]))

        pointsNumTotal = resizedPointMatrix.shape[0]

        maxPoint = resizedPointMatrix.max()
        minPoint = resizedPointMatrix.min()
        counterFlag = -1
        for xx in range(pointsNumTotal):
            temp = resizedPointMatrix[xx]
            resizedPointMatrix[xx] = np.float32((temp - minPoint) * pow((maxPoint - minPoint), -1))
            # print(xx)
            counter = int(xx / pointsNumTotal * 100)
            if counter % 10 == 0 and counterFlag != counter:
                counterFlag = counter
                print(xx, ' / ', pointsNumTotal, ' / ', counter, '%')

        resizedPointMatrix = resizedPointMatrix.reshape(self.targetSize[0], self.targetSize[1], self.targetSize[2])

        self.dataIn3D = None

        self.dataIn3D = np.zeros((self.targetSize[0] * self.targetSize[1] * self.targetSize[2], 4), dtype=np.float32)

        for z in range(self.targetSize[2]):
            counter_point = 0
            baseAddress = z * (self.targetSize[1] * self.targetSize[0])
            for y in range(self.targetSize[1]):
                for x in range(self.targetSize[0]):
                    self.dataIn3D[baseAddress + counter_point] = ([x, y, z, resizedPointMatrix[x, y, z]])
                    resizedPointMatrix[x, y, z] = 0
                    counter_point = counter_point + 1



def getargv():
    """
    Parse input arguments
    """
    parser = GooeyParser()

    filesSetting = parser.add_argument_group('Files', gooey_options={'show_border': True, 'columns': 1})

    filesSetting.add_argument('Input_File', type=str, help='Intput File Path',
                              widget="FileChooser")

    filesSetting.add_argument('Output_File', type=str, help='Output File Path',
                              widget="FileSaver")

    inputSetting = parser.add_argument_group(' ', gooey_options={'show_border': True, 'columns': 2})
    inputSetting.add_argument('--Input_setting_ignore-me', gooey_options={'visible': False})

    inputSetting.add_argument("-p", "--Preview_Mode", action='store_true', help="Enable preview mode")
    inputSetting.add_argument("-tr", "--Trimming_Switch", action='store_true', help="Disable Trimming")
    TrimmingFile = inputSetting.add_argument_group('Trimming input file', gooey_options={
        'show_border': True,
        'columns': 2
    })
    width_info = 'Please set bigger than 0'
    height_info = 'Please set bigger than 0'
    width_info_end = 'Please set bigger than Trimming width start'
    height_info_end = 'Please set bigger than Trimming height start'
    TrimmingFile.add_argument('T_width_start', metavar='Trimming width start', type=int, default=140, help=width_info)
    TrimmingFile.add_argument('T_height_start', metavar='Trimming height start', type=int, default=288,
                              help=height_info)
    TrimmingFile.add_argument('T_width_end', metavar='Trimming width end', type=int, default=239, help=width_info_end)
    TrimmingFile.add_argument('T_height_end', metavar='Trimming height end', type=int, default=478,
                              help=height_info_end)

    OutputSetting = inputSetting.add_argument_group('Output setting', gooey_options={
        'show_border': True,
        'columns': 1
    })

    OutputSetting.add_argument('I_width', metavar='Resized model width', default=100, type=int,
                               help="Please set bigger than 0")
    OutputSetting.add_argument('I_height', metavar='Resized model height', default=100, type=int,
                               help="Please set bigger than 0")
    OutputSetting.add_argument('I_depth', metavar='Resized model depth', default=250, type=int,
                               help="Please set bigger than 0")

    args = parser.parse_args()

    inputFilename = args.Input_File
    outputFilename = args.Output_File
    previewMode = args.Preview_Mode
    trimmingSize = [args.T_height_start, args.T_height_end, args.T_width_start, args.T_width_end]
    I_size = [args.I_depth, args.I_width, args.I_height]
    disableTrimming = args.Trimming_Switch

    if inputFilename == outputFilename:
        print("Input file can not be output file")
        exit(-2)

    if trimmingSize[0] >= trimmingSize[1] or trimmingSize[2] >= trimmingSize[3]:
        print('Wrong Trimming size, Please check')
        exit(-1)

    return [inputFilename, outputFilename, previewMode, disableTrimming, trimmingSize, I_size]


@Gooey(program_name="TXM File Converter- Open Source",
       language='english',
       default_size=(800, 900),
       menu=[{'name': 'Help',
              'items': [{
                  'type': 'AboutDialog',
                  'menuTitle': 'About',
                  'name': 'TXM File Converter',
                  'description': 'Zeiss TXM File Converter Open Source Version',
                  'website': 'https://github.com/Ecooll/TXMFileReader',
                  'developer': 'https://github.com/Ecooll',
                  'license': 'Mozilla Public License 2.0',
                  'version': '0.0.1',
                  'copyright': '2022',
              }]
              }]
       )
def main():
    [inputFilename, outputFilename, previewMode, disableTrimming, T_size,
     I_size] = getargv()

    txmReader = OleReader(inputFilename, outputFilename, previewMode, disableTrimming, T_size,
                          I_size)

    status = txmReader.ReadTxm()

    try:
        if status == 0 or status == -1:
            raise ('Program error return code', status)
    except:
        None




if __name__ == '__main__':
    main()
