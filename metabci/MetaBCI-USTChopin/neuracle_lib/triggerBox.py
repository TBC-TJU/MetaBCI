#! /usr/bin/env python  
#  -*- coding:utf-8 -*-
#
# Author: FANG Junying, fangjunying@neuracle.cn
#
# Versions:
# 	v0.1: 2020-02-20, orignal
#
# Copyright (c) 2020 Neuracle, Inc. All Rights Reserved. http://neuracle.cn/

import serial
import serial.tools.list_ports
import time
from ctypes import *

'''
    structure and union
'''

class PackageTriggerBoxBaseFrame(Structure):
    _fields_ = [('deviceID', c_ubyte), ('functionID', c_ubyte), ('payload', c_ushort)]
    _pack_ = 1

class PackageSensorInfo(Structure):
    _fields_ = [('sensorType', c_ubyte), ('sensorNum', c_ubyte)]
    _pack_ = 1

class PackageSensorPara(Structure):
    _fields_ = [('Edge', c_ubyte), ('OutputChannel', c_ubyte), ('TriggerToBeOut', c_ushort),
                ('Threshold', c_ushort), ('EventData', c_ushort)]
    _pack_ = 1

class PackageGetDeviceInfo(Structure):
    # [obj.deviceID 4 typecast(uint16(0), 'uint8')];
    _fields_ = [('frame', PackageTriggerBoxBaseFrame), ('command', c_ubyte)]
    _pack_ = 1

# class PackageGetDeviceName(Structure):
#     # [obj.deviceID 4 typecast(uint16(0), 'uint8')];
#     _fields_ = [('frame', PackageTriggerBoxBaseFrame), ('command', c_ubyte)]
#     _pack_ = 1

class PackageGetSensorPara(Structure):
    _fields_ = [('frame', PackageTriggerBoxBaseFrame), ('sensorInfo', PackageSensorInfo)]
    _pack_ = 1

class PackageSetSensorPara(Structure):
    _fields_ = [('frame', PackageTriggerBoxBaseFrame), ('sensorInfo', PackageSensorInfo), ('sensorPara', PackageSensorPara)]
    _pack_ = 1


class TriggerIn(object):
    def __init__(self, serial_name):
        self._serial_name = serial_name
        self._device_comport_handle = None
        # self.validate_device()

    def validate_device(self):
        self._device_comport_handle = serial.Serial(self._serial_name, baudrate=115200, timeout= 0)
        if self._device_comport_handle.isOpen():
            print("Open %s successfully." % (self._serial_name))
            return True
        else:
            print("Open %s failed." % (self._serial_name))
            return False

    def output_event_data(self,eventData):
        cmd = PackageGetDeviceInfo()
        cmd.command = eventData
        cmd.frame.deviceID = 1
        cmd.frame.functionID = 225
        cmd.frame.payload = 1
        self._device_comport_handle.flushInput()
        self._device_comport_handle.write(cmd)

class TriggerBox(object):
    functionIDSensorParaGet = 1
    functionIDSensorParaSet = 2
    functionIDDeviceInfoGet = 3
    functionIDDeviceNameGet = 4
    functionIDSensorSampleGet = 5
    functionIDSensorInfoGet = 6
    functionIDOutputEventData = 225
    functionIDError = 131

    sensorTypeDigitalIN = 1
    sensorTypeLight = 2
    sensorTypeLineIN = 3
    sensorTypeMic = 4
    sensorTypeKey = 5
    sensorTypeTemperature = 6
    sensorTypeHumidity = 7
    sensorTypeAmbientlight = 8
    sensorTypeDebug = 9
    sensorTypeAll = 255

    sensorTypeMap = {sensorTypeDigitalIN: 'DigitalIN',
                     sensorTypeLight: 'Light',
                     sensorTypeLineIN: 'LineIN',
                     sensorTypeMic: 'Mic',
                     sensorTypeKey: 'Key',
                     sensorTypeTemperature: 'Temperature',
                     sensorTypeHumidity: 'Humidity',
                     sensorTypeAmbientlight: 'Ambientlight',
                     sensorTypeDebug: 'Debug'}
    _deviceID = 1
    _sensor_info = []
    def __init__(self, serial_name):
        self._serial_name = serial_name
        self._port_list = self.refresh_serial_list()
        self._device_comport_handle = None
        self._device_name = None
        self._device_info = None
        self.validate_device()
        self.get_device_name()
        self.get_device_info()
        self.get_sensor_info()

    def refresh_serial_list(self):
        return list(serial.tools.list_ports.comports())

    def validate_device(self):
        if not self.check_online():
            return False
        self._device_comport_handle = serial.Serial(self._serial_name,baudrate=115200,timeout=60)
        if self._device_comport_handle.isOpen():
            print("Open %s successfully." % (self._serial_name))
            recv = self.get_device_name()
            if recv == None:
                print('Not a valid device due to response for getting device name is none!!')
                return False
            self._device_name = recv
            return True

        else:
            print("Open %s failed." % (self._serial_name))
            return False

    def get_device_name(self):
        cmd = PackageTriggerBoxBaseFrame()
        cmd.deviceID = self._deviceID
        cmd.functionID = self.functionIDDeviceNameGet
        cmd.payload = 0
        self.send(cmd)
        data = self.read(cmd.functionID)
        # data[0] = deviceID, data[1] = functionID, data[2:3] = payload,device_name = str(data[4:])
        device_name = str(data)
        return device_name

    def get_device_info(self):
        cmd = PackageGetDeviceInfo()
        cmd.command = 1
        cmd.frame.deviceID = self._deviceID
        cmd.frame.functionID = self.functionIDDeviceInfoGet
        # cmd.frame.payload = len(cmd.command)
        cmd.frame.payload = 1
        self.send(cmd)
        data = self.read(cmd.frame.functionID)
        '''
            # rspPayload = data[2] | (data[3] << 8)
            # print("getDeviceInfo response payload : %d" % (rspPayload))
            # HardwareVersion = data[4]
            # FirmwareVersion = data[5]
            # sensorSum   = data[6]
            # # resv1       = data[7]
            # ID = (data[8] << 24) | (data[9] << 16) | (data[10] << 8) | data[11]
        '''
        HardwareVersion = data[0]
        FirmwareVersion = data[1]
        sensorSum = data[2]
        # resv1       = data[3]
        ID = (data[4] << 24) | (data[5] << 16) | (data[6] << 8) | data[7]
        # print("HardwareVersion: %d, firmwareVersion: %d, sensorNum: %d,  ID: %d"
        #       % (HardwareVersion, FirmwareVersion, sensorSum, int(ID)))
        self._device_info = dict(HardwareVersion=HardwareVersion, FirmwareVersion=FirmwareVersion, sensorSum=sensorSum, ID=ID)

    def _getSensorTypeString(self, sensorType):
        if sensorType not in self.sensorTypeMap.keys():
            return "Undefined"
        return self.sensorTypeMap[sensorType]

    def get_sensor_info(self):
        '''
        a little strange:
        SensorType :           Light, SensorNum: 1
        SensorType :           Light, SensorNum: 2    <---- Why have 2 sensorType?
        SensorType :          LineIN, SensorNum: 1
        SensorType :          LineIN, SensorNum: 2
        SensorType :    Ambientlight, SensorNum: 1
        SensorType :             Mic, SensorNum: 1
        SensorType :        Humidity, SensorNum: 1
        SensorType :     Temperature, SensorNum: 1
        SensorType :           Debug, SensorNum: 1
        SensorType :       Undefined, SensorNum: 0
        SensorType :       Undefined, SensorNum: 0
        '''
        cmd = PackageTriggerBoxBaseFrame()
        cmd.deviceID = self._deviceID
        cmd.functionID = self.functionIDSensorInfoGet
        cmd.payload = 0
        self.send(cmd)
        info = self.read(cmd.functionID)
        if len(info) % 2 != 0:
            raise Exception("Response length is not correct %d" % (len(info)))
        for i in range(int(len(info) / 2)):
            sensorTypeIdx = info[i * 2]
            sensorNum = info[i * 2 + 1]
            sensorType = self._getSensorTypeString(sensorTypeIdx)
            print("SensorType : %15s, SensorNum: %d " % (sensorType, sensorNum))
            self._sensor_info.append(dict(Type=sensorType, Number=sensorNum))
        # print(self._sensor_info)
        return

    def _sensor_type(self, typeString):
        if typeString == 'DigitalIN':
            typeNum = self.sensorTypeDigitalIN
        elif typeString == 'Light':
            typeNum = self.sensorTypeLight
        elif typeString == 'LineIN':
            typeNum = self.sensorTypeLineIN
        elif typeString == 'Mic':
            typeNum = self.sensorTypeMic
        elif typeString == 'Key':
            typeNum = self.sensorTypeKey
        elif typeString == 'Temperature':
            typeNum = self.sensorTypeTemperature
        elif typeString == 'Humidity':
            typeNum = self.sensorTypeHumidity
        elif typeString == 'Ambientlight':
            typeNum = self.sensorTypeAmbientlight
        elif typeString == 'Debug':
            typeNum = self.sensorTypeDebug
        else:
            raise Exception('Undefined sensor type')
        return typeNum

    def get_sensor_para(self, sensorID):
        sensor = self._sensor_info[sensorID]
        cmd = PackageGetSensorPara()
        cmd.sensorInfo.sensorType = self._sensor_type(typeString=sensor['Type'])
        cmd.sensorInfo.sensorNum = sensor['Number']
        cmd.frame.deviceID = 1
        cmd.frame.functionID = self.functionIDSensorParaGet
        cmd.frame.payload = 2
        self.send(cmd)
        para = self.read(cmd.frame.functionID)
        sensorPara = PackageSensorPara()
        sensorPara.Edge = para[0]
        sensorPara.OutputChannel = para[1]
        sensorPara.TriggerToBeOut = para[2] | (para[3] << 8)
        sensorPara.Threshold = para[4] | (para[5] << 8)
        sensorPara.EventData = para[6] | (para[7] << 8)
        # print("Edge: %d, OutputChannel: %d, TriggerToBeOut: %d, Threshold: %d, EventData: %d" % (
        # sensorPara.Edge, sensorPara.OutputChannel, sensorPara.TriggerToBeOut, sensorPara.Threshold, sensorPara.EventData))
        return sensorPara

    def set_sensor_para(self, sensorID, sensorPara):
        sensor = self._sensor_info[sensorID]
        cmd = PackageSetSensorPara()
        cmd.frame.deviceID = self._deviceID
        cmd.frame.functionID = self.functionIDOutputEventData #self.functionIDSensorParaSetF
        cmd.frame.payload = 10
        cmd.sensorInfo.sensorType = self._sensor_type(typeString=sensor['Type'])
        cmd.sensorInfo.sensorNum = sensor['Number']
        cmd.sensorPara.Edge = sensorPara.Edge
        cmd.sensorPara.OutputChannel = sensorPara.OutputChannel
        cmd.sensorPara.TriggerToBeOut = sensorPara.TriggerToBeOut
        cmd.sensorPara.Threshold = sensorPara.Threshold
        cmd.sensorPara.EventData = sensorPara.EventData
        self.send(cmd)
        data = self.read(cmd.frame.functionID)
        if data[0] == cmd.sensorInfo.sensorType and data[1] == cmd.sensorInfo.sensorNum:
            print("setSensorPara successfully...")
        else:
            print("setSensorPara failed...")
        return

    def get_sensor_sample(self, sensorID):
        sensor = self._sensor_info[sensorID]
        cmd = PackageGetSensorPara()
        cmd.frame.deviceID = 1
        cmd.frame.functionID = self.functionIDSensorSampleGet
        cmd.frame.payload = 2
        cmd.sensorInfo.sensorType = self._sensor_type(typeString=sensor['Type'])
        cmd.sensorInfo.sensorNum = sensor['Number']
        self.send(cmd)
        data = self.read(cmd.frame.functionID)
        adcResult = 0
        if data[0] == cmd.sensorInfo.sensorType and data[1] == cmd.sensorInfo.sensorNum:
            adcResult = data[2] | (data[3] << 8)
            print("getSensorSample successfully...adcResult: %d" % (adcResult))
        else:
            print("getSensorSample failed...")
        return adcResult

    def set_event_data(self, sensorID, eventData, triggerTOBeOut=1):
        sensorPara = self.get_sensor_para(sensorID)
        sensorPara.TriggerToBeOut = triggerTOBeOut
        sensorPara.EventData = eventData
        self.set_sensor_para(sensorID,sensorPara)
        return

    def output_event_data(self, eventData, triggerToBeOut=1):
        '''

        :param eventData:
        :param triggerToBeOut:
        :return:
        '''

        # 9 : debug
        # sensorID = 8
        # sensorPara = PackageSensorPara()
        # sensorPara.Edge = 1
        # sensorPara.OutputChannel = 3
        # sensorPara.TriggerToBeOut = triggerToBeOut
        # sensorPara.Threshold = 0
        # sensorPara.EventData = eventData
        # self.set_sensor_para(sensorID, sensorPara)

        cmd = PackageGetDeviceInfo()
        cmd.command = eventData
        cmd.frame.deviceID = self._deviceID
        cmd.frame.functionID = self.functionIDOutputEventData
        cmd.frame.payload = 1
        self.send(cmd)
        data = self.read(cmd.frame.functionID)
        isSucceed = data[0] == self.functionIDOutputEventData
        return

    def check_online(self):
        if len(self._port_list) <= 0:
            print("Can't find any serial port online.")
            return False
        for idx, p in enumerate(self._port_list):
            if p.device == self._serial_name:
                print("Target serial [%s] port (%s) online." % (p.device, p.description))
                return True
        print("Target serial [%s] port offline.\n" % (self._serial_name))
        print("Online serial list:")
        for idx, p in enumerate(self._port_list):
            print("%s : %s" % (p.device, p.description))
        return False

    def send(self, data):
        self._device_comport_handle.flushInput()
        self._device_comport_handle.write(data)
        # time.sleep(0.5)
        return
    def read(self,functionID):
        self._device_comport_handle.flushOutput()
        message = self._device_comport_handle.read(4)
        print(message)
        if message[0] != self._deviceID:
            raise Exception("Response error: request deviceID %d, return deviceID %d" % (self._deviceID, message[0]))
        if message[1] != functionID:
            if message[1] == self.functionIDError:
                error_type = self._device_comport_handle.read()[0]
                if error_type == 0:
                    error_message = 'None'
                elif error_type == 1:
                    error_message = 'FrameHeader'
                elif error_type == 2:
                    error_message = 'FramePayload'
                elif error_type == 3:
                    error_message = 'ChannelNotExist'
                elif error_type == 4:
                    error_message = 'DeviceID'
                elif error_type == 5:
                    error_message = 'FunctionID'
                elif error_type == 6:
                    error_message = 'SensorType'
                else:
                    raise Exception('Undefined error type')
                raise Exception("Response error: %s" % (error_message))
            else:
                raise Exception("Response error: request functionID %d, return functionID  %d" % (functionID, message[1]))

        rspPayload = message[2] | (message[3] << 8)
        # print("getDeviceInfo response payload : %d" % (rspPayload))
        # recv = self._device_comport_handle.read_all()
        recv = self._device_comport_handle.read(rspPayload)
        print(str(recv))
        return recv

    def set_audioSensor_threshold(self, sensorID):
        # Not used now
        pass

    def init_audioSensor(self, sensorID):
        # Not used now
        pass
    def set_lightSensor_threshold(self, sensorID):
        # Not used now
        pass
    def init_lightSensor(self, sensorID):
        sensorPara = self.get_sensor_para(sensorID)
        sensorPara.OutputChannel = 3
        sensorPara.TriggerToBeOut = 0
        sensorPara.EventData = 0
        self.set_sensor_para(sensorID, sensorPara)
        self.set_lightSensor_threshold(sensorID)
        # set_lightSensor_threshold should be moved out of class TriggerBox
        return

    def closeSerial(self):
        self._device_comport_handle.close()















