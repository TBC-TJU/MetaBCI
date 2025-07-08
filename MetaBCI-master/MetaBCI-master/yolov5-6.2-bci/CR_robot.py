import socket

import numpy as np
from realsenseD415 import Camera

class CR_bringup:
    def __init__(self, ip, port):
        self.ip = ip
        self.port = port
        self.socket_dashboard = 0

            
    def EnableRobot(self):
        """
        Enable the robot
        """
        if self.port == 29999:
            try:
                self.socket_dashboard = socket.socket()
                self.socket_dashboard.connect((self.ip, self.port))
                string = "EnableRobot()"
                print(string)
                self.socket_dashboard.send(str.encode(string, 'utf-8'))

            except socket.error:
                print("Fail to setup socket connection !", socket.error)
        else:
            print("Connect to dashboard server need use port 29999 !")
        data = self.socket_dashboard.recv(1024)
        print('receive:', bytes.decode(data, 'utf-8'))
        if(self.socket_dashboard != 0):
            self.socket_dashboard.close()

    def DisableRobot(self):
        """
        Disabled the robot
        """
        if self.port == 29999:
            try:     
                self.socket_dashboard = socket.socket() 
                self.socket_dashboard.connect((self.ip, self.port))
                string = "DisableRobot()"
                print(string)
                self.socket_dashboard.send(str.encode(string, 'utf-8'))
            except socket.error:
                print("Fail to setup socket connection !", socket.error)
        else:
            print("Connect to dashboard server need use port 29999 !")
        data = self.socket_dashboard.recv(1024)
        print('receive:', bytes.decode(data, 'utf-8'))
        if(self.socket_dashboard != 0):
            self.socket_dashboard.close()
            
    def GetPose(self):
        """
        get robot's pose
        """
        if self.port == 29999:
            try:     
                self.socket_dashboard = socket.socket() 
                self.socket_dashboard.connect((self.ip, self.port))
                string = "GetPose()"
                print(string)
                self.socket_dashboard.send(str.encode(string, 'utf-8'))
            except socket.error:
                print("Fail to setup socket connection !", socket.error)
        else:
            print("Connect to dashboard server need use port 29999 !")
        data = self.socket_dashboard.recv(1024)
        receive = bytes.decode(data, 'utf-8')
        # print('receive:', receive)
        
        actual_tool_positions = []
        msg = receive.split(',')
        if int(msg[0]) == 0:
            for i in range(1, 7):
                if i == 1:
                    msg_ = msg[i].split('{')
                    msg[i] = float(msg_[1])
                if i == 6:
                    msg_ = msg[i].split('}')
                    msg[i] = float(msg_[0])
                else:
                    msg[i] = float(msg[i])
                actual_tool_positions.append(msg[i])
        
        if(self.socket_dashboard != 0):
            self.socket_dashboard.close()
        
        if(self.socket_dashboard != 0):
            self.socket_dashboard.close()
            
        return actual_tool_positions
            
    def GetAngle(self):
        """
        get robot's pose
        """
        if self.port == 29999:
            try:     
                self.socket_dashboard = socket.socket() 
                self.socket_dashboard.connect((self.ip, self.port))
                string = "GetAngle()"
                print(string)
                self.socket_dashboard.send(str.encode(string, 'utf-8'))
            except socket.error:
                print("Fail to setup socket connection !", socket.error)
        else:
            print("Connect to dashboard server need use port 29999 !")
        data = self.socket_dashboard.recv(1024)
        print('receive:', bytes.decode(data, 'utf-8'))
        if(self.socket_dashboard != 0):
            self.socket_dashboard.close()
            
    def SetArmOrientation(self, r, d, n, cfg):
        """
        Set the hand command
        r : Mechanical arm direction, forward/backward (1:forward -1:backward)
        d : Mechanical arm direction, up elbow/down elbow (1:up elbow -1:down elbow)
        n : Whether the wrist of the mechanical arm is flipped (1:The wrist does not flip -1:The wrist flip)
        cfg :Sixth axis Angle identification
            (1, - 2... : Axis 6 Angle is [0,-90] is -1; [90, 180] - 2; And so on
            1, 2... : axis 6 Angle is [0,90] is 1; [90180] 2; And so on)
        """
        if self.port == 29999:
            try:     
                self.socket_dashboard = socket.socket() 
                self.socket_dashboard.connect((self.ip, self.port))
                string = "SetArmOrientation({:d},{:d},{:d},{:d})".format(r,d,n,cfg)
                print(string)
                self.socket_dashboard.send(str.encode(string, 'utf-8'))
            except socket.error:
                print("Fail to setup socket connection !", socket.error)
        else:
            print("Connect to dashboard server need use port 29999 !")
        data = self.socket_dashboard.recv(1024)
        print('receive:', bytes.decode(data, 'utf-8'))
        if(self.socket_dashboard != 0):
            self.socket_dashboard.close()
            
    def ClearError(self):
        """
        Clear controller alarm information
        """
        if self.port == 29999:
            try:     
                self.socket_dashboard = socket.socket() 
                self.socket_dashboard.connect((self.ip, self.port))
                string = "ClearError()"
                print(string)
                self.socket_dashboard.send(str.encode(string, 'utf-8'))
            except socket.error:
                print("Fail to setup socket connection !", socket.error)
        else:
            print("Connect to dashboard server need use port 29999 !")
        data = self.socket_dashboard.recv(1024)
        print('receive:', bytes.decode(data, 'utf-8'))
        if(self.socket_dashboard != 0):
            self.socket_dashboard.close()
            
    def ToolDOExecute(self, index, status):
        """
        Set terminal signal output (Instructions immediately)
        index : Terminal output index (Value range:1~2)
        status : Status of digital signal output port(0:Low level，1:High level)
        """
        if self.port == 29999:
            try:     
                self.socket_dashboard = socket.socket() 
                self.socket_dashboard.connect((self.ip, self.port))
                string = "ToolDOExecute({:d},{:d})".format(index,status)
                print(string)
                self.socket_dashboard.send(str.encode(string, 'utf-8'))
            except socket.error:
                print("Fail to setup socket connection !", socket.error)
        else:
            print("Connect to dashboard server need use port 29999 !")
        data = self.socket_dashboard.recv(1024)
        print('receive:', bytes.decode(data, 'utf-8'))
        if(self.socket_dashboard != 0):
            self.socket_dashboard.close()
        

class CR_ROBOT:
    def __init__(self, ip, port):
        self.ip = ip
        self.port = port
        self.is_use_camera = True
        self.socket_feedback = 0 
        
        # realsense configuration
        if(self.is_use_camera):
            # Fetch RGB-D data from RealSense camera
            self.camera = Camera()
            #self.cam_intrinsics = self.camera.intrinsics  # get camera intrinsics
        self.cam_intrinsics = np.array([380.15350342, 0., 316.6630249,0., 379.74990845, 239.63673401,0,0,1]).reshape(3,3)

        # self.cam_pose = np.loadtxt('cam_pose/camera_pose.txt', delimiter=' ')
        # self.cam_depth_scale = np.loadtxt('cam_pose/camera_depth_scale.txt', delimiter=' ')

    def MovJ(self,  x, y, z, a, b, c):
        """
        Joint motion interface (point-to-point motion mode)
        x: A number in the Cartesian coordinate system x
        y: A number in the Cartesian coordinate system y
        z: A number in the Cartesian coordinate system z
        a: A number in the Cartesian coordinate system a
        b: A number in the Cartesian coordinate system b
        c: A number in the Cartesian coordinate system c
        """
        if self.port == 30003:
            try:     
                self.socket_feedback = socket.socket() 
                self.socket_feedback.connect((self.ip, self.port))
                string = "MovJ({:f},{:f},{:f},{:f},{:f},{:f})".format(x,y,z,a,b,c)
                print(string)
                self.socket_feedback.send(str.encode(string, 'utf-8'))
            except socket.error:
                print("Fail to setup socket connection !", socket.error)
        else:
            print("Connect to dashboard server need use port 29999 !")
        data = self.socket_feedback.recv(1024)
        print('receive:', bytes.decode(data, 'utf-8'))
        if(self.socket_feedback != 0):
            self.socket_feedback.close()
            
    def JointMovJ(self, j1, j2, j3, j4, j5, j6):
        """
        Joint motion interface (linear motion mode)
        j1~j6:Point position values on each joint
        """
        if self.port == 30003:
            try:     
                self.socket_feedback = socket.socket() 
                self.socket_feedback.connect((self.ip, self.port))
                string = "JointMovJ({:f},{:f},{:f},{:f},{:f},{:f})".format(j1,j2,j3,j4,j5,j6)
                print(string)
                self.socket_feedback.send(str.encode(string, 'utf-8'))
            except socket.error:
                print("Fail to setup socket connection !", socket.error)
        else:
            print("Connect to dashboard server need use port 29999 !")
        data = self.socket_feedback.recv(1024)
        print('receive:', bytes.decode(data, 'utf-8'))
        if(self.socket_feedback != 0):
            self.socket_feedback.close()
            
    def MovL(self, x, y, z, a, b, c):
        """
        Coordinate system motion interface (linear motion mode)
        x: A number in the Cartesian coordinate system x
        y: A number in the Cartesian coordinate system y
        z: A number in the Cartesian coordinate system z
        a: A number in the Cartesian coordinate system a
        b: A number in the Cartesian coordinate system b
        c: a number in the Cartesian coordinate system c
        """
        if self.port == 30003:
            try:     
                self.socket_feedback = socket.socket() 
                self.socket_feedback.connect((self.ip, self.port))
                string = "MovL({:f},{:f},{:f},{:f},{:f},{:f})".format(x,y,z,a,b,c)
                print(string)
                self.socket_feedback.send(str.encode(string, 'utf-8'))
            except socket.error:
                print("Fail to setup socket connection !", socket.error)
        else:
            print("Connect to dashboard server need use port 29999 !")
        data = self.socket_feedback.recv(1024)
        print('receive:', bytes.decode(data, 'utf-8'))
        if(self.socket_feedback != 0):
            self.socket_feedback.close()
        
            
    def get_camera_data(self):
        color_img, depth_img = self.camera.get_data()
        return color_img, depth_img