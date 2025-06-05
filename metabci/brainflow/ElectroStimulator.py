# -*- coding: utf-8 -*-
"""
Control the electrical stimulator.Implements control of electrical stimulation
parameters via serial communication, including channel selection, waveform
parameter configuration, parameter locking, and therapy start/stop operations.
"""
import serial
import struct
import time
from enum import IntEnum
from serial.serialutil import SerialException
from typing import Set, Dict


class ElectroStimulator:
    """
    Electrical stimulator controller class for
    multichannel parameter configuration and pulse therapy control.

    author: Haixia Lei <leihaixia@tju.edu.cn>

    Created on: 2024-04-08

    update log:
        None

    Parameters
    ----------
    port : str
        Serial port device path.
    baudrate : int
        Communication baud rate (default 115200).

    Attributes
    ----------
    _is_locked : bool
        Parameter locking status, prohibits parameter modification when locked.
    _selected_channels : Set[int]
        Currently enabled therapy channels (0-12).

    Raises
    ----------
    RuntimeError
        Raised when serial connection fails
        or state machine rules are violated.
    ValueError
        Raised when channel number or parameter values exceed valid ranges.

    Note
    ----------
    1. Parameters must be modified before locking.
       Only start/stop operations are allowed after locking.
    2. The device connection must be reinitialized after calling close().
    3. For channel-specific parameters, ensure the channel is selected first.

    """
    class _Param(IntEnum):
        """Parameter address"""
        channel_select = 0x10      # 通道选择
        rise_time = 0x11           # 斜升时间 (ms)
        stable_time = 0x12         # 稳定时间 (ms)
        descent_time = 0x13        # 斜降时间 (ms)
        current_positive = 0x18    # 正脉冲峰值 (mA)
        pulse_positive = 0x19      # 正脉冲宽度 (us)
        current_negative = 0x1A    # 负脉冲峰值 (mA)
        pulse_negative = 0x1B      # 负脉冲宽度 (us)
        frequency = 0x1D           # 频率 (Hz)
        small_cycles = 0x20        # 小周期次数
        big_cycles = 0x21          # 大周期次数
        small_interval = 0x22      # 小周期间隔
        big_interval = 0x23        # 大周期间隔
        lock = 0xF1                # 参数锁定
        start = 0xFA               # 开始治疗
        stop = 0xFC                # 停止治疗

    def __init__(self, port, baudrate=115200):
        self.ser = None
        self._is_locked = False  # 参数锁定状态
        self._selected_channels: Set[int] = set()  # 存储已选通道

        # 初始化串口连接
        try:
            self.ser = serial.Serial(
                port=port,
                baudrate=baudrate,
                bytesize=8,
                parity='N',
                stopbits=1,
                timeout=1
            )
            print(f"Connected to {port}")
        except SerialException as e:
            raise RuntimeError(f"Failed to open serial port: {e}") from None

    def _validate_channel(self, channel):
        """Verify channel number validity (0-12)"""
        if not 0 <= channel <= 12:
            raise ValueError(f"Invalid channel {channel}, must be 0-12")

    def select_channel(self, channel: int, enable: bool = True):
        """Select or deselect a therapy channel
        (must be called before locking).

        Parameters
        ----------
        channel : int
            Target channel number (0-12).
        enable : bool
            Enable/disable the channel (default True).
        """
        if self._is_locked:
            raise RuntimeError("Channel selection cannot be modified")

        self._validate_channel(channel)

        # 设置通道使能位
        # 数据格式：0x0001 表示启用，0x0000 表示禁用
        value = 0x0001 if enable else 0x0000
        self.set_parameter(channel, self._Param.channel_select, value)

        # 更新已选通道集合
        if enable:
            self._selected_channels.add(channel)
        elif channel in self._selected_channels:
            self._selected_channels.remove(channel)

        print(f"通道 {channel} {'已启用' if enable else '已禁用'}")

    def disable_channel(self, channel: int):
        """Disable channel."""
        self.select_channel(channel, enable=False)

    def set_channel_parameters(self, channel: int, params: Dict[_Param, int]):
        """Set channel parameters in batches."""
        for param, value in params.items():
            self.set_parameter(channel, param, value)

    def _build_frame(self, channel, param_addr, data_value):
        """Build protocol data frames."""
        # 验证通道号
        self._validate_channel(channel)

        # 数据区转换（16位高位在前）
        try:
            data_bytes = struct.pack('>H', data_value)
        except struct.error:
            raise ValueError(f"Invalid data value:{data_value}") from None

        # 计算总长度：n*2 + 4（n=1）
        total_length = struct.pack('B', 1*2 + 4)

        # 组合数据帧
        return (b'\x5A\xA5' +         # 帧头
                total_length +        # 总长度
                b'\x93' +             # 写命令
                struct.pack('B', channel) +
                struct.pack('B', param_addr) +
                b'\x01' +            # 数据长度
                data_bytes)

    def set_parameter(self, channel, param_addr, value):
        """Set stimulation parameters for a channel.

        Parameters
        ----------
        channel : int
            Target channel number (0-12). Use 0 for global commands.
        param_addr : ElectroStimulator._Param
            Register address (e.g., _Param.frequency).
        value : int
            Parameter value (0-65535). Specific ranges depend on the parameter.
        """
        try:
            # 参数锁定后禁止修改参数（全局命令除外）
            if self._is_locked and channel != 0:
                raise RuntimeError("Cannot modify parameters after locking")

            frame = self._build_frame(channel, param_addr, value)
            self.ser.write(frame)
            print(f"Set Channel {channel}: Addr 0x{param_addr:02X} = {value}")

            # 添加操作间隔防止设备过载
            time.sleep(0.1)

        except SerialException as e:
            raise RuntimeError(f"Serial communication failed: {e}") from None

    def lock_parameters(self):
        """Lock all parameters to prevent accidental modifications.
        At least one channel must be selected."""
        if not self._selected_channels:
            raise RuntimeError("At least one channel must be selected")

        if self._is_locked:
            print("Parameters already locked")
            return

        self.set_parameter(0, self._Param.lock, 0x0001)
        self._is_locked = True
        print("Parameters Locked")

        # 等待设备确认锁定
        time.sleep(0.5)

    def run_stimulation(self, duration: int):
        """Start therapy and automatically stop after a specified duration.

        Parameters
        ----------
        duration : int
            Therapy duration.
        """
        if not self._is_locked:
            raise RuntimeError("Must lock parameters before starting")
        if not self._selected_channels:
            raise RuntimeError("There is no effective treatment channel")
        self.set_parameter(0, self._Param.start, 0x0001)
        print(f"治疗已启动，激活通道: {sorted(self._selected_channels)}")

        time.sleep(duration)
        self.set_parameter(0, self._Param.stop, 0x0001)
        print("治疗结束")
        self._is_locked = False  # 停止后自动解锁

    def close(self):
        """Safely terminate the serial connection."""
        if self.ser and self.ser.is_open:
            self.ser.close()
            print("Serial port closed")
