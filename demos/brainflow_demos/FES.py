from metabci.brainflow.ElectroStimulator import ElectroStimulator
stim = ElectroStimulator('COM1')  # 串口号

# 启用通道1并设置参数
stim.select_channel(1)  # 启用通道1
stim.set_channel_parameters(1, {
            ElectroStimulator._Param.current_positive: 2,
            ElectroStimulator._Param.current_negative: 2,
            ElectroStimulator._Param.pulse_positive: 250,
            ElectroStimulator._Param.pulse_negative: 250,
            ElectroStimulator._Param.frequency: 50,
            ElectroStimulator._Param.rise_time: 500,
            ElectroStimulator._Param.stable_time: 3000,
            ElectroStimulator._Param.descent_time: 500
        })

# 锁定参数并启动
stim.lock_parameters()
stim.run_stimulation(duration=10)

if 'stim' in locals():
    stim.close()
