from psychopy import monitors
import numpy as np
from metabci.brainstim.paradigm import (
    SSVEP,
    AVEP,
    SSAVEP,
    paradigm,
    pix2height,
    code_sequence_generate,
)
from metabci.brainstim.framework import Experiment
from psychopy.tools.monitorunittools import deg2pix

def SSVEP_WIN():
    mon = monitors.Monitor(
        name="primary_monitor",
        width=59.6,
        distance=60,  # width 显示器尺寸cm; distance 受试者与显示器间的距离
        verbose=False,
    )
    # mon.setSizePix([1920, 1080])  # 显示器的分辨率
    # mon.save()
    bg_color_warm = np.array([0, 0, 0])
    win_size = np.array([1920, 1080])
    # # esc/q退出开始选择界面
    ex = Experiment(
        monitor=mon,
        bg_color_warm=bg_color_warm,  # 范式选择界面背景颜色[-1~1,-1~1,-1~1]
        screen_id=0,
        win_size=win_size,  # 范式边框大小(像素表示)，默认[1920,1080]
        is_fullscr=False,  # True全窗口,此时win_size参数默认屏幕分辨率
        record_frames=False,
        disable_gc=False,
        process_priority="normal",
        use_fbo=False,
    )
    win = ex.get_window()

    # q退出范式界面
    """
    SSVEP
    """
    n_elements, rows, columns = 20, 4, 5  # n_elements 指令数量;  rows 行;  columns 列
    stim_length, stim_width = 200, 200  # ssvep单指令的尺寸
    stim_color, tex_color = [1, 1, 1], [1, 1, 1]  # 指令的颜色，文字的颜色
    fps = 240  # 屏幕刷新率
    stim_time = 2  # 刺激时长
    stim_opacities = 1  # 刺激对比度
    freqs = np.arange(8, 16, 0.4)  # 指令的频率
    phases = np.array([i * 0.35 % 2 for i in range(n_elements)])  # 指令的相位

    basic_ssvep = SSVEP(win=win)

    basic_ssvep.config_pos(
        n_elements=n_elements,
        rows=rows,
        columns=columns,
        stim_length=stim_length,
        stim_width=stim_width,
    )
    basic_ssvep.config_text(tex_color=tex_color)
    basic_ssvep.config_color(
        refresh_rate=fps,
        stim_time=stim_time,
        stimtype="sinusoid",
        stim_color=stim_color,
        stim_opacities=stim_opacities,
        freqs=freqs,
        phases=phases,
    )
    basic_ssvep.config_index()
    basic_ssvep.config_response()

    bg_color = np.array([0.3, 0.3, 0.3])  # 背景颜色
    display_time = 1  # 范式开始1s的warm时长
    index_time = 1  # 提示时长，转移视线
    rest_time = 0.5  # 提示后的休息时长
    response_time = 1  # 在线反馈
    port_addr = "COM8"  #  0xdefc                                  # 采集主机端口
    port_addr = None  #  0xdefc
    nrep = 2  # block数目
    lsl_source_id = "meta_online_worker"  # None                 # source id
    online = False  # True                                       # 在线实验的标志
    ex.register_paradigm(
        "basic SSVEP",
        paradigm,
        VSObject=basic_ssvep,
        bg_color=bg_color,
        display_time=display_time,
        index_time=index_time,
        rest_time=rest_time,
        response_time=response_time,
        port_addr=port_addr,
        nrep=nrep,
        pdim="ssvep",
        lsl_source_id=lsl_source_id,
        online=online,
    )


    """
    AVEP
    """
    n_elements, rows, columns = 20, 5, 4  # n_elements 指令数量;  rows 行;  columns 列
    stim_length, stim_width = 3, 3  # avep刺激点的尺寸
    tex_height = 25  # avep指令的大小
    stim_color, tex_color = [0.7, 0.7, 0.7], [1, 1, 1]  # 指令的颜色，文字的颜色
    fps = 60  # 屏幕刷新率
    stim_time = 1  # 刺激时长
    stim_opacities = 1  # 刺激对比度
    freqs = 4  # 指令的频率
    # phases = np.array([i * 0.35 % 2 for i in range(n_elements)])  # 指令的相位
    stim_num = 2
    avep = AVEP(win=win, dot_shape="cluster")
    sequence = [avep.num2bin_ary(i, n_elements) for i in range(n_elements)]
    # sequence = [[1,2,3,4] for i in range(n_elements)]
    if len(sequence) != n_elements:
        raise Exception("Incorrect spatial code amount!")
    avep.tex_height = tex_height
    avep.config_pos(
        n_elements=n_elements,
        rows=rows,
        columns=columns,
        stim_length=stim_length,
        stim_width=stim_width,
    )
    avep.config_color(
        refresh_rate=fps,
        stim_time=stim_time,
        stimtype="sinusoid",
        stim_color=stim_color,
        sequence=sequence,
        stim_opacities=stim_opacities,
        freqs=np.ones((n_elements)) * freqs,
        stim_num=stim_num,
    )

    avep.config_text(symbol_height=tex_height, tex_color=tex_color)
    avep.config_index(index_height=40)
    avep.config_response()

    bg_color = np.array([-1, -1, -1])  # 背景颜色
    display_time = 1  # 范式开始1s的warm时长
    index_time = 0.5  # 提示时长，转移视线
    rest_time = 0.5  # 提示后的休息时长
    response_time = 1  # 在线反馈
    port_addr = None  # 0xdefc                                  # 采集主机端口
    nrep = 1  # block数目
    lsl_source_id = "meta_online_worker"  # None                 # source id
    online = False  # True                                       # 在线实验的标志
    ex.register_paradigm(
        "avep",
        paradigm,
        VSObject=avep,
        bg_color=bg_color,
        display_time=display_time,
        index_time=index_time,
        rest_time=rest_time,
        response_time=response_time,
        port_addr=port_addr,
        nrep=nrep,
        pdim="avep",
        lsl_source_id=lsl_source_id,
        online=online,
    )


    """
    SSaVEP
    """
    n_elements, rows, columns = 20, 4, 5
    n_members = 8
    stim_length, stim_width = 150, 150
    stim_color, tex_color = [1, 1, 1], [1, 1, 1]
    fps = 240
    stim_time_member = 0.5
    stim_opacities = [1]
    freqs = np.array(
        [4, 8, 12, 16, 20, 4, 8, 12, 16, 20, 4, 8, 12, 16, 20, 4, 8, 12, 16, 20]
    )
    phases = np.zeros((n_elements, 1))
    basic_code = [[0, 1], [2, 3], [4, 5], [6, 7]]
    code_sequences = [
        [0, 1, 2, 3],
        [0, 1, 2, 3],
        [0, 1, 2, 3],
        [0, 1, 2, 3],
        [0, 1, 2, 3],
        [1, 2, 3, 0],
        [1, 2, 3, 0],
        [1, 2, 3, 0],
        [1, 2, 3, 0],
        [1, 2, 3, 0],
        [2, 3, 0, 1],
        [2, 3, 0, 1],
        [2, 3, 0, 1],
        [2, 3, 0, 1],
        [2, 3, 0, 1],
        [3, 0, 1, 2],
        [3, 0, 1, 2],
        [3, 0, 1, 2],
        [3, 0, 1, 2],
        [3, 0, 1, 2],
        [3, 2, 1, 0],
        [3, 2, 1, 0],
        [3, 2, 1, 0],
        [3, 2, 1, 0],
        [3, 2, 1, 0],
    ]

    code = code_sequence_generate(basic_code, code_sequences)
    n_sequence = np.shape(code)[1]
    angles = np.zeros(n_elements)
    outter_deg = 4
    inner_deg = 1.5
    radius = deg2pix(outter_deg, mon) / win_size[1] * 0.7
    basic_ssavep = SSAVEP(win=win, n_elements=n_elements, n_members=n_members)
    basic_ssavep.config_pos(
        n_elements=n_elements,
        rows=rows,
        columns=columns,
        stim_length=stim_length,
        stim_width=stim_width,
    )
    basic_ssavep.stim_width = pix2height(win_size, basic_ssavep.stim_width)
    basic_ssavep.config_member_pos(
        win,
        radius=radius,
        angles=angles,
        outter_deg=outter_deg,
        inner_deg=inner_deg,
        tex_pix=256,
        sep_line_pix=16,
    )
    basic_ssavep.config_text(tex_color=tex_color, unit="height", symbol_height=0.03)
    basic_ssavep.config_stim(
        win,
        sizes=[[basic_ssavep.radius * 0.9, basic_ssavep.radius * 0.9]],
        member_degree=None,
        stim_color=stim_color,
        stim_opacities=stim_opacities,
    )
    # win.close()

    basic_ssavep.config_flash_array(
        refresh_rate=fps,
        freqs=freqs,
        phases=phases,
        codes=code,
        stim_time_member=stim_time_member,
        stimtype="sinusoid",
        stim_color=stim_color,
    )
    basic_ssavep.config_color(
        win,
        refresh_rate=fps,
        freqs=freqs,
        phases=phases,
        codes=code,
        stim_time_member=stim_time_member,
        stimtype="sinusoid",
        stim_color=stim_color,
        sizes=[[basic_ssavep.radius * 0.9, basic_ssavep.radius * 0.9]],
    )
    basic_ssavep.config_ring(
        win,
        sizes=[[basic_ssavep.radius * 2.15, basic_ssavep.radius * 2.15]],
        ring_colors=[2 * np.array([160, 160, 160]) / 255 - 1],
        opacities=stim_opacities,
    )
    basic_ssavep.config_target(
        win,
        sizes=[[basic_ssavep.radius * 0.2, basic_ssavep.radius * 0.2]],
        target_colors=[1, 1, 0],
        opacities=stim_opacities,
    )
    basic_ssavep.config_index(index_height=0.08, units="height")
    basic_ssavep.config_response()

    bg_color = np.array([-1, -1, -1])  # 背景颜色
    display_time = 0.5  # 范式开始1s的warm时长
    index_time = 1  # 提示时长，转移视线
    rest_time = 0.5  # 提示后的休息时长
    response_time = 1  # 在线反馈
    # port_addr = 'COM8'  #  0xdefc                                  # 采集主机端口
    port_addr = None
    nrep = 2  # block数目
    lsl_source_id = "meta_online_worker"  # None                 # source id
    online = False  # True                                       # 在线实验的标志
    ex.register_paradigm(
        "basic SSaVEP",
        paradigm,
        VSObject=basic_ssavep,
        bg_color=bg_color,
        display_time=display_time,
        index_time=index_time,
        rest_time=rest_time,
        response_time=response_time,
        port_addr=port_addr,
        nrep=nrep,
        pdim="ssavep",
        lsl_source_id=lsl_source_id,
        online=online,
    )

    ex.run()

