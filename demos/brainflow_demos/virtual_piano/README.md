# Virtual_Piano

### 更新内容 After 2023-04-28：

注意！！！目前发现pygame库已更新，安装最新的pygame库会对键盘操作无响应（代码未进行更改）因此安装pygame库时应指定版本2.0.0

pip install pygame==2.0.0

增加了一些新的字体适配方法，global_settings.ini 中包含字体路径、字体位置偏移量的相关设置

删除了原先的TW CEN MT字体（显示存在一些问题）

增加从单独的midi文件读取踏板功能（把音符事件当作踏板）

顶端矩形宽度可更改

更改了和弦显示样式

### 演示视频：

目前该演示视频已过时（录制时无霓虹灯、透明瀑布流等效果），可进入主页查看最新视频

https://www.bilibili.com/video/BV1D8411x78X

### 主要功能：

1.连接midi键盘弹奏，实时显示瀑布流/五线谱，实时分析和弦，实时显示当前调性（按自然大调理解）

2.播放midi文件，实时显示瀑布流/五线谱（可自定义根音，对和弦做更准确的判断）

### 其他功能：

1.将自定义图片保存至backgrounds文件夹，运行时按G切换背景图片

2.按S选择开启/关闭背景前透明矩形（方便显示，默认五线谱模式开启，瀑布流模式关闭）

3.按T选择瀑布流、键盘、顶端矩形透明或不透明

4.按R调整瀑布流颜色模式（共有一种颜色、两种颜色、三种颜色、随机颜色四种模式）

3.按D取消调性自动识别（离调和弦密集、特殊调式识别不准确时用）

4.按C/B手动切换调性，按U将调性设为Unsettled；数字1-0及左右键对应1=C-1=B

5.按A重新开启调性自动识别

6.按M切换瀑布流/五线谱显示模式（仅midi键盘模式有效）

7.按P更改和弦显示位置或取消显示（仅瀑布流模式）

8.按E清除当前所有瀑布流

9.按H开启/关闭键盘白色灯光

9.按Q退出

### 运行方式：

安装pygame==2.0.0, mido等相关库，执行 piano_main.py 

### global_settings.ini 中的相关设置：

set_root_from_file		仅midi播放模式可用，是否从指定的midi文件读取根音（避免一些转位和弦的误识别）

get_sustain_from_file	仅midi播放模式可用，是否从指定的midi文件读取踏板信号（该midi文件用音符作为踏板信号，按下/松开任意音符为按下/松开踏板）

background_folder_path	背景图片目录（无需修改文件名）

font_path			字体文件目录（更改字体可能会显示错位，需要在ini中修改相关位置参数）

font_settings_path	字体设置目录（对应字体大小、偏移量等值）

midi_file_path		midi播放模式播放的midi文件（暂时只支持一个轨道）

root_file_path		当set_root_from_file = 1时，从指定midi文件读取根音

sustain_file_path		当get_sustain_from_file = 1时，从指定midi文件读取踏板信号

light_file_path		白色灯光png文件路径

key_light_open		默认开启键盘白色灯光

light_on_sustain		松开踏板，按下的键是否仍然亮灯

light_offset_white_x		白键灯光水平偏移量

light_offset_black_x		黑键灯光水平偏移量

light_offset_y		灯光垂直偏移量

[WhiteKeyColor]		白色琴键按下时显示的颜色RGB值

[BlackKeyColor]		黑色琴键按下时显示的颜色RGB值

waterfall_color_control	瀑布流颜色模式（0、1、2、3对应一种颜色、两种颜色、三种颜色、随机颜色四种模式）

WaterFallWidth		瀑布流宽度

WaterFallOffset		瀑布流偏移量

[WaterFallColorMain]	瀑布流轮廓&内容第一种颜色RGB值

[WaterFallColor2]		瀑布流轮廓&内容第二种颜色RGB值

[WaterFallColor3]		瀑布流轮廓&内容第三种颜色RGB值

[BlackColorDim]		黑键对应瀑布流相对于白键瀑布流颜色的减暗值（轮廓&内容）

[ColorBoundary]		瀑布流颜色分界线

[WhiteKeyOnSustain]	白色琴键松开，踏板踩下时显示的颜色RGB值

[BlackKeyOnSustain]		黑色琴键松开，踏板踩下时显示的颜色RGB值

[ChordTextColor]		和弦字体颜色RGB值

[NoteListTextColor]		音符列表字体颜色RGB值

[SustainTextColor]		踏板状态字体颜色RGB值

[KeyTextColor]		调号字体颜色RGB值

[SpeedTextColor]		速度字体颜色RGB值

[TopSquareColor]		顶端矩形颜色RGB值

[TopSquareWidth]		顶端矩形宽度

[TransScreenColor]		背景前透明矩形颜色RGB值

transparent_or_not		是否默认瀑布流、顶端矩形、键盘透明

trans_screen_opacity	背景前透明矩形透明度

top_square_opacity		顶端矩形透明度

waterfall_opacity		瀑布流透明度

piano_key_opacity		键盘透明度

[GlobalResolution]		全局分辨率（默认1920x1080，更改分辨率为裁剪而非缩放，需要修改offset）

[BackGroundOffset]		背景图片位移值

piano_key_offset		钢琴键水平位移值

[MusicScoreOffset]		五线谱位移值

time_delta		midi播放速度

root_delta		当set_root_from_file = 1时，该值为待播放midi文件与根音midi文件时间差

sustain_delta	当get_sustain_from_file = 1时，该值为待播放midi文件与踏板midi文件时间差

flash_neon_prepare		是否提前载入动态霓虹灯效果

flash_neon_pic_path		动态霓虹灯效果图片路径（逐帧）

flash_neon_gap_time	霓虹灯每一帧间隔

### fonts/fontxx_settings.ini 中的相关设置：

font_size_1		顶端字体大小

font_size_2		顶端延音踏板作用等标记的字体大小

font_size_3		五线谱模式下和弦字体大小

font_size_4		五线谱模式下对应的音名标记字体大小

font_size_5		单独瀑布流模式下对应的和弦字体大小

font_size_6		单独瀑布流模式下对应的音名标记字体大小

[SustainLabel]	延音踏板字体偏移量

[SustainState]	延音踏板作用标记字体偏移量

[MajorKey]		调性字体偏移量

[SpeedLabel]	音符速度字体偏移量

[Tonicization]	离调标记字体偏移量，对应到每个调式

chord_text_score_offset_y		五线谱模式下和弦标记纵坐标偏移量

bass_treble_text_offset_y			五线谱模式下高/低音标记纵坐标偏移量

note_list_text_offset_y = 566		五线谱模式下音名标记纵坐标偏移量

waterfall_chord_mode_1(2,3)_x		单独瀑布流模式下和弦标记横坐标偏移量（默认使用位置1，按P切换2，3）

waterfall_chord_mode_1(2,3)_y		单独瀑布流模式下和弦标记纵坐标偏移量

waterfall_bass_treble_mode_1(2,3)_x		单独瀑布流模式下高/低音标记横坐标偏移量

waterfall_bass_treble_mode_1(2,3)_y		单独瀑布流模式下高/低音标记纵坐标偏移量

waterfall_note_list_mode_1(2,3)_x			单独瀑布流模式下音名标记横坐标偏移量

waterfall_note_list_mode_1(2,3)_y			单独瀑布流模式下音名标记纵坐标偏移量

### 存在问题：

1.部分和弦识别不准确，斜杠和弦/非三度叠置和弦/过于复杂的和弦难以识别

2.调性识别功能有待进一步增强，目前仅能在没有离调的情况下识别出自然大调

3.不支持低分辨率

4.存在一些不规范的记谱（非错误，不严谨但是不影响识谱）

5.midi仅支持单轨，不支持变速MIDI播放

6.无法直观调节BPM（可通过global_settings中的time_delta更改）

7.MIDI播放时声音略有错位（有些音偏前有些音偏后，midi音符密集时这种情况很显著，推荐宿主单独导出音频，之后用Pr等软件将音轨与视频轨对齐导出）


这里是我添加的
test_client.py是客户端用来发数字的，发1-5,模拟通信过程
piano_main.py是主程序，接收数字并播放音乐
