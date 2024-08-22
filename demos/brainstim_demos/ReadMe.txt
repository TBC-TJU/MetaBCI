MetaBCI-USTChopin代码主要包括：

USTChopin_metabci_stim.py   手势数据采集
USTChopin_metabci_stim_Atten.py   专注度数据采集

USTChopin_Offline.py 离线模型训练和测试
USTChopin_Offline_Atten.py 离线模型训练和测试
UCTChopin_Online.py  在线模型测试

virtual_pian软件包  虚拟钢琴软件
将以上文件与文件夹metabci和图片文件夹Stim_images放置于同一路径下.


使用说明，在设备准备好后，完成如下操作：
首先，使用数据采集代码采集数据；
第二步，使用离线离线模型训练和测试文件获取训练模型；
第三步，打开virtual_pian文件夹，运行piano_main.py打开虚拟钢琴，接着使用UCTChopin_Online.py进行在线测试
