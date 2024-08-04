MetaBCI-USTChopin代码主要包括：

ME_pygame_BrainStim.py   数据采集
offline_test.py 离线模型训练和测试
UCTChopin_online.py  在线模型测试
ui_visualization.py  可视化界面
virtual_pian软件包  虚拟钢琴软件

可下拉选择训练算法，目前支持的算法有：LDA, SVM, Random Forest, MLP, Unimodal, Late_Fusion, Early_Fusion

带有简谱的桌面背景图片

使用说明，在设备准备好后，完成如下操作：
第一步：启动ui_visualization.py，进入功能界面；
第二步：在功能界面填写ip和port，连接设备，开始采集训练数据，自动执行ME_pygame_BrainStim.py文件，进入pygame设计的数据采集界面直至完成数据采集，回到功能界面；
第三步：在功能界面选择所需算法，确保数据路径正确，点击离线训练按钮，自动执行offline_test.py ，开始离线训练模型并完成测试，回到功能界面；
第四步：打开virtual_pian软件包，执行其中的piano_main.py文件，打开虚拟钢琴界面；在功能界面点击转到在线测试按钮，自动执行UCTChopin_online.py，被试看着带有琴谱的桌面和虚拟钢琴界面进行在线测试。



