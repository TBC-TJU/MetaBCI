# AssistBCI——通用脑机接口辅助系统：基于 MetaBCI 的高效二次开发框架与全面个性化应用解决方案
澳门大学

李浩博, 李思源, 徐启昊, 朱峻毅

主要联系人. Tel.: (+86) 13581975632; email: li.haobo@connect.um.edu.mo

## 摘要
MetaBCI提供新范式、新算法、新设备的实验平台，但缺乏高效的技术部署方案。
为解决这一问题，我们设计了基于MetaBCI的脑机接口应用开发平台—AssistBCI，
允许神经肌肉疾病患者根据使用场景独立设计视觉刺激布局，如SSVEP刺激位置
和大小，并支持自定义辅助操作，如鼠标操作。
  
![微信图片_20240804210208](https://github.com/user-attachments/assets/9302d6f1-28be-4762-b716-311a7cf1da9e)
  
AssistBCI兼容MetaBCI平台设备、算法和范式。对于在MetaBCI验证的技术，无需
额外开发，可通过AssistBCI直接适配。患者可快速应用最新BCI技术，提高使用
乐趣，实现独立的电脑操作。
  
<img width="416" alt="微信图片_20240804210214" src="https://github.com/user-attachments/assets/1dde5b67-bfd3-4a9d-9e0b-aa1b4040e90c">
  
对于新功能需求，如青光眼检测，开发者可在AssistBCI中进行开发。AssistBCI提
供便捷开发工具，允许调用MetaBCI中的算法、设备和数据库，并提供可视化界
面快速创建刺激界面。系统主界面基于pyglet开发，功能实现支持pyglet和QT双
模式，刺激模块仅支持MetaBCI和psychopy软件包。
  
<img width="410" alt="微信图片_20240804210202" src="https://github.com/user-attachments/assets/a6135eef-e6d5-4671-8a29-14dbb5fde770">
  
**关键词: 计算机辅助控制系统，二次开发平台, SSVEP, ms-eCCA, LinkMe, BlueBCIMetaBCI 创新应用开发赛项**
  
## 项目功能点明细表

 序号 | 功能点描述  | 量化指标
 ---- | ----- | ------  
 1  | 新增 Brainstim 中 SSVEP 范式刺激参数可视化设计工具 | 无 
 2  | 新增基于 MetaBCI 的二次开发框架 | 无  
 3  | 新增 Brainstim 中 SSVEP 指令尺寸独立设计 | 无 
 4  | 新增 Brainda 中 SSVEP 范式识别算法 | 2 种  
 5  | 新增 Brainflow 中采集数据实时保存方法 | 1 种 
 6  | 新增 Brainda 中采集数据便捷调用 | 1 种  
 7  | 新增 Brainflow 中设备支持 | 3 种 
 8  | 优化 Brainflow 中下载数据集存放位置 | 无  
 9  | 优化 Brainflow 中 worker 中 Brainda 算法的使用 | 无 


# 常见问题：

## 黑屏，显示问题：

### 可能原因：

psychopy版本不为1.5.27 

pyglet版本不为2022.1.4

多个显示器
