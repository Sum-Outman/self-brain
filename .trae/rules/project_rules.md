1.使用简洁的黑白灰风格样式,浅色风格。
2.修改时将页面修改为全英文形式。
3.请保持对话语言为中文。
4.防止重复功能的文件出现。
5.所有审核文件全部自动接受。
6.增强和新增的文件，增强功能文件严格的在原有的文件上进行增强，新增的文件进行功能相同的合并处理删除多余文件，功能重复文件出现。
7.过程测试文件在测试完成后及时删除，不要产生无效的垃圾文件。
8.Self Brain AGI System Architecture:
├── A_management - Central Coordinator
├── B_language - Natural Language Processing
├── C_audio - Sound Analysis & Synthesis
├── D_image - Computer Vision
├── E_video - Video Understanding
├── F_spatial - 3D Spatial Awareness
├── G_sensor - IoT Data Processing
├── H_computer_control - System Automation
├── I_knowledge - Knowledge Graph
├── J_motion - Motion Control
└── K_programming - Code Generation & Understanding
9.Access Interface
After startup, visit: http://localhost:5000
10.API Endpoints
- `curl http://localhost:5015/api/health` - Health check
- `curl http://localhost:5015/api/stats` - System stats
- `curl http://localhost:5015/api/system/stats` - Detailed system stats
- `curl http://localhost:5015/api/models` - Available models list
11.Main Web Interface | 5000 
| A Management Model | 5001 
| B Language Model | 5002 
| C Audio Model | 5003 
| D Image Model | 5004 
| E Video Model | 5005 
| F Spatial Model | 5006 
| G Sensor Model | 5007 
| H Computer Control | 5008 
| I Knowledge Model | 5009 
| J Motion Model | 5010 
| K Programming Model | 5011 
| Manager Model API | 5015 
| Working Enhanced Chat | 5016 
12.项目根目录
├── advanced_system_launcher.py     # 系统启动器，管理所有组件的启动和停止
├── config\                         # 全局配置文件
│   ├── config_loader.py
│   ├── model_registry.json
│   ├── system_config.yaml
│   └── training_config.json
├── manager_model\                  # 中央协调器模块
├── web_interface\                  # Web界面模块
├── sub_models\                     # 功能子模块集合
├── training_manager\               # 训练管理模块
└── requirements.txt                # 项目依赖
13.中央协调器模块 (manager_model)
manager_model/               # 中央协调器模块
  ├── app.py                 # 管理模型API服务器
  ├── data_bus.py            # 数据总线，处理组件间通信
  │   ├── register_component() # 注册组件到数据总线
  ├── core_system_merged.py  # 合并的核心系统功能
  ├── model_registry.py      # 模型注册和管理
  ├── self_learning.py       # 自学习功能实现
  │   ├── _upgrade_architecture() # 架构升级功能
  ├── sub_models.py          # 子模型管理
  └── training_control.py    # 训练控制功能
14.功能子模块 (sub_models)
├── B_language\                      # 自然语言处理模块
│   ├── app.py
│   └── model.py
├── C_audio\                         # 音频处理模块
│   ├── app.py
│   └── model.py
├── D_image\                         # 图像处理模块
│   ├── app.py
│   └── model.py
├── E_video\                         # 视频处理模块
│   ├── app.py
│   └── model.py
├── F_spatial\                       # 空间感知模块
│   ├── app.py
│   └── model.py
├── G_sensor\                        # 传感器数据处理
│   ├── app.py
│   └── model.py
├── H_computer_control\              # 计算机控制模块
│   ├── app.py
│   └── model.py
├── I_knowledge\                     # 知识图谱模块
│   ├── app.py
│   ├── knowledge_base.py
│   └── knowledge_api.py
├── J_motion\                        # 运动控制模块
│   ├── app.py
│   └── model.py
└── K_programming\                   # 代码生成与理解
    ├── app.py
    └── code_analysis.py
15.训练管理模块 (training_manager)
training_manager/            # 训练管理模块
  ├── advanced_train_control.py # 高级训练控制面板
  │   ├── TrainingMode       # 训练模式枚举类
  │   ├── TrainingStatus     # 训练状态枚举类
  │   └── LanguageManager    # 语言管理器
  ├── enhanced_joint_trainer.py # 增强的联合训练器
  └── train_scheduler.py     # 训练任务调度器
16.页面结构
d:\shiyan/
├── web_interface/
│   ├── templates/
│   │   ├── base.html          # 主基础模板
│   │   ├── help.html          # 帮助文档页面
│   │   └── test.html          # 测试功能页面
│   ├── static/
│   │   ├── css/               # 样式表文件
│   │   ├── fonts/             # 字体文件
│   │   ├── images/            # 图像资源
│   │   ├── js/                # JavaScript脚本
│   │   └── webfonts/          # Web字体文件
│   ├── app.py                 # Web界面主应用
│   ├── app_fixed.py           # 修复版Web应用
│   └── config/                # Web界面配置
├── manager_model/
│   ├── templates/             # Manager模型页面模板
│   ├── static/
│   │   └── js/                # Manager模型JavaScript
│   └── app.py                 # Manager模型Web应用
├── sub_models/
│   ├── B_language/
│   │   ├── app.py             # 语言模型Web应用
│   │   └── config/            # 语言模型配置
│   ├── C_audio/
│   │   ├── app.py             # 音频模型Web应用
│   │   ├── templates/         # 音频处理界面模板
│   │   ├── static/            # 音频模型静态资源
│   │   └── config/            # 音频模型配置
│   ├── D_image/
│   │   ├── app.py             # 图像模型Web应用
│   │   └── api.py             # 图像模型API接口
│   ├── E_video/
│   │   ├── app.py             # 视频模型Web应用
│   │   └── api.py             # 视频模型API接口
│   ├── F_spatial/
│   │   ├── app.py             # 空间模型Web应用
│   │   └── api.py             # 空间模型API接口
│   ├── G_sensor/
│   │   ├── app.py             # 传感器模型Web应用
│   │   └── api.py             # 传感器模型API接口
│   ├── H_computer_control/
│   │   └──           # 计算机控制模型
│   ├── I_knowledge/
│   │   ├── app.py             # 知识模型Web应用
│   │   ├── static/            # 知识模型静态资源
│   │   └── backups/           # 知识库备份
│   ├── J_motion/
│   │   └── app.py             # 运动模型Web应用
│   └── K_programming/
│       └── app.py             # 编程模型Web应用
├── docs/
│   ├── 01Home.png             # 首页界面截图
│   ├── 02knowledge.png        # 知识库界面截图
│   ├── 03training.png         # 训练界面截图
│   ├── 04settings.png         # 设置界面截图
│   ├── 05help.png             # 帮助界面截图
│   ├── API.md                 # API文档
│   └── INSTALL.md             # 安装文档
├── icons/
│   ├── self_brain.ico         # 应用图标(ico格式)
│   ├── self_brain.svg         # 应用图标(SVG格式)
│   ├── self_brain_16.png      # 16x16像素图标
│   ├── self_brain_32.png      # 32x32像素图标
│   ├── self_brain_48.png      # 48x48像素图标
│   ├── self_brain_64.png      # 64x64像素图标
│   ├── self_brain_128.png     # 128x128像素图标
│   └── self_brain_256.png     # 256x256像素图标
├── config/
│   ├── system_config.yaml     # 系统主配置
│   ├── model_registry.json    # 模型注册表
│   └── training_config.json   # 训练配置
├── start_system.py            # 系统主启动脚本
├── start_system.bat           # Windows启动批处理
└── start_system.ps1           # PowerShell启动脚本