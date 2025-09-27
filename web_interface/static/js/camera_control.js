// Camera Control Module with Multi-Camera and Stereo Vision Support
// 使用立即执行函数表达式(IIFE)来避免全局命名空间污染
(function(window) {
    // 检查 CameraControl 是否已存在，如果存在则不重新创建
    if (window.CameraControl) {
        console.log('CameraControl is already initialized');
        return;
    }
    
    // 使用对象字面量创建CameraControl
    const CameraControl = {
        // API endpoints
        API_ENDPOINTS: {
            INPUTS: '/api/camera/inputs',
            START: '/api/camera/start/',
            STOP: '/api/camera/stop/',
            SNAPSHOT: '/api/camera/take-snapshot/',
            GET_SETTINGS: '/api/camera/settings/',
            UPDATE_SETTINGS: '/api/camera/settings/',
            // 立体视觉相关API端点
            STEREO_PAIRS: '/api/camera/stereo-pairs',
            DEPTH_DATA: '/api/camera/depth-data/',
            ENABLE_STEREO: '/api/camera/enable-stereo/',
            DISABLE_STEREO: '/api/camera/disable-stereo/'
        },
        
        // 存储活动相机流的状态
        activeStreams: {},
        
        // 存储立体视觉对配置
        stereoPairs: [],
        
        // 活动的立体视觉处理会话
        activeStereoSessions: {},
        
        // 初始化相机控制
        init: function() {
            console.log('Camera Control Module with Multi-Camera and Stereo Vision initialized');
            
            // 测试API端点连接性
            this.testCameraAPI();
            
            // 加载立体视觉对配置
            this.loadStereoPairs();
        },
        
        // 加载立体视觉对配置
        loadStereoPairs: async function() {
            try {
                const response = await fetch(this.API_ENDPOINTS.STEREO_PAIRS);
                const data = await response.json();
                
                if (data.status === 'success') {
                    this.stereoPairs = data.stereo_pairs || [];
                    console.log(`Loaded ${this.stereoPairs.length} stereo camera pairs`);
                }
            } catch (error) {
                console.error('Error loading stereo pairs:', error);
            }
        },
        
        // 测试相机API端点
        testCameraAPI: async function() {
            try {
                const response = await fetch(this.API_ENDPOINTS.INPUTS);
                const data = await response.json();
                console.log('Camera API test response:', data);
            } catch (error) {
                console.error('Camera API test failed:', error);
            }
        },
        
        // 测试立体视觉API
        testStereoAPI: async function() {
            try {
                const response = await fetch(this.API_ENDPOINTS.STEREO_PAIRS);
                const data = await response.json();
                console.log('Stereo Vision API test response:', data);
                return { success: true, data: data };
            } catch (error) {
                console.error('Stereo Vision API test failed:', error);
                return { success: false, error: error.message };
            }
        },
        
        // 测试函数以验证模块是否已加载
        testAPI: async function() {
            try {
                console.log('Testing Camera API connection...');
                const response = await fetch(this.API_ENDPOINTS.INPUTS);
                const data = await response.json();
                console.log('Camera API test successful:', data);
                return { success: true, data: data };
            } catch (error) {
                console.error('Camera API test failed:', error);
                return { success: false, error: error.message };
            }
        },
        
        // 获取所有活动的相机流
        getActiveStreams: function() {
            return this.activeStreams;
        },
        
        // 检查相机是否活动
        isCameraActive: function(cameraId) {
            return this.activeStreams.hasOwnProperty(cameraId);
        },
        
        // 从API获取活动相机输入
        getActiveCameraInputs: async function() {
            try {
                const response = await fetch(this.API_ENDPOINTS.INPUTS);
                if (!response.ok) {
                    throw new Error(`HTTP error! Status: ${response.status}`);
                }
                const data = await response.json();
                
                if (data.status === 'success') {
                    console.log(`Successfully retrieved ${data.camera_count} camera inputs`);
                } else {
                    console.error('Failed to get camera inputs:', data.message);
                }
                
                return data;
            } catch (error) {
                console.error('Error retrieving camera inputs:', error);
                return { status: 'error', message: error.message };
            }
        },
        
        // 通过API启动相机
        startCamera: async function(cameraId) {
            try {
                const response = await fetch(`${this.API_ENDPOINTS.START}${cameraId}`, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({})
                });
                
                if (!response.ok) {
                    throw new Error(`HTTP error! Status: ${response.status}`);
                }
                
                const data = await response.json();
                console.log(`Camera ${cameraId} started successfully`, data);
                
                // 更新活动流状态
                if (data.status === 'success') {
                    this.activeStreams[cameraId] = {
                        started: true,
                        lastUpdated: new Date()
                    };
                }
                
                return data;
            } catch (error) {
                console.error(`Error starting camera ${cameraId}:`, error);
                throw error;
            }
        },
        
        // 启动多个相机
        startMultipleCameras: async function(cameraIds) {
            try {
                const promises = cameraIds.map(id => this.startCamera(id));
                const results = await Promise.all(promises);
                console.log(`Started ${cameraIds.length} cameras`);
                return results;
            } catch (error) {
                console.error('Error starting multiple cameras:', error);
                throw error;
            }
        },
        
        // 通过API停止相机
        stopCamera: async function(cameraId) {
            try {
                const response = await fetch(`${this.API_ENDPOINTS.STOP}${cameraId}`, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    }
                });
                
                if (!response.ok) {
                    throw new Error(`HTTP error! Status: ${response.status}`);
                }
                
                const data = await response.json();
                console.log(`Camera ${cameraId} stopped successfully`, data);
                
                // 更新活动流状态
                if (data.status === 'success' && this.activeStreams.hasOwnProperty(cameraId)) {
                    delete this.activeStreams[cameraId];
                }
                
                return data;
            } catch (error) {
                console.error(`Error stopping camera ${cameraId}:`, error);
                throw error;
            }
        },
        
        // 停止所有活动的相机
        stopAllCameras: async function() {
            try {
                const cameraIds = Object.keys(this.activeStreams);
                const promises = cameraIds.map(id => this.stopCamera(id));
                await Promise.all(promises);
                console.log('All cameras stopped');
                this.activeStreams = {};
                return { status: 'success', message: 'All cameras stopped' };
            } catch (error) {
                console.error('Error stopping all cameras:', error);
                return { status: 'error', message: error.message };
            }
        },
        
        // 通过API拍摄快照
        takeCameraSnapshot: async function(cameraId) {
            try {
                const response = await fetch(`${this.API_ENDPOINTS.SNAPSHOT}${cameraId}`, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    }
                });
                
                if (!response.ok) {
                    throw new Error(`HTTP error! Status: ${response.status}`);
                }
                
                const data = await response.json();
                console.log(`Snapshot taken from camera ${cameraId}`, data);
                return data;
            } catch (error) {
                console.error(`Error taking snapshot from camera ${cameraId}:`, error);
                throw error;
            }
        },
        
        // 通过API获取相机设置
        getCameraSettings: async function(cameraId) {
            try {
                const response = await fetch(`${this.API_ENDPOINTS.GET_SETTINGS}${cameraId}`, {
                    method: 'GET',
                    headers: {
                        'Content-Type': 'application/json'
                    }
                });
                
                if (!response.ok) {
                    throw new Error(`HTTP error! Status: ${response.status}`);
                }
                
                const data = await response.json();
                console.log(`Settings retrieved for camera ${cameraId}`, data);
                return data;
            } catch (error) {
                console.error(`Error retrieving settings for camera ${cameraId}:`, error);
                throw error;
            }
        },
        
        // 通过API更新相机设置
        updateCameraSettings: async function(cameraId, settings) {
            try {
                const response = await fetch(`${this.API_ENDPOINTS.UPDATE_SETTINGS}${cameraId}`, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify(settings)
                });
                
                if (!response.ok) {
                    throw new Error(`HTTP error! Status: ${response.status}`);
                }
                
                const data = await response.json();
                console.log(`Settings updated for camera ${cameraId}`, data);
                return data;
            } catch (error) {
                console.error(`Error updating settings for camera ${cameraId}:`, error);
                throw error;
            }
        },
        
        // 使用navigator.mediaDevices.enumerateDevices获取可用相机
        getAvailableCameras: async function() {
            try {
                if (!navigator.mediaDevices || !navigator.mediaDevices.enumerateDevices) {
                    console.log('enumerateDevices not supported');
                    return [];
                }
                
                const devices = await navigator.mediaDevices.enumerateDevices();
                const cameras = devices.filter(device => device.kind === 'videoinput');
                
                console.log(`Found ${cameras.length} cameras`);
                return cameras;
            } catch (error) {
                console.error('Error enumerating cameras:', error);
                return [];
            }
        },
        
        // 启动相机流
        startCameraStream: async function(deviceId, constraints = {}) {
            try {
                // 默认约束（如果未提供）
                const defaultConstraints = {
                    video: {
                        width: { ideal: 1280 },
                        height: { ideal: 720 },
                        frameRate: { ideal: 30 }
                    }
                };
                
                // 如果提供了设备ID，添加到约束中
                if (deviceId) {
                    defaultConstraints.video.deviceId = deviceId;
                }
                
                const mergedConstraints = { ...defaultConstraints, ...constraints };
                const stream = await navigator.mediaDevices.getUserMedia(mergedConstraints);
                
                console.log(`Camera stream for device ${deviceId || 'default'} started successfully`);
                
                // 存储流引用
                if (deviceId) {
                    this.activeStreams[deviceId] = {
                        stream: stream,
                        started: true,
                        lastUpdated: new Date()
                    };
                }
                
                return stream;
            } catch (error) {
                console.error('Error starting camera stream:', error);
                
                // 处理常见错误
                if (error.name === 'NotAllowedError') {
                    alert('Camera access was denied. Please allow camera access in your browser settings.');
                } else if (error.name === 'NotFoundError') {
                    alert('No camera found on this device.');
                } else if (error.name === 'NotReadableError') {
                    alert('Camera is already in use by another application.');
                }
                
                throw error;
            }
        },
        
        // 停止相机流
        stopCameraStream: function(deviceId) {
            // 如果提供了设备ID，停止特定的流
            if (deviceId && this.activeStreams[deviceId] && this.activeStreams[deviceId].stream) {
                const stream = this.activeStreams[deviceId].stream;
                if (stream.getTracks) {
                    stream.getTracks().forEach(track => track.stop());
                }
                delete this.activeStreams[deviceId];
                console.log(`Camera stream for device ${deviceId} stopped`);
            } else if (arguments.length === 0) {
                // 如果没有提供设备ID，停止所有流
                this.stopAllCameras();
            }
        },
        
        // 启用立体视觉处理
        enableStereoVision: async function(stereoPairId) {
            try {
                const response = await fetch(`${this.API_ENDPOINTS.ENABLE_STEREO}${stereoPairId}`, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    }
                });
                
                if (!response.ok) {
                    throw new Error(`HTTP error! Status: ${response.status}`);
                }
                
                const data = await response.json();
                console.log(`Stereo vision enabled for pair ${stereoPairId}`, data);
                
                if (data.status === 'success' && data.stereo_pair) {
                    this.activeStereoSessions[stereoPairId] = {
                        pair: data.stereo_pair,
                        enabled: true,
                        lastUpdated: new Date()
                    };
                }
                
                return data;
            } catch (error) {
                console.error(`Error enabling stereo vision for pair ${stereoPairId}:`, error);
                throw error;
            }
        },
        
        // 禁用立体视觉处理
        disableStereoVision: async function(stereoPairId) {
            try {
                const response = await fetch(`${this.API_ENDPOINTS.DISABLE_STEREO}${stereoPairId}`, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    }
                });
                
                if (!response.ok) {
                    throw new Error(`HTTP error! Status: ${response.status}`);
                }
                
                const data = await response.json();
                console.log(`Stereo vision disabled for pair ${stereoPairId}`, data);
                
                if (data.status === 'success' && this.activeStereoSessions[stereoPairId]) {
                    delete this.activeStereoSessions[stereoPairId];
                }
                
                return data;
            } catch (error) {
                console.error(`Error disabling stereo vision for pair ${stereoPairId}:`, error);
                throw error;
            }
        },
        
        // 获取深度数据
        getDepthData: async function(stereoPairId) {
            try {
                const response = await fetch(`${this.API_ENDPOINTS.DEPTH_DATA}${stereoPairId}`);
                
                if (!response.ok) {
                    throw new Error(`HTTP error! Status: ${response.status}`);
                }
                
                const data = await response.json();
                console.log(`Depth data retrieved for stereo pair ${stereoPairId}`, data);
                return data;
            } catch (error) {
                console.error(`Error retrieving depth data for stereo pair ${stereoPairId}:`, error);
                throw error;
            }
        },
        
        // 显示深度图
        displayDepthMap: function(canvasId, depthData, width, height) {
            const canvas = document.getElementById(canvasId);
            if (!canvas) {
                console.error(`Canvas element with id ${canvasId} not found`);
                return false;
            }
            
            try {
                canvas.width = width;
                canvas.height = height;
                const ctx = canvas.getContext('2d');
                const imageData = ctx.createImageData(width, height);
                
                // 假设depthData是一维数组，包含归一化的深度值(0-1)
                for (let i = 0; i < depthData.length; i++) {
                    const pixelIndex = i * 4;
                    // 将深度值映射到灰度
                    const grayValue = Math.floor(depthData[i] * 255);
                    
                    imageData.data[pixelIndex] = grayValue;     // R
                    imageData.data[pixelIndex + 1] = grayValue; // G
                    imageData.data[pixelIndex + 2] = grayValue; // B
                    imageData.data[pixelIndex + 3] = 255;       // A
                }
                
                ctx.putImageData(imageData, 0, 0);
                console.log(`Depth map displayed on canvas ${canvasId}`);
                return true;
            } catch (error) {
                console.error('Error displaying depth map:', error);
                return false;
            }
        },
        
        // 获取当前立体视觉对配置
        getStereoPairs: function() {
            return this.stereoPairs;
        },
        
        // 获取活动的立体视觉会话
        getActiveStereoSessions: function() {
            return this.activeStereoSessions;
        },
        
        // 将相机流附加到视频元素
        attachStreamToVideo: function(stream, videoElementId) {
            const videoElement = document.getElementById(videoElementId);
            if (!videoElement) {
                console.error(`Video element with id ${videoElementId} not found`);
                return false;
            }
            
            try {
                videoElement.srcObject = stream;
                
                videoElement.onloadedmetadata = function() {
                    console.log(`Camera stream attached to video element ${videoElementId}`);
                    // videoElement.play(); // 自动播放可能会被浏览器策略阻止
                };
                
                return true;
            } catch (error) {
                console.error('Error attaching stream to video element:', error);
                return false;
            }
        },
        
        // 从相机流创建画布快照
        takeSnapshot: function(videoElementId) {
            const videoElement = document.getElementById(videoElementId);
            if (!videoElement) {
                console.error(`Video element with id ${videoElementId} not found`);
                return null;
            }
            
            try {
                const canvas = document.createElement('canvas');
                canvas.width = videoElement.videoWidth;
                canvas.height = videoElement.videoHeight;
                
                const context = canvas.getContext('2d');
                context.drawImage(videoElement, 0, 0, canvas.width, canvas.height);
                
                // 将画布转换为数据URL
                const imageUrl = canvas.toDataURL('image/png');
                console.log('Camera snapshot taken');
                
                return imageUrl;
            } catch (error) {
                console.error('Error taking camera snapshot:', error);
                return null;
            }
        }
    };
    
    // 将CameraControl对象添加到window对象上
    window.CameraControl = CameraControl;
    
    // 初始化CameraControl当DOM加载完成
    document.addEventListener('DOMContentLoaded', function() {
        // 再次检查确保CameraControl存在
        if (window.CameraControl) {
            window.CameraControl.init();
        }
    });
})(window);