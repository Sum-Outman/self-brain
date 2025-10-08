// 训练控制状态
let trainingState = {
    currentLanguage: 'zh',
    selectedModels: [],
    trainingMode: 'single',
    isTraining: false,
    isPaused: false,
    progress: 0,
    logEntries: [],
    statusInterval: null
};

// 初始化函数
$(document).ready(function() {
    // 设置初始语言
    setLanguage('zh');
    
    // 加载可用模型
    loadAvailableModels();
    
    // 绑定事件处理
    bindEvents();
    
    // 初始化日志（双语）
    addLogEntry({
        zh: '系统已就绪，请选择模型开始训练',
        en: 'System is ready, please select models to start training'
    }, 'info');
});

// 设置页面语言
function setLanguage(lang) {
    trainingState.currentLanguage = lang;
    
    // 更新所有语言元素
    $('[data-lang]').each(function() {
        if ($(this).attr('data-lang') === lang) {
            $(this).show();
        } else {
            $(this).hide();
        }
    });
    
    // 更新语言选择器
    $('#language-select').val(lang);
    
    // 重新加载模型（更新语言）
    if (trainingState.selectedModels.length > 0) {
        loadAvailableModels();
    }
}

// 加载可用模型
function loadAvailableModels() {
    $.ajax({
        url: '/available_models',
        type: 'GET',
        data: { language: trainingState.currentLanguage },
        success: function(response) {
            renderModels(response.models);
        },
        error: function() {
        addLogEntry({
            zh: '加载模型失败，请重试',
            en: 'Failed to load models, please try again'
        }, 'error');
        }
    });
}

// 渲染模型卡片 | Render model cards
function renderModels(models) {
    const container = $('#model-container');
    container.empty();
    
    // 添加联合训练说明卡片
    if (trainingState.trainingMode === 'joint') {
        const jointInfoHtml = `
            <div class="col-12 mb-4">
                <div class="alert alert-info">
                    <i class="fas fa-info-circle me-2"></i>
                    <span data-lang="zh">联合训练模式：请选择至少两个模型进行协同训练。知识库模型(I)将自动参与所有联合训练。</span>
                    <span data-lang="en" style="display: none;">Joint training mode: Please select at least two models for collaborative training. Knowledge model (I) will automatically participate in all joint training.</span>
                </div>
            </div>
        `;
        container.append(jointInfoHtml);
    }
    
    models.forEach(model => {
        // 在联合训练模式下，知识库模型自动选中且不可取消
        const isKnowledgeModel = model.type === 'I';
        const isAutoSelected = trainingState.trainingMode === 'joint' && isKnowledgeModel;
        const isSelected = isAutoSelected || trainingState.selectedModels.includes(model.id);
        const cardClass = isSelected ? 'model-card selected' : 'model-card';
        
        // 添加来源标识
        const sourceBadge = model.source === 'api' ? 
            `<span class="badge bg-warning ms-2" data-lang="zh">API</span>
             <span class="badge bg-warning ms-2" data-lang="en" style="display: none;">API</span>` : 
            `<span class="badge bg-info ms-2" data-lang="zh">本地</span>
             <span class="badge bg-info ms-2" data-lang="en" style="display: none;">Local</span>`;
        
        // 知识库模型特殊标记
        const knowledgeBadge = isKnowledgeModel ? 
            `<span class="badge bg-success ms-2" data-lang="zh">知识库</span>
             <span class="badge bg-success ms-2" data-lang="en" style="display: none;">Knowledge</span>` : '';
        
        const cardHtml = `
            <div class="col-md-6 col-lg-4">
                <div class="${cardClass}" data-model-id="${model.id}" 
                     ${isAutoSelected ? 'data-auto-selected="true"' : ''}>
                    <div class="model-header">
                        <h6 class="model-name">${model.name}${sourceBadge}${knowledgeBadge}</h6>
                        <span class="model-type">${model.type}</span>
                    </div>
                    <p class="model-description">${model.description}</p>
                    <div class="model-footer">
                        <span class="model-status ${model.active ? 'active' : 'inactive'}">
                            ${model.active ? 
                                (trainingState.currentLanguage === 'zh' ? '可用' : 'Available') : 
                                (trainingState.currentLanguage === 'zh' ? '不可用' : 'Unavailable')}
                        </span>
                        <div>
                            ${isAutoSelected ? 
                                `<span class="badge bg-success me-2" data-lang="zh">自动参与</span>
                                 <span class="badge bg-success me-2" data-lang="en" style="display: none;">Auto</span>` : 
                                `<button class="btn btn-sm ${isSelected ? 'btn-outline-danger' : 'btn-outline-primary'} select-btn me-1">
                                    ${isSelected ? 
                                        (trainingState.currentLanguage === 'zh' ? '取消选择' : 'Deselect') : 
                                        (trainingState.currentLanguage === 'zh' ? '选择' : 'Select')}
                                </button>`
                            }
                            <button class="btn btn-sm btn-outline-secondary config-btn" data-model-id="${model.id}">
                                <i class="fas fa-cog"></i>
                            </button>
                        </div>
                    </div>
                </div>
            </div>
        `;
        
        container.append(cardHtml);
    });
    
    // 绑定模型选择事件
    $('.model-card').click(function(e) {
        // 防止配置按钮触发选择
        if ($(e.target).hasClass('config-btn') || 
            $(e.target).closest('.config-btn').length ||
            $(this).attr('data-auto-selected') === 'true') {
            return;
        }
        const modelId = $(this).data('model-id');
        toggleModelSelection(modelId);
    });
    
    // 绑定配置按钮事件
    $('.config-btn').click(function(e) {
        e.stopPropagation();
        const modelId = $(this).data('model-id');
        openModelConfig(modelId);
    });
}

// 切换模型选择 | Toggle model selection
function toggleModelSelection(modelId) {
    const index = trainingState.selectedModels.indexOf(modelId);
    
    if (index === -1) {
        // 单训练模式只能选择一个模型
        if (trainingState.trainingMode === 'single') {
            trainingState.selectedModels = [modelId];
        } else {
            trainingState.selectedModels.push(modelId);
        }
    } else {
        trainingState.selectedModels.splice(index, 1);
    }
    
    // 在联合训练模式下，确保知识库模型始终被包含
    if (trainingState.trainingMode === 'joint') {
        const knowledgeModel = models.find(m => m.type === 'I');
        if (knowledgeModel && !trainingState.selectedModels.includes(knowledgeModel.id)) {
            trainingState.selectedModels.push(knowledgeModel.id);
        }
    }
    
    loadAvailableModels();
    updateUI();
}

// 设置训练模式 | Set training mode
function setTrainingMode(mode) {
    trainingState.trainingMode = mode;
    trainingState.selectedModels = [];
    
    // 在联合训练模式下自动包含知识库模型
    if (mode === 'joint') {
        // 获取知识库模型ID
        $.ajax({
            url: '/get_knowledge_model',
            type: 'GET',
            success: function(response) {
                if (response.status === 'success' && response.model_id) {
                    trainingState.selectedModels.push(response.model_id);
                }
                loadAvailableModels();
            }
        });
    }
    
    // 更新按钮激活状态
    if (mode === 'single') {
        $('#single-training').addClass('active');
        $('#joint-training').removeClass('active');
    } else {
        $('#single-training').removeClass('active');
        $('#joint-training').addClass('active');
    }
    
    // 更新UI并重新加载模型
    updateUI();
    loadAvailableModels();
}

// 绑定事件处理
function bindEvents() {
    // 语言切换
    $('#language-select').change(function() {
        setLanguage($(this).val());
    });
    
    // 训练模式切换
    $('#single-training').click(function() {
        setTrainingMode('single');
    }).keydown(function(e) {
        if (e.key === 'Enter' || e.key === ' ') {
            setTrainingMode('single');
        }
    });
    
    $('#joint-training').click(function() {
        setTrainingMode('joint');
    }).keydown(function(e) {
        if (e.key === 'Enter' || e.key === ' ') {
            setTrainingMode('joint');
        }
    });
    
    // 开始训练
    $('#start-training').click(function() {
        startTraining();
    }).keydown(function(e) {
        if (e.key === 'Enter' || e.key === ' ') {
            startTraining();
        }
    });
    
    // 暂停/恢复训练
    $('#pause-training').click(function() {
        togglePauseTraining();
    }).keydown(function(e) {
        if (e.key === 'Enter' || e.key === ' ') {
            togglePauseTraining();
        }
    });
    
    // 停止训练
    $('#stop-training').click(function() {
        stopTraining();
    }).keydown(function(e) {
        if (e.key === 'Enter' || e.key === ' ') {
            stopTraining();
        }
    });
    
    // 模型卡片键盘支持
    $(document).on('keydown', '.model-card', function(e) {
        if (e.key === 'Enter' || e.key === ' ') {
            const modelId = $(this).data('model-id');
            toggleModelSelection(modelId);
            e.preventDefault();
        }
    });
    
    // 实时输入源切换
    $('#video-source').change(function() {
        if ($(this).val() === 'stream') {
            $('#stream-url-group').show();
        } else {
            $('#stream-url-group').hide();
        }
    });
    
    $('#audio-source').change(function() {
        if ($(this).val() === 'stream') {
            $('#audio-url-group').show();
        } else {
            $('#audio-url-group').hide();
        }
    });
    
    // 保存接口配置
    $('#save-interfaces').click(function() {
        saveInterfaceConfig();
    });
    
    // 模型来源切换
    $('#model-source').change(function() {
        if ($(this).val() === 'api') {
            $('#api-config').show();
        } else {
            $('#api-config').hide();
        }
    });
    
    // 测试连接
    $('#test-connection').click(function() {
        testApiConnection();
    });
    
    // 保存模型配置
    $('#save-model-config').click(function() {
        saveModelConfig();
    });
    
    // 知识库辅助开关
    $('#knowledge-assist').change(function() {
        toggleKnowledgeAssist($(this).prop('checked'));
    });
}

// 开始训练 | Start training
function startTraining() {
    if (trainingState.isTraining) {
        addLogEntry({
            zh: '训练已在进行中',
            en: 'Training is already in progress'
        }, 'info');
        return;
    }
    
    // 验证选择
    if (trainingState.selectedModels.length === 0) {
        addLogEntry({
            zh: '请至少选择一个模型',
            en: 'Please select at least one model'
        }, 'error');
        return;
    }
    
    if (trainingState.trainingMode === 'single' && trainingState.selectedModels.length > 1) {
        addLogEntry({
            zh: '单训练模式只能选择一个模型',
            en: 'Single training mode can only select one model'
        }, 'error');
        return;
    }
    
    if (trainingState.trainingMode === 'joint' && trainingState.selectedModels.length < 2) {
        // 检查是否包含知识库模型
        const hasKnowledgeModel = trainingState.selectedModels.some(id => {
            const model = models.find(m => m.id === id);
            return model && model.type === 'I';
        });
        
        if (!hasKnowledgeModel) {
            addLogEntry({
                zh: '联合训练需要至少选择两个模型（包含知识库模型）',
                en: 'Joint training requires at least two models (including knowledge model)'
            }, 'error');
            return;
        }
    }
    
    // 准备请求数据
    const requestData = {
        mode: trainingState.trainingMode,
        language: trainingState.currentLanguage,
        // 添加知识库辅助状态
        knowledge_assist: $('#knowledge-assist').prop('checked')
    };
    
    if (trainingState.trainingMode === 'single') {
        requestData.model_id = trainingState.selectedModels[0];
    } else {
        requestData.model_ids = trainingState.selectedModels;
        
        // 添加联合训练配置
        requestData.joint_config = {
            data_sharing: true,  // 启用模型间数据共享
            coordinator: 'A'     // 指定管理模型A作为协调器
        };
    }
    
    // 添加实时输入配置
    requestData.realtime_input = {
        video_source: $('#video-source').val(),
        video_stream_url: $('#video-stream-url').val(),
        audio_source: $('#audio-source').val(),
        audio_stream_url: $('#audio-stream-url').val()
    };
    
    // 发送请求
    $.ajax({
        url: '/start_training',
        type: 'POST',
        contentType: 'application/json',
        data: JSON.stringify(requestData),
        success: function(response) {
            if (response.status === 'success') {
                trainingState.isTraining = true;
                trainingState.isPaused = false;
                trainingState.progress = 0;
                trainingState.logEntries = [];
                
                addLogEntry({
                    zh: '训练已开始',
                    en: 'Training started'
                }, 'success');
                
                // 添加联合训练特定日志
                if (trainingState.trainingMode === 'joint') {
                    addLogEntry({
                        zh: '联合训练模式：模型间数据共享已启用',
                        en: 'Joint training: Model data sharing enabled'
                    }, 'info');
                    
                    addLogEntry({
                        zh: '知识库模型(I)正在为其他模型提供辅助',
                        en: 'Knowledge model (I) is assisting other models'
                    }, 'info');
                }
                
                updateUI();
                startStatusPolling();
            } else {
                addLogEntry(response.message, 'error');
            }
        },
        error: function() {
                addLogEntry({
                    zh: '启动训练失败',
                    en: 'Failed to start training'
                }, 'error');
        }
    });
}

// 暂停/恢复训练
function togglePauseTraining() {
    if (!trainingState.isTraining) return;
    
    $.ajax({
        url: '/pause_resume_training',
        type: 'POST',
        contentType: 'application/json',
        data: JSON.stringify({ language: trainingState.currentLanguage }),
        success: function(response) {
            if (response.status === 'success') {
                trainingState.isPaused = response.is_paused;
                updateUI();
                
                if (trainingState.isPaused) {
                addLogEntry({
                    zh: '训练已暂停',
                    en: 'Training paused'
                }, 'info');
                } else {
                addLogEntry({
                    zh: '训练已恢复',
                    en: 'Training resumed'
                }, 'info');
                }
            }
        },
        error: function() {
                addLogEntry({
                    zh: '操作失败',
                    en: 'Operation failed'
                }, 'error');
        }
    });
}

// 停止训练
function stopTraining() {
    if (!trainingState.isTraining) return;
    
    $.ajax({
        url: '/stop_training',
        type: 'POST',
        contentType: 'application/json',
        data: JSON.stringify({ language: trainingState.currentLanguage }),
        success: function(response) {
            if (response.status === 'success') {
                addLogEntry({
                    zh: '训练停止请求已发送',
                    en: 'Training stop request sent'
                }, 'info');
            }
        },
        error: function() {
                addLogEntry({
                    zh: '停止训练失败',
                    en: 'Failed to stop training'
                }, 'error');
        }
    });
}

// 开始轮询训练状态
function startStatusPolling() {
    if (trainingState.statusInterval) {
        clearInterval(trainingState.statusInterval);
    }
    
    trainingState.statusInterval = setInterval(function() {
        $.ajax({
            url: '/training_status',
            type: 'GET',
            data: { language: trainingState.currentLanguage },
            success: function(response) {
                trainingState.progress = response.progress;
                trainingState.isTraining = response.is_training;
                trainingState.isPaused = response.is_paused;
                
                // 添加新日志
                if (response.log && response.log.length > trainingState.logEntries.length) {
                    const newEntries = response.log.slice(trainingState.logEntries.length);
                    newEntries.forEach(entry => {
                        addLogEntry(entry, 'info');
                    });
                    trainingState.logEntries = response.log;
                }
                
                // 更新进度条
                updateProgressBar();
                
                // 如果训练结束，停止轮询
                if (!trainingState.isTraining) {
                    clearInterval(trainingState.statusInterval);
                    trainingState.statusInterval = null;
                    updateUI();
                }
            },
            error: function() {
            addLogEntry({
                zh: '获取训练状态失败',
                en: 'Failed to get training status'
            }, 'error');
            }
        });
    }, 1000);
}

// 更新进度条
function updateProgressBar() {
    const progressBar = $('#progress-fill');
    progressBar.css('width', trainingState.progress + '%');
    progressBar.attr('aria-valuenow', trainingState.progress);
    $('#progress-text').text(trainingState.progress + '%');
}

// 添加日志条目
function addLogEntry(message, type) {
    const logContainer = $('#training-log');
    
    // 创建日志条目容器
    const logEntry = $('<div class="log-entry"></div>');
    logEntry.addClass(type);
    
    // 支持中英文消息
    let zhMessage, enMessage;
    if (typeof message === 'object') {
        zhMessage = message.zh;
        enMessage = message.en;
    } else {
        zhMessage = message;
        enMessage = message;
    }
    
    // 添加中文版本
    const zhSpan = $('<span data-lang="zh"></span>').text(zhMessage);
    logEntry.append(zhSpan);
    
    // 添加英文版本
    const enSpan = $('<span data-lang="en"></span>').text(enMessage);
    logEntry.append(enSpan);
    
    // 设置语言显示
    logEntry.attr('data-lang', trainingState.currentLanguage);
    $('[data-lang]', logEntry).hide();
    $(`[data-lang="${trainingState.currentLanguage}"]`, logEntry).show();
    
    logContainer.append(logEntry);
    logContainer.scrollTop(logContainer[0].scrollHeight);
}

// 打开模型配置
function openModelConfig(modelId) {
    // 获取模型配置
    $.ajax({
        url: '/get_model_config',
        type: 'GET',
        data: { model_id: modelId },
        success: function(response) {
            if (response.status === 'success') {
                const config = response.config;
                
                // 填充模态框
                $('#model-source').val(config.source || 'local');
                $('#api-endpoint').val(config.api_endpoint || '');
                $('#api-key').val(config.api_key || '');
                $('#api-model-name').val(config.model_name || '');
                
                // 显示/隐藏API配置
                if (config.source === 'api') {
                    $('#api-config').show();
                } else {
                    $('#api-config').hide();
                }
                
                // 设置当前模型ID
                $('#modelConfigModal').data('model-id', modelId);
                
                // 显示模态框
                const modal = new bootstrap.Modal(document.getElementById('modelConfigModal'));
                modal.show();
            } else {
                addLogEntry(response.message, 'error');
            }
        },
        error: function() {
            addLogEntry({
                zh: '获取模型配置失败',
                en: 'Failed to get model configuration'
            }, 'error');
        }
    });
}

// 测试API连接
function testApiConnection() {
    const endpoint = $('#api-endpoint').val();
    const apiKey = $('#api-key').val();
    const modelName = $('#api-model-name').val();
    
    if (!endpoint || !apiKey || !modelName) {
        addLogEntry({
            zh: '请填写所有API配置字段',
            en: 'Please fill all API configuration fields'
        }, 'error');
        return;
    }
    
    $('#test-connection').prop('disabled', true);
    $('#connection-result').html(`
        <div class="spinner-border spinner-border-sm" role="status">
            <span class="visually-hidden">Loading...</span>
        </div>
        <span data-lang="zh">测试连接中...</span>
        <span data-lang="en" style="display: none;">Testing connection...</span>
    `);
    
    $.ajax({
        url: '/test_api_connection',
        type: 'POST',
        contentType: 'application/json',
        data: JSON.stringify({
            endpoint: endpoint,
            api_key: apiKey,
            model_name: modelName
        }),
        success: function(response) {
            $('#test-connection').prop('disabled', false);
            if (response.status === 'success') {
                $('#connection-result').html(`
                    <div class="text-success">
                        <i class="fas fa-check-circle me-1"></i>
                        <span data-lang="zh">连接成功！延迟: ${response.latency}ms</span>
                        <span data-lang="en" style="display: none;">Connection successful! Latency: ${response.latency}ms</span>
                    </div>
                `);
            } else {
                $('#connection-result').html(`
                    <div class="text-danger">
                        <i class="fas fa-times-circle me-1"></i>
                        <span data-lang="zh">连接失败: ${response.message}</span>
                        <span data-lang="en" style="display: none;">Connection failed: ${response.message}</span>
                    </div>
                `);
            }
        },
        error: function() {
            $('#test-connection').prop('disabled', false);
            $('#connection-result').html(`
                <div class="text-danger">
                    <i class="fas fa-times-circle me-1"></i>
                    <span data-lang="zh">连接测试请求失败</span>
                    <span data-lang="en" style="display: none;">Connection test request failed</span>
                </div>
            `);
        }
    });
}

// 保存模型配置
function saveModelConfig() {
    const modelId = $('#modelConfigModal').data('model-id');
    const source = $('#model-source').val();
    const apiEndpoint = $('#api-endpoint').val();
    const apiKey = $('#api-key').val();
    const modelName = $('#api-model-name').val();
    
    $.ajax({
        url: '/save_model_config',
        type: 'POST',
        contentType: 'application/json',
        data: JSON.stringify({
            model_id: modelId,
            source: source,
            api_endpoint: apiEndpoint,
            api_key: apiKey,
            model_name: modelName
        }),
        success: function(response) {
            if (response.status === 'success') {
                addLogEntry({
                    zh: `模型 ${modelId} 配置已保存`,
                    en: `Model ${modelId} configuration saved`
                }, 'success');
                
                // 关闭模态框
                const modal = bootstrap.Modal.getInstance(document.getElementById('modelConfigModal'));
                modal.hide();
                
                // 重新加载模型
                loadAvailableModels();
            } else {
                addLogEntry(response.message, 'error');
            }
        },
        error: function() {
            addLogEntry({
                zh: '保存模型配置失败',
                en: 'Failed to save model configuration'
            }, 'error');
        }
    });
}

// 保存接口配置
function saveInterfaceConfig() {
    const videoSource = $('#video-source').val();
    const videoStreamUrl = $('#video-stream-url').val();
    const audioSource = $('#audio-source').val();
    const audioStreamUrl = $('#audio-stream-url').val();
    
    $.ajax({
        url: '/save_interface_config',
        type: 'POST',
        contentType: 'application/json',
        data: JSON.stringify({
            video_source: videoSource,
            video_stream_url: videoStreamUrl,
            audio_source: audioSource,
            audio_stream_url: audioStreamUrl
        }),
        success: function(response) {
            if (response.status === 'success') {
                addLogEntry({
                    zh: '接口配置已保存',
                    en: 'Interface configuration saved'
                }, 'success');
            } else {
                addLogEntry(response.message, 'error');
            }
        },
        error: function() {
            addLogEntry({
                zh: '保存接口配置失败',
                en: 'Failed to save interface configuration'
            }, 'error');
        }
    });
}

// 切换知识库辅助
function toggleKnowledgeAssist(enabled) {
    $.ajax({
        url: '/toggle_knowledge_assist',
        type: 'POST',
        contentType: 'application/json',
        data: JSON.stringify({
            enabled: enabled
        }),
        success: function(response) {
            if (response.status === 'success') {
                addLogEntry({
                    zh: `知识库辅助已${enabled ? '启用' : '禁用'}`,
                    en: `Knowledge assist ${enabled ? 'enabled' : 'disabled'}`
                }, 'success');
            } else {
                addLogEntry(response.message, 'error');
            }
        },
        error: function() {
            addLogEntry({
                zh: '切换知识库辅助失败',
                en: 'Failed to toggle knowledge assist'
            }, 'error');
        }
    });
}

// 更新UI状态
function updateUI() {
    // 更新按钮状态
    $('#start-training').prop('disabled', trainingState.isTraining);
    $('#pause-training').prop('disabled', !trainingState.isTraining);
    $('#stop-training').prop('disabled', !trainingState.isTraining);
    
    // 更新暂停按钮文本
    if (trainingState.isPaused) {
        $('#pause-training').html('<i class="fas fa-play me-2"></i><span data-lang="zh">恢复训练</span><span data-lang="en" style="display: none;">Resume Training</span>');
    } else {
        $('#pause-training').html('<i class="fas fa-pause me-2"></i><span data-lang="zh">暂停训练</span><span data-lang="en" style="display: none;">Pause Training</span>');
    }
    
    // 更新按钮激活状态
    if (trainingState.trainingMode === 'single') {
        $('#single-training').addClass('active');
        $('#joint-training').removeClass('active');
    } else {
        $('#single-training').removeClass('active');
        $('#joint-training').addClass('active');
    }
    
    // 更新语言显示
    $('[data-lang]').each(function() {
        if ($(this).attr('data-lang') === trainingState.currentLanguage) {
            $(this).show();
        } else {
            $(this).hide();
        }
    });
    
    // 更新实时输入源显示
    if ($('#video-source').val() === 'stream') {
        $('#stream-url-group').show();
    } else {
        $('#stream-url-group').hide();
    }
    
    if ($('#audio-source').val() === 'stream') {
        $('#audio-url-group').show();
    } else {
        $('#audio-url-group').hide();
    }
}
