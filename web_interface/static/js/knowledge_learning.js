// 知识库自主学习功能

// 自主学习状态
let knowledgeLearningState = {
    isLearning: false,
    progress: 0,
    selectedModel: 'all',
    progressInterval: null
};

// 初始化知识库学习功能
$(document).ready(function() {
    // 绑定模型选择事件
    $('#model-select').change(function() {
        knowledgeLearningState.selectedModel = $(this).val();
    });
    
    // 定期检查学习状态
    checkLearningStatus();
});

// 开始知识库自主学习
function startKnowledgeLearning() {
    if (knowledgeLearningState.isLearning) {
        showLearningMessage('Learning is already in progress', 'info');
        return;
    }
    
    const model = knowledgeLearningState.selectedModel;
    
    // 显示加载状态
    $('#start-knowledge-learning').prop('disabled', true);
    $('#start-knowledge-learning').html('<i class="fas fa-spinner fa-spin me-1"></i> Starting...');
    
    $.ajax({
        url: '/api/knowledge/self_learning/start',
        type: 'POST',
        contentType: 'application/json',
        data: JSON.stringify({ model: model }),
        success: function(response) {
            if (response.status === 'success') {
                knowledgeLearningState.isLearning = true;
                knowledgeLearningState.progress = 0;
                
                // 更新按钮状态
                $('#start-knowledge-learning').prop('disabled', true);
                $('#stop-knowledge-learning').prop('disabled', false);
                $('#model-select').prop('disabled', true);
                
                showLearningMessage('Autonomous learning started successfully', 'success');
                
                // 开始轮询进度
                startProgressPolling();
            } else {
                showLearningMessage(response.message || 'Failed to start learning', 'error');
            }
        },
        error: function() {
            showLearningMessage('Failed to connect to learning API', 'error');
        },
        complete: function() {
            // 恢复按钮状态
            $('#start-knowledge-learning').html('<i class="bi bi-lightbulb me-1"></i>Start Autonomous Learning');
        }
    });
}

// 停止知识库自主学习
function stopKnowledgeLearning() {
    if (!knowledgeLearningState.isLearning) {
        showLearningMessage('No learning process is active', 'info');
        return;
    }
    
    // 显示加载状态
    $('#stop-knowledge-learning').prop('disabled', true);
    $('#stop-knowledge-learning').html('<i class="fas fa-spinner fa-spin me-1"></i> Stopping...');
    
    $.ajax({
        url: '/api/knowledge/self_learning/stop',
        type: 'POST',
        contentType: 'application/json',
        data: '{}',
        success: function(response) {
            if (response.status === 'success') {
                knowledgeLearningState.isLearning = false;
                
                // 更新按钮状态
                $('#start-knowledge-learning').prop('disabled', false);
                $('#stop-knowledge-learning').prop('disabled', true);
                $('#model-select').prop('disabled', false);
                
                showLearningMessage('Autonomous learning stopped successfully', 'success');
                
                // 停止轮询进度
                stopProgressPolling();
            } else {
                showLearningMessage(response.message || 'Failed to stop learning', 'error');
            }
        },
        error: function() {
            showLearningMessage('Failed to connect to learning API', 'error');
        },
        complete: function() {
            // 恢复按钮状态
            $('#stop-knowledge-learning').html('<i class="bi bi-lightbulb-off me-1"></i>Stop Learning');
        }
    });
}

// 检查学习状态
function checkLearningStatus() {
    $.ajax({
        url: '/api/knowledge/self_learning/status',
        type: 'GET',
        success: function(response) {
            if (response.status === 'success') {
                const data = response.data;
                knowledgeLearningState.isLearning = data.active;
                
                if (data.active) {
                    // 如果正在学习，更新UI和开始轮询
                    $('#start-knowledge-learning').prop('disabled', true);
                    $('#stop-knowledge-learning').prop('disabled', false);
                    $('#model-select').prop('disabled', true);
                    
                    if (!knowledgeLearningState.progressInterval) {
                        startProgressPolling();
                    }
                } else {
                    // 如果没有学习，更新UI
                    $('#start-knowledge-learning').prop('disabled', false);
                    $('#stop-knowledge-learning').prop('disabled', true);
                    $('#model-select').prop('disabled', false);
                }
            }
        },
        error: function() {
            console.error('Failed to check learning status');
        }
    });
}

// 开始轮询进度
function startProgressPolling() {
    // 清除现有轮询
    if (knowledgeLearningState.progressInterval) {
        clearInterval(knowledgeLearningState.progressInterval);
    }
    
    // 设置新的轮询
    knowledgeLearningState.progressInterval = setInterval(function() {
        $.ajax({
            url: '/api/knowledge/self_learning/progress',
            type: 'GET',
            success: function(response) {
                if (response.status === 'success') {
                    const data = response.data;
                    
                    // 更新进度
                    knowledgeLearningState.progress = data.progress || 0;
                    updateProgressBar();
                    
                    // 如果学习停止了，清理轮询
                    if (!data.active) {
                        stopProgressPolling();
                        $('#start-knowledge-learning').prop('disabled', false);
                        $('#stop-knowledge-learning').prop('disabled', true);
                        $('#model-select').prop('disabled', false);
                    }
                }
            },
            error: function() {
                console.error('Failed to get learning progress');
            }
        });
    }, 2000); // 每2秒更新一次
}

// 停止轮询进度
function stopProgressPolling() {
    if (knowledgeLearningState.progressInterval) {
        clearInterval(knowledgeLearningState.progressInterval);
        knowledgeLearningState.progressInterval = null;
    }
}

// 更新进度条
function updateProgressBar() {
    const progressBar = $('#knowledge-progress-bar');
    const percentageText = $('#knowledge-percentage');
    
    progressBar.css('width', knowledgeLearningState.progress + '%');
    progressBar.attr('aria-valuenow', knowledgeLearningState.progress);
    percentageText.text(knowledgeLearningState.progress + '%');
}

// 显示学习消息
function showLearningMessage(message, type) {
    // 创建消息元素
    const messageElement = $('<div class="alert alert-dismissible fade show mt-3" role="alert"></div>');
    
    // 设置消息类型
    if (type === 'success') {
        messageElement.addClass('alert-success');
        messageElement.html(`<i class="fas fa-check-circle me-2"></i>${message}`);
    } else if (type === 'error') {
        messageElement.addClass('alert-danger');
        messageElement.html(`<i class="fas fa-times-circle me-2"></i>${message}`);
    } else {
        messageElement.addClass('alert-info');
        messageElement.html(`<i class="fas fa-info-circle me-2"></i>${message}`);
    }
    
    // 添加关闭按钮
    messageElement.append('<button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>');
    
    // 添加到页面
    $('#knowledge-progress-bar').closest('.mb-3').after(messageElement);
    
    // 3秒后自动关闭
    setTimeout(function() {
        messageElement.alert('close');
    }, 3000);
}

// 查看知识分类
function viewKnowledgeCategory(category) {
    // 这里可以实现查看特定分类的知识内容
    // 例如跳转到特定页面或显示模态框
    alert(`Viewing knowledge category: ${category}`);
}

// 当页面关闭时清理轮询
$(window).on('beforeunload', function() {
    stopProgressPolling();
});