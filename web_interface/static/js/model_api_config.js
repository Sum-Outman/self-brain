/**
 * Model API Configuration - Module for managing external API connections for models
 * Allows each model to connect to external APIs independently
 */
class ModelAPIConfig {
    /**
     * Constructor for ModelAPIConfig class
     */
    constructor() {
        // Store API configurations for each model type
        this.apiConfigs = this.loadConfigsFromStorage();
        
        // Default API settings for various providers
        this.defaultProviderSettings = {
            'openai': {
                baseUrl: 'https://api.openai.com/v1',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': 'Bearer {{apiKey}}'
                },
                models: ['gpt-4', 'gpt-3.5-turbo', 'davinci-002'],
                endpointMapping: {
                    'chat': '/chat/completions',
                    'completion': '/completions',
                    'embeddings': '/embeddings'
                }
            },
            'anthropic': {
                baseUrl: 'https://api.anthropic.com/v1',
                headers: {
                    'Content-Type': 'application/json',
                    'x-api-key': '{{apiKey}}',
                    'anthropic-version': '2023-06-01'
                },
                models: ['claude-3-opus-20240229', 'claude-3-sonnet-20240229', 'claude-3-haiku-20240307'],
                endpointMapping: {
                    'chat': '/messages'
                }
            },
            'google': {
                baseUrl: 'https://generativelanguage.googleapis.com/v1',
                headers: {
                    'Content-Type': 'application/json'
                },
                models: ['gemini-pro', 'gemini-pro-vision'],
                endpointMapping: {
                    'chat': '/models/{{modelName}}:generateContent?key={{apiKey}}',
                    'vision': '/models/{{modelName}}:generateContent?key={{apiKey}}'
                }
            },
            'mistral': {
                baseUrl: 'https://api.mistral.ai/v1',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': 'Bearer {{apiKey}}'
                },
                models: ['mistral-large-latest', 'mistral-medium-latest', 'mistral-small-latest'],
                endpointMapping: {
                    'chat': '/chat/completions',
                    'completion': '/completions',
                    'embeddings': '/embeddings'
                }
            },
            'cohere': {
                baseUrl: 'https://api.cohere.ai/v1',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': 'Bearer {{apiKey}}'
                },
                models: ['command-r-plus', 'command-r', 'command'],
                endpointMapping: {
                    'chat': '/chat',
                    'completion': '/generate',
                    'embeddings': '/embed'
                }
            }
        };
    }

    /**
     * Load configurations from localStorage
     * @returns {Object} Loaded configurations
     */
    loadConfigsFromStorage() {
        try {
            const saved = localStorage.getItem('modelAPIConfigs');
            return saved ? JSON.parse(saved) : {};
        } catch (error) {
            console.error('Error loading API configurations:', error);
            return {};
        }
    }

    /**
     * Save configurations to localStorage
     */
    saveConfigsToStorage() {
        try {
            localStorage.setItem('modelAPIConfigs', JSON.stringify(this.apiConfigs));
            console.log('API configurations saved to storage');
        } catch (error) {
            console.error('Error saving API configurations:', error);
        }
    }

    /**
     * Get configuration for a specific model
     * @param {string} modelId - The model identifier (A-K)
     * @returns {Object|null} The model's API configuration or null if not found
     */
    getModelConfig(modelId) {
        return this.apiConfigs[modelId] || null;
    }

    /**
     * Set configuration for a specific model
     * @param {string} modelId - The model identifier (A-K)
     * @param {Object} config - The API configuration
     */
    setModelConfig(modelId, config) {
        this.apiConfigs[modelId] = {
            enabled: false, // Default to disabled until tested
            provider: config.provider || 'custom',
            apiKey: config.apiKey || '',
            baseUrl: config.baseUrl || '',
            modelName: config.modelName || '',
            headers: config.headers || {},
            endpointMapping: config.endpointMapping || {},
            timeout: config.timeout || 30000,
            createdAt: new Date().toISOString()
        };
        
        // If using a known provider, apply default settings
        if (this.defaultProviderSettings[config.provider]) {
            const defaults = this.defaultProviderSettings[config.provider];
            this.apiConfigs[modelId] = {
                ...this.apiConfigs[modelId],
                baseUrl: this.apiConfigs[modelId].baseUrl || defaults.baseUrl,
                headers: this.apiConfigs[modelId].headers || defaults.headers,
                endpointMapping: this.apiConfigs[modelId].endpointMapping || defaults.endpointMapping
            };
        }
        
        this.saveConfigsToStorage();
        return this.apiConfigs[modelId];
    }

    /**
     * Toggle external API usage for a model
     * @param {string} modelId - The model identifier (A-K)
     * @param {boolean} enable - Whether to enable external API
     */
    toggleModelAPI(modelId, enable) {
        if (this.apiConfigs[modelId]) {
            this.apiConfigs[modelId].enabled = enable;
            this.saveConfigsToStorage();
            return true;
        }
        return false;
    }

    /**
     * Test API connection for a model
     * @param {string} modelId - The model identifier (A-K)
     * @returns {Promise<Object>} Test result
     */
    async testAPIConnection(modelId) {
        const config = this.getModelConfig(modelId);
        if (!config || !config.baseUrl || !config.apiKey) {
            return {
                success: false,
                message: 'Invalid configuration: Missing baseUrl or API key'
            };
        }

        try {
            // Generate appropriate test endpoint based on provider
            let testEndpoint = '';
            let testMethod = 'GET';
            let testBody = null;
            
            if (config.provider === 'openai') {
                // For OpenAI, use models endpoint for testing
                testEndpoint = `${config.baseUrl}/models`;
            } else if (config.provider === 'anthropic') {
                // For Anthropic, use a minimal completion request
                testEndpoint = `${config.baseUrl}/messages`;
                testMethod = 'POST';
                testBody = JSON.stringify({
                    model: config.modelName || 'claude-3-haiku-20240307',
                    max_tokens: 10,
                    messages: [{ role: 'user', content: 'Hello' }]
                });
            } else if (config.provider === 'google') {
                // For Google Gemini
                const model = config.modelName || 'gemini-pro';
                testEndpoint = `${config.baseUrl}/models/${model}:generateContent?key=${config.apiKey}`;
                testMethod = 'POST';
                testBody = JSON.stringify({
                    contents: [{ parts: [{ text: 'Hello' }] }]
                });
            } else {
                // For other providers, try a simple GET request to base URL
                testEndpoint = config.baseUrl;
            }

            // Prepare headers
            const headers = {};
            for (const [key, value] of Object.entries(config.headers)) {
                // Replace placeholders in headers
                headers[key] = String(value).replace('{{apiKey}}', config.apiKey)
                                           .replace('{{modelName}}', config.modelName || '');
            }

            console.log('Testing API connection:', {
                modelId,
                endpoint: testEndpoint,
                headers: { ...headers, 'Authorization': headers['Authorization'] ? 'Bearer ***' : headers['Authorization'] },
                method: testMethod
            });

            const controller = new AbortController();
            const timeoutId = setTimeout(() => controller.abort(), config.timeout || 30000);

            const response = await fetch(testEndpoint, {
                method: testMethod,
                headers: headers,
                body: testBody,
                signal: controller.signal
            });

            clearTimeout(timeoutId);

            if (!response.ok) {
                throw new Error(`API request failed with status: ${response.status}`);
            }

            const data = await response.json();
            
            // Update the config to enabled since the test passed
            this.apiConfigs[modelId].enabled = true;
            this.apiConfigs[modelId].lastTested = new Date().toISOString();
            this.apiConfigs[modelId].testResult = 'success';
            this.saveConfigsToStorage();

            return {
                success: true,
                message: 'API connection successful',
                data: { status: response.status }
            };
        } catch (error) {
            console.error('API connection test failed:', error);
            
            // Update the config with test result
            if (this.apiConfigs[modelId]) {
                this.apiConfigs[modelId].lastTested = new Date().toISOString();
                this.apiConfigs[modelId].testResult = 'failed';
                this.saveConfigsToStorage();
            }

            return {
                success: false,
                message: `API connection failed: ${error.message || 'Unknown error'}`
            };
        }
    }

    /**
     * Get available API providers
     * @returns {Array} List of available providers
     */
    getAvailableProviders() {
        return Object.keys(this.defaultProviderSettings);
    }

    /**
     * Get models available for a specific provider
     * @param {string} provider - The provider name
     * @returns {Array} List of available models
     */
    getProviderModels(provider) {
        return this.defaultProviderSettings[provider]?.models || [];
    }

    /**
     * Get default settings for a provider
     * @param {string} provider - The provider name
     * @returns {Object|null} Default settings or null
     */
    getProviderDefaults(provider) {
        return this.defaultProviderSettings[provider] || null;
    }

    /**
     * Send a request to an external API model
     * @param {string} modelId - The model identifier (A-K)
     * @param {string} endpointType - Type of endpoint to use (chat, completion, etc.)
     * @param {Object} data - Request data
     * @returns {Promise<Object>} API response
     */
    async sendAPIRequest(modelId, endpointType, data) {
        const config = this.getModelConfig(modelId);
        
        // Check if external API is enabled and properly configured
        if (!config || !config.enabled || !config.baseUrl || !config.apiKey) {
            return {
                success: false,
                error: 'External API not configured or disabled'
            };
        }

        try {
            // Get the endpoint path based on endpoint type
            const endpointPath = config.endpointMapping[endpointType];
            if (!endpointPath) {
                throw new Error(`Endpoint type '${endpointType}' not supported for this provider`);
            }

            // Replace placeholders in endpoint path
            const endpoint = `${config.baseUrl}${endpointPath.replace('{{modelName}}', config.modelName || '').replace('{{apiKey}}', config.apiKey)}`;

            // Prepare headers with replaced placeholders
            const headers = {};
            for (const [key, value] of Object.entries(config.headers)) {
                headers[key] = String(value).replace('{{apiKey}}', config.apiKey)
                                           .replace('{{modelName}}', config.modelName || '');
            }

            console.log('Sending API request:', {
                modelId,
                endpoint: endpoint,
                endpointType: endpointType
            });

            const controller = new AbortController();
            const timeoutId = setTimeout(() => controller.abort(), config.timeout || 30000);

            const response = await fetch(endpoint, {
                method: 'POST',
                headers: headers,
                body: JSON.stringify(data),
                signal: controller.signal
            });

            clearTimeout(timeoutId);

            if (!response.ok) {
                throw new Error(`API request failed with status: ${response.status}`);
            }

            const responseData = await response.json();
            
            return {
                success: true,
                data: responseData
            };
        } catch (error) {
            console.error('API request failed:', error);
            return {
                success: false,
                error: error.message || 'Unknown error'
            };
        }
    }

    /**
     * Get all API configurations
     * @returns {Object} All model API configurations
     */
    getAllConfigs() {
        return { ...this.apiConfigs };
    }

    /**
     * Delete configuration for a specific model
     * @param {string} modelId - The model identifier (A-K)
     */
    deleteModelConfig(modelId) {
        if (this.apiConfigs[modelId]) {
            delete this.apiConfigs[modelId];
            this.saveConfigsToStorage();
            return true;
        }
        return false;
    }

    /**
     * Export configurations to a JSON file
     */
    exportConfigs() {
        const dataStr = JSON.stringify(this.apiConfigs, null, 2);
        const dataUri = 'data:application/json;charset=utf-8,'+ encodeURIComponent(dataStr);
        
        const exportFileDefaultName = `self-brain-api-configs-${new Date().toISOString().split('T')[0]}.json`;
        
        const linkElement = document.createElement('a');
        linkElement.setAttribute('href', dataUri);
        linkElement.setAttribute('download', exportFileDefaultName);
        linkElement.click();
    }

    /**
     * Import configurations from a JSON file
     * @param {File} file - The JSON file to import
     * @returns {Promise<Object>} Import result
     */
    importConfigs(file) {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            
            reader.onload = (event) => {
                try {
                    const importedConfigs = JSON.parse(event.target.result);
                    this.apiConfigs = { ...this.apiConfigs, ...importedConfigs };
                    this.saveConfigsToStorage();
                    resolve({ success: true, message: 'Configurations imported successfully' });
                } catch (error) {
                    reject({ success: false, message: 'Failed to parse import file: ' + error.message });
                }
            };
            
            reader.onerror = () => {
                reject({ success: false, message: 'Error reading file' });
            };
            
            reader.readAsText(file);
        });
    }
}

// Create a global instance for easy access
window.ModelAPIConfig = new ModelAPIConfig();

// Export the class for ES6 modules
export default ModelAPIConfig;