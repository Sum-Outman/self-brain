/**
 * DeviceCommunication - Class for handling device communication with backend APIs
 */
class DeviceCommunication {
    /**
     * Constructor for DeviceCommunication class
     */
    constructor() {
        this.baseUrl = '/api/device';
        this.deviceCommunicationBaseUrl = '/api/device_communication';
        this.serialConnection = null;
        this.isConnected = false;
        this.sensorUpdateCallbacks = [];
        this.sensorPollingInterval = null;
        this.sensorPollingRate = 1000; // 1 second
    }

    /**
     * Register a callback for sensor data updates
     * @param {Function} callback - The callback function to register
     */
    onSensorUpdate(callback) {
        if (typeof callback === 'function' && !this.sensorUpdateCallbacks.includes(callback)) {
            this.sensorUpdateCallbacks.push(callback);
        }
    }

    /**
     * Unregister a callback for sensor data updates
     * @param {Function} callback - The callback function to unregister
     */
    offSensorUpdate(callback) {
        const index = this.sensorUpdateCallbacks.indexOf(callback);
        if (index > -1) {
            this.sensorUpdateCallbacks.splice(index, 1);
        }
    }

    /**
     * Notify all registered callbacks with new sensor data
     * @param {Object} sensorData - The new sensor data
     */
    notifySensorUpdate(sensorData) {
        this.sensorUpdateCallbacks.forEach(callback => {
            try {
                callback(sensorData);
            } catch (error) {
                console.error('Error in sensor update callback:', error);
            }
        });
    }

    /**
     * Get system sensor data
     * @returns {Promise<Object>} The sensor data response
     */
    async getSensorData() {
        try {
            const response = await fetch(`${this.baseUrl}/sensor_data`);
            const data = await response.json();
            
            // Notify callbacks if data is successful
            if (data.status === 'success') {
                this.notifySensorUpdate(data);
            }
            
            return data;
        } catch (error) {
            console.error('Error fetching sensor data:', error);
            return {
                status: 'error',
                message: 'Failed to fetch sensor data',
                error: error.message
            };
        }
    }

    /**
     * Get available serial ports
     * @returns {Promise<Object>} The serial ports response
     */
    async getSerialPorts() {
        try {
            const response = await fetch(`${this.baseUrl}/serial_ports`);
            return await response.json();
        } catch (error) {
            console.error('Error fetching serial ports:', error);
            return {
                status: 'error',
                message: 'Failed to fetch serial ports',
                error: error.message
            };
        }
    }

    /**
     * Connect to a serial port
     * @param {string} port - The serial port name
     * @param {number} baudRate - The baud rate
     * @returns {Promise<Object>} The connection response
     */
    async connectSerialPort(port, baudRate) {
        try {
            // Try the new API path first
            try {
                const response = await fetch(`${this.deviceCommunicationBaseUrl}/serial/connect`, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        port: port,
                        baudrate: baudRate
                    })
                });
                
                const data = await response.json();
                
                if (data.status === 'success') {
                    this.isConnected = true;
                    this.serialConnection = {
                        port: port,
                        baudRate: baudRate
                    };
                }
                
                return data;
            } catch (newApiError) {
                console.warn('New API failed, trying legacy API:', newApiError);
                // Fallback to legacy API if new API fails
                const response = await fetch(`${this.baseUrl}/serial_connect`, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        port: port,
                        baud_rate: baudRate
                    })
                });
                
                const data = await response.json();
                
                if (data.status === 'success') {
                    this.isConnected = true;
                    this.serialConnection = {
                        port: port,
                        baudRate: baudRate
                    };
                }
                
                return data;
            }
        } catch (error) {
            console.error('Error connecting to serial port:', error);
            return {
                status: 'error',
                message: 'Failed to connect to serial port',
                error: error.message
            };
        }
    }

    /**
     * Disconnect from the current serial port
     * @returns {Promise<Object>} The disconnection response
     */
    async disconnectSerialPort() {
        try {
            // Try the new API path first
            try {
                const response = await fetch(`${this.deviceCommunicationBaseUrl}/serial/disconnect`, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        port: this.serialConnection?.port
                    })
                });
                
                const data = await response.json();
                
                if (data.status === 'success') {
                    this.isConnected = false;
                    this.serialConnection = null;
                }
                
                return data;
            } catch (newApiError) {
                console.warn('New API failed, trying legacy API:', newApiError);
                // Fallback to legacy API if new API fails
                const response = await fetch(`${this.baseUrl}/serial_disconnect`, {
                    method: 'POST'
                });
                
                const data = await response.json();
                
                if (data.status === 'success') {
                    this.isConnected = false;
                    this.serialConnection = null;
                }
                
                return data;
            }
        } catch (error) {
            console.error('Error disconnecting from serial port:', error);
            return {
                status: 'error',
                message: 'Failed to disconnect from serial port',
                error: error.message
            };
        }
    }

    /**
     * Send a command to the serial port
     * @param {string} command - The command to send
     * @returns {Promise<Object>} The command response
     */
    async sendSerialCommand(command) {
        if (!this.isConnected) {
            return {
                status: 'error',
                message: 'Not connected to any serial port'
            };
        }
        
        try {
            // Try the new API path first
            try {
                const response = await fetch(`${this.deviceCommunicationBaseUrl}/serial/send`, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        port: this.serialConnection.port,
                        command: command
                    })
                });
                
                return await response.json();
            } catch (newApiError) {
                console.warn('New API failed, trying legacy API:', newApiError);
                // Fallback to legacy API if new API fails
                const response = await fetch(`${this.baseUrl}/serial_command`, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        command: command
                    })
                });
                
                return await response.json();
            }
        } catch (error) {
            console.error('Error sending serial command:', error);
            return {
                status: 'error',
                message: 'Failed to send serial command',
                error: error.message
            };
        }
    }

    /**
     * Get all available devices (serial ports, cameras, etc.)
     * @returns {Promise<Object>} The available devices response
     */
    async getAvailableDevices() {
        try {
            const response = await fetch(`${this.deviceCommunicationBaseUrl}/available_devices`);
            return await response.json();
        } catch (error) {
            console.error('Error fetching available devices:', error);
            return {
                status: 'error',
                message: 'Failed to fetch available devices',
                error: error.message
            };
        }
    }

    /**
     * Get status of all connected serial devices
     * @returns {Promise<Object>} The serial devices status response
     */
    async getSerialDevicesStatus() {
        try {
            const response = await fetch(`${this.deviceCommunicationBaseUrl}/serial/devices`);
            return await response.json();
        } catch (error) {
            console.error('Error fetching serial devices status:', error);
            return {
                status: 'error',
                message: 'Failed to fetch serial devices status',
                error: error.message
            };
        }
    }

    /**
     * Get all sensor data from the device communication manager
     * @returns {Promise<Object>} The sensor data response
     */
    async getAllSensorData() {
        try {
            const response = await fetch(`${this.deviceCommunicationBaseUrl}/sensors/data`);
            const data = await response.json();
            
            // Notify callbacks if data is successful
            if (data.status === 'success') {
                this.notifySensorUpdate(data);
            }
            
            return data;
        } catch (error) {
            console.error('Error fetching all sensor data:', error);
            return {
                status: 'error',
                message: 'Failed to fetch all sensor data',
                error: error.message
            };
        }
    }

    /**
     * Test microphone functionality
     * @returns {Promise<Object>} The microphone test response
     */
    async testMicrophone() {
        try {
            const response = await fetch(`${this.baseUrl}/test_microphone`);
            return await response.json();
        } catch (error) {
            console.error('Error testing microphone:', error);
            return {
                status: 'error',
                message: 'Failed to test microphone',
                error: error.message
            };
        }
    }

    /**
     * Test serial port permission
     * @returns {Promise<Object>} The serial permission test response
     */
    async testSerialPermission() {
        try {
            const response = await fetch(`${this.baseUrl}/test_serial_permission`);
            return await response.json();
        } catch (error) {
            console.error('Error testing serial permission:', error);
            return {
                status: 'error',
                message: 'Failed to test serial permission',
                error: error.message
            };
        }
    }

    /**
     * Start polling for sensor data updates
     * @param {number} rate - Polling rate in milliseconds (default: 1000)
     */
    startSensorPolling(rate = 1000) {
        // Stop any existing polling
        this.stopSensorPolling();
        
        this.sensorPollingRate = rate;
        this.sensorPollingInterval = setInterval(async () => {
            try {
                await this.getSensorData();
            } catch (error) {
                console.error('Error in sensor polling:', error);
            }
        }, this.sensorPollingRate);
        
        console.log(`Sensor polling started with rate: ${this.sensorPollingRate}ms`);
    }

    /**
     * Stop polling for sensor data updates
     */
    stopSensorPolling() {
        if (this.sensorPollingInterval) {
            clearInterval(this.sensorPollingInterval);
            this.sensorPollingInterval = null;
            console.log('Sensor polling stopped');
        }
    }

    /**
     * Get current connection status
     * @returns {Object} Connection status information
     */
    getConnectionStatus() {
        return {
            isConnected: this.isConnected,
            connection: this.serialConnection,
            isPolling: !!this.sensorPollingInterval,
            pollingRate: this.sensorPollingRate
        };
    }

    /**
     * Initialize the device communication system
     * @param {Object} options - Initialization options
     */
    async initialize(options = {}) {
        // Set base URL if provided
        if (options.baseUrl) {
            this.baseUrl = options.baseUrl;
        }
        
        if (options.deviceCommunicationBaseUrl) {
            this.deviceCommunicationBaseUrl = options.deviceCommunicationBaseUrl;
        }
        
        // Test if device communication API is available
        try {
            const response = await fetch(`${this.baseUrl}/ping`);
            if (response.ok) {
                console.log('Device communication initialized successfully');
                
                // If autoStartPolling is enabled, start polling for sensor data
                if (options.autoStartPolling !== false) {
                    this.startSensorPolling(options.pollingRate || this.sensorPollingRate);
                }
                
                return { status: 'success', message: 'Device communication initialized' };
            } else {
                console.warn('Device communication API is not available');
                return { status: 'warning', message: 'Device communication API is not available' };
            }
        } catch (error) {
            console.error('Error initializing device communication:', error);
            return { status: 'error', message: 'Failed to initialize device communication', error: error.message };
        }
    }

    /**
     * Cleanup resources and disconnect
     */
    async cleanup() {
        this.stopSensorPolling();
        if (this.isConnected) {
            await this.disconnectSerialPort();
        }
        this.sensorUpdateCallbacks = [];
        console.log('Device communication resources cleaned up');
    }
}

// Create a global instance for easy access
window.DeviceCommunication = new DeviceCommunication();