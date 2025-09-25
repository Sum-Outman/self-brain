/**
 * DeviceCommunication - Class for handling device communication with backend APIs
 */
class DeviceCommunication {
    /**
     * Constructor for DeviceCommunication class
     */
    constructor() {
        this.baseUrl = '/api/device';
        this.serialConnection = null;
        this.isConnected = false;
        this.sensorUpdateCallbacks = [];
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
            const response = await fetch(`${this.baseUrl}/serial_disconnect`, {
                method: 'POST'
            });
            
            const data = await response.json();
            
            if (data.status === 'success') {
                this.isConnected = false;
                this.serialConnection = null;
            }
            
            return data;
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
     * Get current connection status
     * @returns {Object} Connection status information
     */
    getConnectionStatus() {
        return {
            isConnected: this.isConnected,
            connection: this.serialConnection
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
        
        // Test if device communication API is available
        try {
            const response = await fetch(`${this.baseUrl}/ping`);
            if (response.ok) {
                console.log('Device communication initialized successfully');
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
}

// Create a global instance for easy access
window.DeviceCommunication = new DeviceCommunication();