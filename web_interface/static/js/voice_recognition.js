// Voice Recognition Module
// Self Brain AGI

class VoiceRecognition {
    constructor() {
        this.recognition = null;
        this.isRecognizing = false;
        this.isSupported = false;
        this.lastResult = '';
        this.lang = 'en-US';
        this.continuous = true;
        this.interimResults = true;
        this.onResultCallback = null;
        this.onErrorCallback = null;
        this.onStartCallback = null;
        this.onEndCallback = null;
    }

    // Initialize the voice recognition module
    init() {
        console.log('Voice Recognition module initialized');
        
        // Check if the browser supports speech recognition
        if ('webkitSpeechRecognition' in window) {
            this.recognition = new webkitSpeechRecognition();
            this.isSupported = true;
        } else if ('SpeechRecognition' in window) {
            this.recognition = new SpeechRecognition();
            this.isSupported = true;
        } else {
            console.warn('Speech recognition is not supported in this browser');
            this.isSupported = false;
            return;
        }

        // Configure the recognition instance
        this.configureRecognition();
        this.setupEventListeners();

        console.log('Voice Recognition is supported and ready');
    }

    // Configure the speech recognition instance
    configureRecognition() {
        if (!this.recognition) return;
        
        this.recognition.lang = this.lang;
        this.recognition.continuous = this.continuous;
        this.recognition.interimResults = this.interimResults;
        this.recognition.maxAlternatives = 1;
    }

    // Setup event listeners for the recognition instance
    setupEventListeners() {
        if (!this.recognition) return;

        // Result event handler
        this.recognition.onresult = (event) => {
            let finalTranscript = '';
            let interimTranscript = '';

            // Process all results
            for (let i = event.resultIndex; i < event.results.length; i++) {
                const transcript = event.results[i][0].transcript;
                if (event.results[i].isFinal) {
                    finalTranscript += transcript + ' ';
                } else {
                    interimTranscript += transcript;
                }
            }

            this.lastResult = finalTranscript || interimTranscript;

            // Update UI elements if they exist
            const voiceInput = document.getElementById('voice-input');
            if (voiceInput) {
                voiceInput.value = this.lastResult;
            }

            // Call the result callback if provided
            if (this.onResultCallback) {
                this.onResultCallback(this.lastResult, !!finalTranscript);
            }
        };

        // Error event handler
        this.recognition.onerror = (event) => {
            console.error('Speech recognition error:', event.error);
            
            // Handle different error types
            let errorMessage = 'Speech recognition error';
            switch (event.error) {
                case 'no-speech':
                    errorMessage = 'No speech detected';
                    break;
                case 'audio-capture':
                    errorMessage = 'No microphone available';
                    break;
                case 'not-allowed':
                    errorMessage = 'Microphone permission denied';
                    break;
                case 'aborted':
                    errorMessage = 'Speech recognition aborted';
                    break;
                default:
                    errorMessage = `Speech error: ${event.error}`;
            }

            // Update UI with error message
            this.updateStatus(errorMessage, 'error');

            // Call the error callback if provided
            if (this.onErrorCallback) {
                this.onErrorCallback(event.error);
            }

            // Automatically restart after certain errors
            if (event.error === 'no-speech' || event.error === 'audio-capture') {
                setTimeout(() => {
                    if (this.isRecognizing) {
                        this.startRecognition();
                    }
                }, 1000);
            }
        };

        // Start event handler
        this.recognition.onstart = () => {
            this.isRecognizing = true;
            this.updateStatus('Listening...', 'listening');
            
            // Call the start callback if provided
            if (this.onStartCallback) {
                this.onStartCallback();
            }
        };

        // End event handler
        this.recognition.onend = () => {
            this.isRecognizing = false;
            this.updateStatus('Not listening', 'idle');
            
            // Call the end callback if provided
            if (this.onEndCallback) {
                this.onEndCallback();
            }
        };

        // Sound start event handler
        this.recognition.onsoundstart = () => {
            this.updateStatus('Processing sound...', 'processing');
        };

        // Sound end event handler
        this.recognition.onsoundend = () => {
            this.updateStatus('Processing result...', 'processing');
        };
    }

    // Start speech recognition
    startRecognition() {
        if (!this.isSupported || !this.recognition) {
            console.warn('Speech recognition is not supported');
            return false;
        }

        if (this.isRecognizing) {
            console.warn('Speech recognition is already running');
            return false;
        }

        try {
            this.recognition.start();
            return true;
        } catch (error) {
            console.error('Failed to start speech recognition:', error);
            return false;
        }
    }

    // Stop speech recognition
    stopRecognition() {
        if (!this.isSupported || !this.recognition || !this.isRecognizing) {
            return false;
        }

        try {
            this.recognition.stop();
            return true;
        } catch (error) {
            console.error('Failed to stop speech recognition:', error);
            return false;
        }
    }

    // Abort speech recognition immediately
    abortRecognition() {
        if (!this.isSupported || !this.recognition || !this.isRecognizing) {
            return false;
        }

        try {
            this.recognition.abort();
            return true;
        } catch (error) {
            console.error('Failed to abort speech recognition:', error);
            return false;
        }
    }

    // Update the status UI elements
    updateStatus(message, statusType) {
        // Update voice input button if it exists
        const voiceBtn = document.getElementById('voice-btn');
        if (voiceBtn) {
            // Change button icon based on status
            if (statusType === 'listening') {
                voiceBtn.innerHTML = '<i class="bi bi-mic-fill"></i> Listening';
                voiceBtn.classList.add('listening');
            } else {
                voiceBtn.innerHTML = '<i class="bi bi-mic"></i> Voice';
                voiceBtn.classList.remove('listening');
            }
        }

        // Update recognition status indicator if it exists
        const recognitionStatus = document.getElementById('recognition-status');
        if (recognitionStatus) {
            recognitionStatus.textContent = message;
            recognitionStatus.className = `recognition-status ${statusType}`;
        }
    }

    // Set the language for speech recognition
    setLanguage(lang) {
        if (!this.recognition) return false;
        
        this.lang = lang;
        this.recognition.lang = lang;
        return true;
    }

    // Toggle continuous recognition mode
    setContinuous(continuous) {
        if (!this.recognition) return false;
        
        this.continuous = continuous;
        this.recognition.continuous = continuous;
        return true;
    }

    // Set callback functions
    setCallbacks(callbacks) {
        if (callbacks.onResult) this.onResultCallback = callbacks.onResult;
        if (callbacks.onError) this.onErrorCallback = callbacks.onError;
        if (callbacks.onStart) this.onStartCallback = callbacks.onStart;
        if (callbacks.onEnd) this.onEndCallback = callbacks.onEnd;
    }

    // Get the last recognized text
    getLastResult() {
        return this.lastResult;
    }

    // Get the recognition status
    isRunning() {
        return this.isRecognizing;
    }

    // Get supported status
    isSupported() {
        return this.isSupported;
    }

    // Simulate voice recognition for testing
    simulateVoiceRecognition(text) {
        if (this.onResultCallback) {
            this.lastResult = text;
            this.onResultCallback(text, true);
            
            // Update UI elements
            const voiceInput = document.getElementById('voice-input');
            if (voiceInput) {
                voiceInput.value = text;
            }
        }
    }

    // Cleanup
    destroy() {
        if (this.recognition) {
            this.stopRecognition();
            this.recognition = null;
        }
        
        this.isRecognizing = false;
        this.isSupported = false;
        this.lastResult = '';
    }
}

// Global functions for quick access
window.startVoiceConversation = function() {
    if (window.voiceRecognition) {
        if (window.voiceRecognition.isRunning()) {
            window.voiceRecognition.stopRecognition();
        } else {
            window.voiceRecognition.startRecognition();
        }
    }
};

window.startVoiceRecognition = function() {
    if (window.voiceRecognition) {
        window.voiceRecognition.startRecognition();
    }
};

// Initialize the voice recognition when selfBrain is available
if (window.selfBrain && window.selfBrain.init) {
    // Wait for selfBrain to be initialized
    const waitForSelfBrain = setInterval(() => {
        if (window.selfBrain && window.selfBrain.state) {
            clearInterval(waitForSelfBrain);
            window.voiceRecognition = new VoiceRecognition();
            window.voiceRecognition.init();
            
            // Set up callbacks for voice recognition
            window.voiceRecognition.setCallbacks({
                onResult: (text, isFinal) => {
                    if (isFinal && text.trim()) {
                        // Send recognized text as a message if it's final
                        window.sendConversationMessage(text);
                    }
                },
                onError: (error) => {
                    console.error('Voice recognition error:', error);
                }
            });
        }
    }, 100);
}