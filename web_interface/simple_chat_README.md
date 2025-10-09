# Simple AI Chat Interface

A minimalist, clean interface for interacting with the AI system using basic chat functionality.

## Features

- ✅ Clean, minimalist black and white interface
- ✅ Real-time message sending and receiving
- ✅ Typing indicator to show when AI is responding
- ✅ Connection to the main AI system
- ✅ Responsive design that works on different screen sizes
- ✅ English-only interface

## Prerequisites

Before running the simple chat interface, make sure you have:

1. Python 3.6 or higher installed
2. Flask library installed (`pip install flask requests`)
3. The main AI system running on `localhost:5015`

## How to Run

1. Open a command prompt or terminal
2. Navigate to the `web_interface` directory
3. Run the following command:
   
   ```
   python run_simple_chat.py
   ```
4. Open a web browser and go to `http://localhost:5001`

## Usage

1. Type your message in the text input area
2. Click the "Send" button or press Enter to send your message
3. Wait for the AI to respond
4. Continue the conversation as needed

## Troubleshooting

If you encounter any issues:

- Make sure the main AI system is running on `localhost:5015`
- Check that you have the required dependencies installed
- Verify that port 5001 is not being used by another application
- Check the command prompt for any error messages

## Technical Details

- The interface is built using HTML, CSS, and JavaScript
- It uses Bootstrap for basic styling and responsiveness
- The backend is a simple Flask application
- Messages are sent to the main AI system via HTTP POST requests
- The application runs on port 5001 by default

## Customization

You can modify the following aspects of the interface:

- Change the colors and styling in the CSS section of `simple_chat_interface.html`
- Adjust the port number in `run_simple_chat.py`
- Modify the timeout settings or error messages in `run_simple_chat.py`