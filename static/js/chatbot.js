// static/js/chatbot.js
(function() {
    let chatbotConfig = window.chatbotConfig || {};
    
    function initChatbot(id, name, websiteUrl) {
        // Store config
        chatbotConfig = { id, name, websiteUrl };
        
        // Create chatbot elements
        createChatbotUI();
        
        // Add event listeners
        addEventListeners();
    }
    
    function createChatbotUI() {
        // Chatbot container
        const chatbotContainer = document.createElement('div');
        chatbotContainer.id = 'chatbot-container';
        chatbotContainer.className = 'chatbot-container';
        
        chatbotContainer.innerHTML = `
            <div class="chatbot-widget">
                <div class="chatbot-header">
                    <div class="chatbot-title">
                        <i class="fas fa-robot"></i>
                        <span>${chatbotConfig.name || 'Chat Assistant'}</span>
                    </div>
                    <button class="chatbot-close" id="chatbot-close">
                        <i class="fas fa-times"></i>
                    </button>
                </div>
                <div class="chatbot-messages" id="chatbot-messages">
                    <div class="message bot-message">
                        <div class="message-content">
                            Hello! I'm your ${chatbotConfig.name || 'assistant'}. How can I help you today?
                        </div>
                        <div class="message-time">${getCurrentTime()}</div>
                    </div>
                </div>
                <div class="chatbot-input-container">
                    <input type="text" class="chatbot-input" id="chatbot-input" placeholder="Type your message..." />
                    <button class="chatbot-send" id="chatbot-send">
                        <i class="fas fa-paper-plane"></i>
                    </button>
                </div>
            </div>
            <button class="chatbot-toggle" id="chatbot-toggle">
                <i class="fas fa-robot"></i>
            </button>
        `;
        
        document.body.appendChild(chatbotContainer);
        
        // Add CSS
        addChatbotStyles();
    }
    
    function addChatbotStyles() {
        const style = document.createElement('style');
        style.textContent = `
            .chatbot-container {
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            }
            
            .chatbot-toggle {
                position: fixed;
                bottom: 20px;
                right: 20px;
                width: 60px;
                height: 60px;
                border-radius: 50%;
                background: #4f46e5;
                color: white;
                border: none;
                cursor: pointer;
                box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
                z-index: 10000;
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 20px;
                transition: all 0.3s ease;
            }
            
            .chatbot-toggle:hover {
                transform: scale(1.1);
                box-shadow: 0 6px 16px rgba(0, 0, 0, 0.2);
            }
            
            .chatbot-widget {
                position: fixed;
                bottom: 90px;
                right: 20px;
                width: 350px;
                height: 450px;
                background: white;
                border-radius: 16px;
                box-shadow: 0 10px 25px rgba(0, 0, 0, 0.2);
                display: flex;
                flex-direction: column;
                z-index: 10000;
                overflow: hidden;
                transform: translateY(20px);
                opacity: 0;
                transition: all 0.3s ease;
            }
            
            .chatbot-widget.active {
                transform: translateY(0);
                opacity: 1;
            }
            
            .chatbot-header {
                background: #4f46e5;
                color: white;
                padding: 16px;
                display: flex;
                justify-content: space-between;
                align-items: center;
            }
            
            .chatbot-title {
                display: flex;
                align-items: center;
                gap: 8px;
                font-weight: 600;
            }
            
            .chatbot-close {
                background: none;
                border: none;
                color: white;
                cursor: pointer;
                font-size: 18px;
            }
            
            .chatbot-messages {
                flex: 1;
                padding: 16px;
                overflow-y: auto;
                background: #f8fafc;
            }
            
            .message {
                margin-bottom: 16px;
                display: flex;
                flex-direction: column;
            }
            
            .bot-message {
                align-items: flex-start;
            }
            
            .user-message {
                align-items: flex-end;
            }
            
            .message-content {
                max-width: 80%;
                padding: 12px 16px;
                border-radius: 18px;
                font-size: 14px;
                line-height: 1.4;
            }
            
            .bot-message .message-content {
                background: #e0e7ff;
                border-bottom-left-radius: 4px;
            }
            
            .user-message .message-content {
                background: #4f46e5;
                color: white;
                border-bottom-right-radius: 4px;
            }
            
            .message-time {
                font-size: 11px;
                color: #64748b;
                margin-top: 4px;
            }
            
            .user-message .message-time {
                text-align: right;
            }
            
            .chatbot-input-container {
                display: flex;
                padding: 12px;
                border-top: 1px solid #e2e8f0;
                background: white;
            }
            
            .chatbot-input {
                flex: 1;
                padding: 12px 16px;
                border: 1px solid #e2e8f0;
                border-radius: 24px;
                outline: none;
                font-size: 14px;
            }
            
            .chatbot-input:focus {
                border-color: #4f46e5;
            }
            
            .chatbot-send {
                background: #4f46e5;
                color: white;
                border: none;
                width: 40px;
                height: 40px;
                border-radius: 50%;
                margin-left: 8px;
                cursor: pointer;
                display: flex;
                align-items: center;
                justify-content: center;
            }
            
            .chatbot-send:hover {
                background: #4338ca;
            }
            
            @media (max-width: 480px) {
                .chatbot-widget {
                    width: calc(100% - 40px);
                    height: 70vh;
                    bottom: 90px;
                    right: 20px;
                }
            }
        `;
        
        document.head.appendChild(style);
    }
    
    function addEventListeners() {
        const toggleBtn = document.getElementById('chatbot-toggle');
        const closeBtn = document.getElementById('chatbot-close');
        const sendBtn = document.getElementById('chatbot-send');
        const input = document.getElementById('chatbot-input');
        const widget = document.querySelector('.chatbot-widget');
        
        toggleBtn.addEventListener('click', () => {
            widget.classList.toggle('active');
        });
        
        closeBtn.addEventListener('click', () => {
            widget.classList.remove('active');
        });
        
        sendBtn.addEventListener('click', sendMessage);
        input.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                sendMessage();
            }
        });
    }
    
    function sendMessage() {
        const input = document.getElementById('chatbot-input');
        const message = input.value.trim();
        
        if (!message) return;
        
        // Add user message to UI
        addMessage(message, 'user');
        input.value = '';
        
        // Simulate bot response (in real app, this would call your backend)
        setTimeout(() => {
            simulateBotResponse(message);
        }, 1000);
    }
    
    function addMessage(text, sender) {
        const messagesContainer = document.getElementById('chatbot-messages');
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${sender}-message`;
        
        messageDiv.innerHTML = `
            <div class="message-content">${text}</div>
            <div class="message-time">${getCurrentTime()}</div>
        `;
        
        messagesContainer.appendChild(messageDiv);
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
    }
    
    function simulateBotResponse(userMessage) {
        // This is where you'd call your actual backend API
        // For now, we'll simulate responses
        
        let response = "I understand your question about '" + userMessage.substring(0, 20) + "...'. ";
        response += "This is a simulated response. In a real implementation, I would use the trained data to provide accurate answers.";
        
        addMessage(response, 'bot');
    }
    
    function getCurrentTime() {
        return new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    }
    
    // Make function available globally
    window.initChatbot = initChatbot;
    
    // Auto-initialize if config exists
    if (window.chatbotConfig) {
        initChatbot(
            window.chatbotConfig.id,
            window.chatbotConfig.name,
            window.chatbotConfig.websiteUrl
        );
    }
})();