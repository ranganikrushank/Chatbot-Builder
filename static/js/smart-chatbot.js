// static/js/smart-chatbot.js
(function() {
    let chatbotConfig = window.chatbotConfig || {};
    let conversationHistory = [];
    
    function initChatbot(id, name, websiteUrl, apiUrl) {
        chatbotConfig = { id, name, websiteUrl, apiUrl };
        createChatbotUI();
        addEventListeners();
        loadChatHistory();
    }
    
    function createChatbotUI() {
        const container = document.createElement('div');
        container.id = 'enhanced-chatbot-container';
        container.className = 'enhanced-chatbot-container';
        
        container.innerHTML = `
            <div class="enhanced-chatbot-widget" id="enhanced-chatbot-widget">
                <!-- Header -->
                <div class="enhanced-chatbot-header">
                    <div class="enhanced-chatbot-header-left">
                        <div class="enhanced-chatbot-avatar">
                            <i class="fas fa-robot"></i>
                        </div>
                        <div class="enhanced-chatbot-header-info">
                            <div class="enhanced-chatbot-title">${chatbotConfig.name || 'AI Assistant'}</div>
                            <div class="enhanced-chatbot-status">
                                <span class="enhanced-chatbot-status-indicator"></span>
                                <span class="enhanced-chatbot-status-text">Online</span>
                            </div>
                        </div>
                    </div>
                    <div class="enhanced-chatbot-header-actions">
                        <button class="enhanced-chatbot-header-btn" id="enhanced-chatbot-minimize" title="Minimize">
                            <i class="fas fa-window-minimize"></i>
                        </button>
                        <button class="enhanced-chatbot-header-btn" id="enhanced-chatbot-close" title="Close">
                            <i class="fas fa-times"></i>
                        </button>
                    </div>
                </div>
                
                <!-- Messages Area -->
                <div class="enhanced-chatbot-messages" id="enhanced-chatbot-messages">
                    <div class="enhanced-chatbot-welcome">
                        <div class="enhanced-chatbot-welcome-avatar">
                            <i class="fas fa-robot"></i>
                        </div>
                        <div class="enhanced-chatbot-welcome-content">
                            <h3>Hello! 👋</h3>
                            <p>I'm your ${chatbotConfig.name || 'AI Assistant'}. I can help you with questions about the content you've provided.</p>
                            
                        </div>
                    </div>
                </div>
                
                <!-- Typing Indicator -->
                <div class="enhanced-chatbot-typing" id="enhanced-chatbot-typing" style="display: none;">
                    <div class="enhanced-chatbot-typing-avatar">
                        <i class="fas fa-robot"></i>
                    </div>
                    <div class="enhanced-chatbot-typing-indicator">
                        <span></span>
                        <span></span>
                        <span></span>
                    </div>
                </div>
                
                <!-- Input Area -->
                <div class="enhanced-chatbot-input-container">
                    <div class="enhanced-chatbot-input-wrapper">
                        <input type="text" class="enhanced-chatbot-input" id="enhanced-chatbot-input" 
                               placeholder="Type your message..." autocomplete="off" />
                        <button class="enhanced-chatbot-send" id="enhanced-chatbot-send" disabled>
                            <i class="fas fa-paper-plane"></i>
                        </button>
                    </div>
                    <div class="enhanced-chatbot-input-hints">
                        <span>Press Enter to send</span>
                        <span>•</span>
                        <span>Shift+Enter for new line</span>
                    </div>
                </div>
            </div>
            
            <!-- Floating Toggle Button -->
            <button class="enhanced-chatbot-toggle" id="enhanced-chatbot-toggle">
                <div class="enhanced-chatbot-toggle-badge" id="enhanced-chatbot-toggle-badge">1</div>
                <i class="fas fa-comment-dots"></i>
            </button>
        `;
        
        document.body.appendChild(container);
        addChatbotStyles();
    }
    
    function addChatbotStyles() {
        const style = document.createElement('style');
        style.textContent = `
            /* Enhanced Chatbot Styles */
            .enhanced-chatbot-container {
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
                position: fixed;
                bottom: 20px;
                right: 20px;
                z-index: 2147483647;
                font-size: 14px;
                line-height: 1.5;
            }
            
            /* Floating Toggle Button */
            .enhanced-chatbot-toggle {
                position: fixed;
                bottom: 20px;
                right: 20px;
                width: 60px;
                height: 60px;
                border-radius: 50%;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                border: none;
                cursor: pointer;
                box-shadow: 0 6px 20px rgba(0, 0, 0, 0.25);
                z-index: 2147483647;
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 24px;
                transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
                animation: enhanced-pulse 2s infinite;
            }
            
            .enhanced-chatbot-toggle:hover {
                transform: scale(1.1);
                box-shadow: 0 8px 25px rgba(0, 0, 0, 0.3);
                animation: none;
            }
            
            .enhanced-chatbot-toggle-badge {
                position: absolute;
                top: -5px;
                right: -5px;
                background: #ef4444;
                color: white;
                border-radius: 50%;
                width: 20px;
                height: 20px;
                font-size: 12px;
                display: flex;
                align-items: center;
                justify-content: center;
                font-weight: bold;
                font-family: Arial, sans-serif;
            }
            
            /* Main Widget */
            .enhanced-chatbot-widget {
                position: fixed;
                bottom: 90px;
                right: 20px;
                width: 380px;
                height: 520px;
                background: white;
                border-radius: 20px;
                box-shadow: 0 20px 40px rgba(0, 0, 0, 0.15);
                display: flex;
                flex-direction: column;
                z-index: 2147483646;
                overflow: hidden;
                transform: translateY(20px);
                opacity: 0;
                transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
                border: 1px solid rgba(0, 0, 0, 0.05);
            }
            
            .enhanced-chatbot-widget.active {
                transform: translateY(0);
                opacity: 1;
            }
            
            /* Header */
            .enhanced-chatbot-header {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 16px;
                display: flex;
                justify-content: space-between;
                align-items: center;
                border-radius: 20px 20px 0 0;
            }
            
            .enhanced-chatbot-header-left {
                display: flex;
                align-items: center;
                gap: 12px;
            }
            
            .enhanced-chatbot-avatar {
                width: 36px;
                height: 36px;
                border-radius: 50%;
                background: rgba(255, 255, 255, 0.2);
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 16px;
            }
            
            .enhanced-chatbot-header-info {
                display: flex;
                flex-direction: column;
            }
            
            .enhanced-chatbot-title {
                font-weight: 600;
                font-size: 16px;
            }
            
            .enhanced-chatbot-status {
                display: flex;
                align-items: center;
                gap: 6px;
                font-size: 12px;
                opacity: 0.9;
            }
            
            .enhanced-chatbot-status-indicator {
                width: 8px;
                height: 8px;
                border-radius: 50%;
                background: #10b981;
                box-shadow: 0 0 0 2px rgba(16, 185, 129, 0.3);
            }
            
            .enhanced-chatbot-header-actions {
                display: flex;
                gap: 8px;
            }
            
            .enhanced-chatbot-header-btn {
                background: rgba(255, 255, 255, 0.2);
                border: none;
                color: white;
                cursor: pointer;
                width: 32px;
                height: 32px;
                border-radius: 50%;
                display: flex;
                align-items: center;
                justify-content: center;
                transition: background 0.2s;
            }
            
            .enhanced-chatbot-header-btn:hover {
                background: rgba(255, 255, 255, 0.3);
            }
            
            /* Messages Area */
            .enhanced-chatbot-messages {
                flex: 1;
                padding: 20px;
                overflow-y: auto;
                background: #f8fafc;
                display: flex;
                flex-direction: column;
                gap: 16px;
            }
            
            /* Welcome Message */
            .enhanced-chatbot-welcome {
                display: flex;
                gap: 12px;
                margin-bottom: 16px;
                animation: enhanced-fadeIn 0.5s ease-out;
            }
            
            .enhanced-chatbot-welcome-avatar {
                width: 36px;
                height: 36px;
                border-radius: 50%;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 16px;
                flex-shrink: 0;
            }
            
            .enhanced-chatbot-welcome-content h3 {
                font-size: 16px;
                font-weight: 600;
                color: #1e293b;
                margin: 0 0 8px 0;
            }
            
            .enhanced-chatbot-welcome-content p {
                font-size: 14px;
                color: #64748b;
                margin: 0 0 16px 0;
                line-height: 1.5;
            }
            
            .enhanced-chatbot-suggestions {
                display: flex;
                flex-direction: column;
                gap: 8px;
            }
            
            .enhanced-chatbot-suggestion {
                background: white;
                border: 1px solid #e2e8f0;
                border-radius: 16px;
                padding: 8px 12px;
                font-size: 13px;
                color: #475569;
                text-align: left;
                cursor: pointer;
                transition: all 0.2s;
                box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
            }
            
            .enhanced-chatbot-suggestion:hover {
                background: #f1f5f9;
                border-color: #cbd5e1;
                transform: translateY(-1px);
                box-shadow: 0 2px 6px rgba(0, 0, 0, 0.1);
            }
            
            /* Individual Messages */
            .enhanced-chatbot-message {
                display: flex;
                gap: 12px;
                animation: enhanced-messageAppear 0.3s ease-out;
                max-width: 85%;
            }
            
            .enhanced-chatbot-message-avatar {
                width: 32px;
                height: 32px;
                border-radius: 50%;
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 14px;
                flex-shrink: 0;
            }
            
            .enhanced-chatbot-bot .enhanced-chatbot-message-avatar {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
            }
            
            .enhanced-chatbot-user .enhanced-chatbot-message-avatar {
                background: #4f46e5;
                color: white;
            }
            
            .enhanced-chatbot-message-content {
                display: flex;
                flex-direction: column;
                max-width: 80%;
            }
            
            .enhanced-chatbot-message-text {
                padding: 12px 16px;
                border-radius: 18px;
                font-size: 14px;
                line-height: 1.5;
                box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
                word-wrap: break-word;
            }
            
            .enhanced-chatbot-bot .enhanced-chatbot-message-text {
                background: white;
                border: 1px solid #e2e8f0;
                border-bottom-left-radius: 4px;
                color: #334155;
            }
            
            .enhanced-chatbot-user .enhanced-chatbot-message-text {
                background: #4f46e5;
                color: white;
                border-bottom-right-radius: 4px;
                margin-left: auto;
            }
            
            .enhanced-chatbot-message-time {
                font-size: 11px;
                color: #64748b;
                margin-top: 4px;
                margin-left: 8px;
            }
            
            .enhanced-chatbot-user .enhanced-chatbot-message-time {
                text-align: right;
                margin-left: auto;
                margin-right: 8px;
            }
            
            /* Typing Indicator */
            .enhanced-chatbot-typing {
                display: flex;
                align-items: flex-start;
                gap: 12px;
                padding: 0 20px 16px 20px;
                background: #f8fafc;
            }
            
            .enhanced-chatbot-typing-avatar {
                width: 32px;
                height: 32px;
                border-radius: 50%;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 14px;
                flex-shrink: 0;
            }
            
            .enhanced-chatbot-typing-indicator {
                display: flex;
                align-items: center;
                gap: 4px;
                padding: 12px 16px;
                background: white;
                border: 1px solid #e2e8f0;
                border-radius: 18px;
                border-bottom-left-radius: 4px;
            }
            
            .enhanced-chatbot-typing-indicator span {
                width: 8px;
                height: 8px;
                background: #4f46e5;
                border-radius: 50%;
                display: inline-block;
                animation: enhanced-typing 1.4s infinite ease-in-out;
            }
            
            .enhanced-chatbot-typing-indicator span:nth-child(1) {
                animation-delay: -0.32s;
            }
            
            .enhanced-chatbot-typing-indicator span:nth-child(2) {
                animation-delay: -0.16s;
            }
            
            /* Input Area */
            .enhanced-chatbot-input-container {
                display: flex;
                flex-direction: column;
                padding: 16px;
                border-top: 1px solid #e2e8f0;
                background: white;
            }
            
            .enhanced-chatbot-input-wrapper {
                display: flex;
                gap: 8px;
                margin-bottom: 8px;
            }
            
            .enhanced-chatbot-input {
                flex: 1;
                padding: 12px 16px;
                border: 2px solid #e2e8f0;
                border-radius: 24px;
                outline: none;
                font-size: 14px;
                transition: border-color 0.2s;
                resize: none;
                min-height: 20px;
                max-height: 100px;
                overflow-y: auto;
            }
            
            .enhanced-chatbot-input:focus {
                border-color: #4f46e5;
                box-shadow: 0 0 0 2px rgba(79, 70, 229, 0.1);
            }
            
            .enhanced-chatbot-send {
                background: #4f46e5;
                color: white;
                border: none;
                width: 44px;
                height: 44px;
                border-radius: 50%;
                cursor: pointer;
                display: flex;
                align-items: center;
                justify-content: center;
                transition: all 0.2s;
                box-shadow: 0 2px 8px rgba(79, 70, 229, 0.3);
            }
            
            .enhanced-chatbot-send:hover:not(:disabled) {
                background: #4338ca;
                transform: translateY(-1px);
                box-shadow: 0 4px 12px rgba(79, 70, 229, 0.4);
            }
            
            .enhanced-chatbot-send:disabled {
                background: #c7d2fe;
                cursor: not-allowed;
                box-shadow: none;
            }
            
            .enhanced-chatbot-input-hints {
                display: flex;
                justify-content: center;
                gap: 4px;
                font-size: 11px;
                color: #94a3b8;
            }
            
            /* Animations */
            @keyframes enhanced-pulse {
                0% {
                    box-shadow: 0 0 0 0 rgba(102, 126, 234, 0.4);
                }
                70% {
                    box-shadow: 0 0 0 10px rgba(102, 126, 234, 0);
                }
                100% {
                    box-shadow: 0 0 0 0 rgba(102, 126, 234, 0);
                }
            }
            
            @keyframes enhanced-messageAppear {
                from {
                    opacity: 0;
                    transform: translateY(10px);
                }
                to {
                    opacity: 1;
                    transform: translateY(0);
                }
            }
            
            @keyframes enhanced-fadeIn {
                from {
                    opacity: 0;
                    transform: translateY(10px);
                }
                to {
                    opacity: 1;
                    transform: translateY(0);
                }
            }
            
            @keyframes enhanced-typing {
                0%, 80%, 100% {
                    transform: scale(0);
                }
                40% {
                    transform: scale(1);
                }
            }
            
            /* Scrollbar Styling */
            .enhanced-chatbot-messages::-webkit-scrollbar {
                width: 6px;
            }
            
            .enhanced-chatbot-messages::-webkit-scrollbar-track {
                background: #f1f5f9;
                border-radius: 3px;
            }
            
            .enhanced-chatbot-messages::-webkit-scrollbar-thumb {
                background: #cbd5e1;
                border-radius: 3px;
            }
            
            .enhanced-chatbot-messages::-webkit-scrollbar-thumb:hover {
                background: #94a3b8;
            }
            
            /* Responsive Design */
            @media (max-width: 480px) {
                .enhanced-chatbot-widget {
                    width: calc(100% - 40px);
                    height: 70vh;
                    bottom: 90px;
                    right: 20px;
                }
            }
            
            /* Dark Mode Support */
            @media (prefers-color-scheme: dark) {
                .enhanced-chatbot-messages {
                    background: #0f172a;
                }
                
                .enhanced-chatbot-welcome-content h3 {
                    color: #f1f5f9;
                }
                
                .enhanced-chatbot-welcome-content p {
                    color: #94a3b8;
                }
                
                .enhanced-chatbot-suggestion {
                    background: #1e293b;
                    border-color: #334155;
                    color: #e2e8f0;
                }
                
                .enhanced-chatbot-suggestion:hover {
                    background: #334155;
                    border-color: #475569;
                }
                
                .enhanced-chatbot-input {
                    background: #1e293b;
                    border-color: #334155;
                    color: #f1f5f9;
                }
                
                .enhanced-chatbot-input:focus {
                    border-color: #4f46e5;
                    background: #1e293b;
                }
                
                .enhanced-chatbot-input-container {
                    background: #0f172a;
                    border-top-color: #1e293b;
                }
                
                .enhanced-chatbot-bot .enhanced-chatbot-message-text {
                    background: #1e293b;
                    border-color: #334155;
                    color: #f1f5f9;
                }
            }
        `;
        
        document.head.appendChild(style);
    }
    
    function addEventListeners() {
        const toggleBtn = document.getElementById('enhanced-chatbot-toggle');
        const closeBtn = document.getElementById('enhanced-chatbot-close');
        const minimizeBtn = document.getElementById('enhanced-chatbot-minimize');
        const sendBtn = document.getElementById('enhanced-chatbot-send');
        const input = document.getElementById('enhanced-chatbot-input');
        const widget = document.getElementById('enhanced-chatbot-widget');
        const suggestions = document.querySelectorAll('.enhanced-chatbot-suggestion');
        
        toggleBtn.addEventListener('click', () => {
            widget.classList.toggle('active');
            if (widget.classList.contains('active')) {
                document.getElementById('enhanced-chatbot-toggle-badge').style.display = 'none';
                input.focus();
            }
        });
        
        closeBtn.addEventListener('click', () => {
            widget.classList.remove('active');
        });
        
        minimizeBtn.addEventListener('click', () => {
            widget.classList.remove('active');
        });
        
        sendBtn.addEventListener('click', sendMessage);
        
        input.addEventListener('keypress', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                if (!sendBtn.disabled) {
                    sendMessage();
                }
            }
        });
        
        input.addEventListener('input', () => {
            sendBtn.disabled = !input.value.trim();
            adjustInputHeight();
        });
        
        // Add suggestion click handlers
        suggestions.forEach(suggestion => {
            suggestion.addEventListener('click', () => {
                const message = suggestion.getAttribute('data-message');
                sendMessageWithText(message);
            });
        });
        
        // Close widget when clicking outside
        document.addEventListener('click', (e) => {
            if (!widget.contains(e.target) && !toggleBtn.contains(e.target) && widget.classList.contains('active')) {
                widget.classList.remove('active');
            }
        });
    }
    
    function adjustInputHeight() {
        const input = document.getElementById('enhanced-chatbot-input');
        input.style.height = 'auto';
        input.style.height = Math.min(input.scrollHeight, 100) + 'px';
    }
    
    async function sendMessage() {
        const input = document.getElementById('enhanced-chatbot-input');
        const message = input.value.trim();
        
        if (!message) return;
        
        sendMessageWithText(message);
    }
    
    async function sendMessageWithText(message) {
        const input = document.getElementById('enhanced-chatbot-input');
        const sendBtn = document.getElementById('enhanced-chatbot-send');
        
        // Add user message to UI
        addMessage(message, 'user');
        input.value = '';
        sendBtn.disabled = true;
        adjustInputHeight();
        
        // Show typing indicator
        showTypingIndicator();
        
        try {
            // Send to backend for intelligent processing
            const response = await fetch(`${chatbotConfig.apiUrl || 'http://localhost:5000'}/chat/${chatbotConfig.id}`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    message: message,
                    history: conversationHistory
                })
            });
            
            const data = await response.json();
            
            // Hide typing indicator
            hideTypingIndicator();
            
            if (data.success) {
                addMessage(data.response, 'bot');
            } else {
                addMessage('Sorry, I encountered an error processing your request.', 'bot');
            }
        } catch (error) {
            console.error('Chat error:', error);
            hideTypingIndicator();
            addMessage('Sorry, I\'m having trouble connecting right now. Please try again.', 'bot');
        }
        
        sendBtn.disabled = false;
        input.focus();
    }
    
    function addMessage(text, sender) {
        const messagesContainer = document.getElementById('enhanced-chatbot-messages');
        const messageDiv = document.createElement('div');
        messageDiv.className = `enhanced-chatbot-message enhanced-chatbot-${sender}-message`;
        
        const avatarIcon = sender === 'bot' ? 'fa-robot' : 'fa-user';
        
        messageDiv.innerHTML = `
            <div class="enhanced-chatbot-message-avatar">
                <i class="fas ${avatarIcon}"></i>
            </div>
            <div class="enhanced-chatbot-message-content">
                <div class="enhanced-chatbot-message-text">${escapeHtml(text)}</div>
                <div class="enhanced-chatbot-message-time">${getCurrentTime()}</div>
            </div>
        `;
        
        // Remove welcome message if this is the first user message
        const welcomeMessage = document.querySelector('.enhanced-chatbot-welcome');
        if (welcomeMessage && sender === 'user') {
            welcomeMessage.remove();
        }
        
        messagesContainer.appendChild(messageDiv);
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
        
        // Add to conversation history
        conversationHistory.push({
            sender: sender,
            message: text,
            timestamp: Date.now()
        });
        
        // Keep only last 10 messages for context
        if (conversationHistory.length > 10) {
            conversationHistory.shift();
        }
        
        // Save to localStorage
        saveChatHistory();
    }
    
    function showTypingIndicator() {
        document.getElementById('enhanced-chatbot-typing').style.display = 'flex';
        const messagesContainer = document.getElementById('enhanced-chatbot-messages');
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
    }
    
    function hideTypingIndicator() {
        document.getElementById('enhanced-chatbot-typing').style.display = 'none';
    }
    
    function getCurrentTime() {
        return new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    }
    
    function escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
    
    function saveChatHistory() {
        try {
            localStorage.setItem(`chatbot_${chatbotConfig.id}_history`, JSON.stringify(conversationHistory));
        } catch (e) {
            console.warn('Could not save chat history to localStorage');
        }
    }
    
    function loadChatHistory() {
        try {
            const saved = localStorage.getItem(`chatbot_${chatbotConfig.id}_history`);
            if (saved) {
                conversationHistory = JSON.parse(saved);
                // Replay conversation (optional)
            }
        } catch (e) {
            console.warn('Could not load chat history from localStorage');
        }
    }
    
    // Make function available globally
    window.initChatbot = initChatbot;
    
    // Auto-initialize if config exists
    if (window.chatbotConfig) {
        // Small delay to ensure DOM is ready
        setTimeout(() => {
            initChatbot(
                window.chatbotConfig.id,
                window.chatbotConfig.name,
                window.chatbotConfig.websiteUrl,
                window.chatbotConfig.apiUrl
            );
        }, 100);
    }
})();