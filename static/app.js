// ---- State ----
let currentSessionId = null;
let selectedFiles = [];

// ---- Init ----
document.addEventListener('DOMContentLoaded', () => {
    initTabs();
    initDropZone();
    initUpload();
    initChat();
    loadStats();
    newSession();
});

// ---- Tabs ----
function initTabs() {
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
            btn.classList.add('active');
            document.getElementById(btn.dataset.tab + '-tab').classList.add('active');
        });
    });
}

// ---- Stats ----
async function loadStats() {
    try {
        const r = await fetch('/api/stats');
        const d = await r.json();
        document.getElementById('header-stats').textContent =
            `CLIP: ${d.clip_count} | BGE: ${d.text_count}`;
    } catch {}
}

// ---- Upload ----
function initDropZone() {
    const zone = document.getElementById('drop-zone');
    const input = document.getElementById('file-input');

    zone.addEventListener('click', () => input.click());
    zone.addEventListener('dragover', e => { e.preventDefault(); zone.classList.add('drag-over'); });
    zone.addEventListener('dragleave', () => zone.classList.remove('drag-over'));
    zone.addEventListener('drop', e => {
        e.preventDefault();
        zone.classList.remove('drag-over');
        addFiles(e.dataTransfer.files);
    });
    input.addEventListener('change', () => {
        addFiles(input.files);
        input.value = '';
    });
}

function addFiles(fileList) {
    for (const f of fileList) {
        if (!selectedFiles.find(sf => sf.name === f.name)) {
            selectedFiles.push(f);
        }
    }
    renderFileList();
}

function renderFileList() {
    const container = document.getElementById('file-list');
    const btn = document.getElementById('upload-btn');
    container.innerHTML = '';

    selectedFiles.forEach((f, i) => {
        const div = document.createElement('div');
        div.className = 'file-item';
        div.innerHTML = `
            <span>${f.name} (${formatSize(f.size)})</span>
            <button class="remove-btn" data-index="${i}">&times;</button>
        `;
        container.appendChild(div);
    });

    container.querySelectorAll('.remove-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            selectedFiles.splice(parseInt(btn.dataset.index), 1);
            renderFileList();
        });
    });

    btn.disabled = selectedFiles.length === 0;
}

function initUpload() {
    document.getElementById('upload-btn').addEventListener('click', uploadFiles);
}

async function uploadFiles() {
    if (selectedFiles.length === 0) return;

    const btn = document.getElementById('upload-btn');
    const text = btn.querySelector('.btn-text');
    const spinner = btn.querySelector('.btn-spinner');
    const resultsDiv = document.getElementById('upload-results');

    btn.disabled = true;
    text.textContent = '处理中...';
    spinner.classList.remove('hidden');

    const formData = new FormData();
    selectedFiles.forEach(f => formData.append('files', f));

    try {
        const r = await fetch('/api/upload', { method: 'POST', body: formData });
        const data = await r.json();

        resultsDiv.innerHTML = '';
        data.results.forEach(item => {
            const div = document.createElement('div');
            div.className = 'result-item ' + item.status;
            div.textContent = item.status === 'ok'
                ? `${item.filename} - 处理成功${item.detail ? ' (' + item.detail + ')' : ''}`
                : `${item.filename} - 失败: ${item.detail}`;
            resultsDiv.appendChild(div);
        });

        selectedFiles = [];
        renderFileList();
        loadStats();
    } catch (e) {
        resultsDiv.innerHTML = `<div class="result-item error">上传失败: ${e.message}</div>`;
    } finally {
        btn.disabled = false;
        text.textContent = '上传并处理';
        spinner.classList.add('hidden');
    }
}

// ---- Chat ----
function initChat() {
    document.getElementById('chat-form').addEventListener('submit', e => {
        e.preventDefault();
        askQuestion();
    });

    document.getElementById('new-session-btn').addEventListener('click', () => {
        newSession();
        document.getElementById('chat-messages').innerHTML =
            '<div class="welcome-msg"><p>欢迎使用多模态 RAG 问答系统！</p><p>请先上传文档，然后在此提问。</p></div>';
    });

    const historyBtn = document.getElementById('history-btn');
    const dropdown = document.getElementById('session-list');

    historyBtn.addEventListener('click', async () => {
        dropdown.classList.toggle('show');
        if (dropdown.classList.contains('show')) {
            await loadSessionList();
        }
    });

    document.addEventListener('click', e => {
        if (!e.target.closest('.session-dropdown')) {
            dropdown.classList.remove('show');
        }
    });
}

async function newSession() {
    try {
        const r = await fetch('/api/session/new', { method: 'POST' });
        const d = await r.json();
        currentSessionId = d.session_id;
        document.getElementById('session-id').textContent = currentSessionId.substring(0, 8) + '...';
    } catch {
        document.getElementById('session-id').textContent = '创建失败';
    }
}

async function loadSessionList() {
    const dropdown = document.getElementById('session-list');
    try {
        const r = await fetch('/api/sessions');
        const sessions = await r.json();
        dropdown.innerHTML = '';

        if (sessions.length === 0) {
            dropdown.innerHTML = '<div class="dropdown-item"><span class="meta">暂无历史会话</span></div>';
            return;
        }

        sessions.reverse().forEach(s => {
            const div = document.createElement('div');
            div.className = 'dropdown-item';
            div.innerHTML = `
                <div>${s.session_id.substring(0, 8)}...</div>
                <div class="meta">${s.created_at} | ${s.message_count} 条消息</div>
            `;
            div.addEventListener('click', () => loadSession(s.session_id));
            dropdown.appendChild(div);
        });
    } catch {}
}

async function loadSession(sessionId) {
    try {
        const r = await fetch(`/api/session/${sessionId}`);
        const d = await r.json();
        currentSessionId = d.session_id;
        document.getElementById('session-id').textContent = currentSessionId.substring(0, 8) + '...';
        document.getElementById('session-list').classList.remove('show');

        const container = document.getElementById('chat-messages');
        container.innerHTML = '';

        d.messages.forEach(msg => {
            if (msg.type === 'question') {
                appendMessage('user', msg.content);
            } else if (msg.type === 'answer') {
                appendMessage('assistant', msg.content, msg.sources);
            }
        });

        if (d.messages.length === 0) {
            container.innerHTML = '<div class="welcome-msg"><p>此会话暂无消息</p></div>';
        }
    } catch {}
}

async function askQuestion() {
    const input = document.getElementById('question-input');
    const question = input.value.trim();
    if (!question) return;

    input.value = '';
    appendMessage('user', question);

    const sendBtn = document.getElementById('send-btn');
    const btnText = sendBtn.querySelector('.btn-text');
    const spinner = sendBtn.querySelector('.btn-spinner');
    sendBtn.disabled = true;
    btnText.textContent = '思考中';
    spinner.classList.remove('hidden');

    const typingEl = appendTyping();

    try {
        const r = await fetch('/api/ask', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                question,
                session_id: currentSessionId,
            }),
        });
        const data = await r.json();
        currentSessionId = data.session_id;
        document.getElementById('session-id').textContent = currentSessionId.substring(0, 8) + '...';

        typingEl.remove();
        appendMessage('assistant', data.answer, data.sources);
    } catch (e) {
        typingEl.remove();
        appendMessage('assistant', `请求失败: ${e.message}`);
    } finally {
        sendBtn.disabled = false;
        btnText.textContent = '发送';
        spinner.classList.add('hidden');
    }
}

function appendMessage(role, content, sources) {
    const container = document.getElementById('chat-messages');

    // Remove welcome message if present
    const welcome = container.querySelector('.welcome-msg');
    if (welcome) welcome.remove();

    const div = document.createElement('div');
    div.className = `message ${role}`;

    let html = `<div class="role">${role === 'user' ? '你' : 'AI'}</div>`;
    html += `<div class="content">${role === 'assistant' ? renderMarkdown(content) : escapeHtml(content)}</div>`;

    if (sources && sources.length > 0) {
        html += '<div class="sources">';
        html += '来源: ';
        sources.forEach(s => {
            html += `<span>${s.file} 第${s.page}页 (${s.rrf_score})</span>`;
        });
        html += '</div>';
    }

    div.innerHTML = html;
    container.appendChild(div);
    container.scrollTop = container.scrollHeight;
}

function appendTyping() {
    const container = document.getElementById('chat-messages');
    const div = document.createElement('div');
    div.className = 'message assistant';
    div.innerHTML = `
        <div class="role">AI</div>
        <div class="typing-indicator">
            <span></span><span></span><span></span>
        </div>
    `;
    container.appendChild(div);
    container.scrollTop = container.scrollHeight;
    return div;
}

// ---- Utils ----
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function formatSize(bytes) {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
    return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
}

function renderMarkdown(text) {
    if (typeof marked !== 'undefined') {
        return marked.parse(text);
    }
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}
