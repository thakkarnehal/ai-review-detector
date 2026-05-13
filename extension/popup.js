// AI Review Detector - Popup Script

document.addEventListener('DOMContentLoaded', init);

async function init() {
  const enableToggle = document.getElementById('enableToggle');
  const apiUrlInput = document.getElementById('apiUrl');
  const statusDot = document.getElementById('statusDot');
  const statusText = document.getElementById('statusText');

  // Load saved settings
  const settings = await chrome.storage.sync.get(['enabled', 'apiUrl']);
  enableToggle.checked = settings.enabled !== false;
  apiUrlInput.value = settings.apiUrl || 'http://54.146.198.130:8080';

  // Check API status
  checkApiStatus(apiUrlInput.value);

  // Enable/disable toggle
  enableToggle.addEventListener('change', async () => {
    await chrome.storage.sync.set({ enabled: enableToggle.checked });
  });

  // API URL change
  let debounceTimer;
  apiUrlInput.addEventListener('input', () => {
    clearTimeout(debounceTimer);
    debounceTimer = setTimeout(async () => {
      const url = apiUrlInput.value.trim();
      await chrome.storage.sync.set({ apiUrl: url });
      checkApiStatus(url);
    }, 500);
  });

  // Get stats from current tab
  loadPageStats();
}

async function checkApiStatus(url) {
  const statusDot = document.getElementById('statusDot');
  const statusText = document.getElementById('statusText');

  statusDot.className = 'status-dot';
  statusText.textContent = 'Checking...';

  try {
    const response = await fetch(`${url}/health`, {
      method: 'GET',
      signal: AbortSignal.timeout(5000)
    });

    if (response.ok) {
      const data = await response.json();
      statusDot.classList.add('connected');
      statusText.textContent = `Connected (${data.model || 'ready'})`;
    } else {
      throw new Error('Not OK');
    }
  } catch (error) {
    statusDot.classList.add('error');
    statusText.textContent = 'Not connected';
  }
}

async function loadPageStats() {
  const totalScanned = document.getElementById('totalScanned');
  const totalFlagged = document.getElementById('totalFlagged');
  const totalAuthentic = document.getElementById('totalAuthentic');

  try {
    // Get current tab
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
    
    if (!tab?.id) return;

    // Send message to content script
    chrome.tabs.sendMessage(tab.id, { action: 'getStats' }, (response) => {
      if (chrome.runtime.lastError || !response) {
        // Content script not loaded on this page
        totalScanned.textContent = '-';
        totalFlagged.textContent = '-';
        totalAuthentic.textContent = '-';
        return;
      }

      totalScanned.textContent = response.total || 0;
      totalFlagged.textContent = response.flagged || 0;
      totalAuthentic.textContent = (response.total - response.flagged) || 0;
    });
  } catch (error) {
    console.error('Error loading stats:', error);
  }
}
