// AI Review Detector - Content Script

(function() {
  'use strict';

  // Configuration
  const CONFIG = {
    API_URL: 'http://54.146.198.130:8080',
    BATCH_SIZE: 10,
    SCAN_DELAY: 1000
  };

  // State
  let isEnabled = true;
  let scannedReviews = new Map();
  let stats = { total: 0, flagged: 0, originalRating: null };

  // Load settings
  chrome.storage.sync.get(['enabled', 'apiUrl'], (result) => {
    isEnabled = result.enabled !== false;
    if (result.apiUrl) CONFIG.API_URL = result.apiUrl;
    if (isEnabled) init();
  });

  // Listen for toggle from popup
  chrome.storage.onChanged.addListener((changes) => {
    if (changes.enabled) {
      isEnabled = changes.enabled.newValue;
      if (isEnabled) {
        init();
      } else {
        cleanup();
      }
    }
    if (changes.apiUrl) {
      CONFIG.API_URL = changes.apiUrl.newValue;
    }
  });

  // Listen for messages from popup
  chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    if (request.action === 'getStats') {
      sendResponse(stats);
    }
    return true;
  });

  function init() {
    if (window.location.hostname.includes('yelp.com')) {
      scanYelpPage();
    } else if (window.location.hostname.includes('tripadvisor.com')) {
      scanTripAdvisorPage();
    }
    createWidget();
  }

  function cleanup() {
    // Remove all badges
    document.querySelectorAll('.ai-review-badge').forEach(el => el.remove());
    document.querySelectorAll('.ai-review-widget').forEach(el => el.remove());
    scannedReviews.clear();
    stats = { total: 0, flagged: 0, originalRating: null };
  }

  // ========== YELP ==========
  function scanYelpPage() {
    // Get business rating
    const ratingEl = document.querySelector('[data-testid="rating-summary"] span');
    if (ratingEl) {
      stats.originalRating = parseFloat(ratingEl.textContent);
    }

    // Find review text elements — Yelp uses lang="en" on review text spans
    const textEls = document.querySelectorAll('[lang="en"]');
    if (textEls.length === 0) {
      setTimeout(scanYelpPage, CONFIG.SCAN_DELAY);
      return;
    }

    const reviews = [];
    textEls.forEach((textEl, idx) => {
      // Skip cookie/non-review elements
      const raw = textEl.textContent.trim();
      if (raw.length < 20 || raw.includes('Cookie preferences')) return;

      // Use text fingerprint as stable ID
      const reviewId = 'yelp-' + raw.slice(0, 40).replace(/\s+/g, '-');
      if (scannedReviews.has(reviewId)) return;

      // Clone the comment container and strip injected badges + Yelp UI elements
      // before extracting text — avoids badge text polluting the classification
      const commentContainer = textEl.closest('[class*="comment"]');
      let text = raw;
      if (commentContainer) {
        const clone = commentContainer.cloneNode(true);
        clone.querySelectorAll('.ai-review-badge, button, [aria-hidden="true"]').forEach(el => el.remove());
        text = clone.textContent.replace(/\s+/g, ' ').trim();
      }

      // Skip if not enough text to classify reliably (truncated with no hidden copy)
      if (text.length < 80) return;

      // Walk up to find the review card container
      const container = textEl.closest('[class*="arrange__"], [class*="review"]')
                     || textEl.parentElement?.parentElement?.parentElement;
      if (!container) return;

      // Find star rating in the container
      const ratingEl = container.querySelector('[aria-label*="star rating"], [class*="stars--"]');
      const ratingAttr = ratingEl?.getAttribute('aria-label') || '';
      const ratingMatch = ratingAttr.match(/(\d+(\.\d+)?)\s*star/i);
      const rating = ratingMatch ? parseFloat(ratingMatch[1]) : null;

      reviews.push({ id: reviewId, text, rating, element: container, textEl });
    });

    if (reviews.length > 0) {
      analyzeReviews(reviews);
    }

    // Watch for new reviews (infinite scroll)
    observeNewReviews(scanYelpPage);
  }

  // ========== TRIPADVISOR ==========
  function scanTripAdvisorPage() {
    // Get business rating
    const ratingEl = document.querySelector('[data-test-target="review-rating"], .ZDEqb');
    if (ratingEl) {
      const ratingText = ratingEl.getAttribute('aria-label') || ratingEl.textContent;
      const match = ratingText.match(/(\d+\.?\d*)/);
      if (match) stats.originalRating = parseFloat(match[1]);
    }

    // Find review containers
    const reviewContainers = document.querySelectorAll('[data-reviewid], .review-container');
    if (reviewContainers.length === 0) {
      setTimeout(scanTripAdvisorPage, CONFIG.SCAN_DELAY);
      return;
    }

    const reviews = [];
    reviewContainers.forEach(container => {
      const reviewId = container.getAttribute('data-reviewid') || 
                       container.id || 
                       `ta-${reviews.length}`;
      
      if (scannedReviews.has(reviewId)) return;

      const textEl = container.querySelector(
        '.partial_entry, ' +
        '[data-test-target="review-body"], ' +
        '.reviewText, ' +
        '[class*="reviewText"], ' +
        '[class*="review-body"]'
      );
      const ratingEl = container.querySelector('.ui_bubble_rating, [data-test-target="review-rating"]');
      
      if (textEl) {
        const text = textEl.textContent.trim();
        let rating = null;
        
        if (ratingEl) {
          const ratingClass = ratingEl.className;
          const match = ratingClass.match(/bubble_(\d+)/);
          if (match) rating = parseInt(match[1]) / 10;
        }
        
        console.debug('[AI-Detector] Found TA review (%d chars): %s…', text.length, text.slice(0, 80));
        
        if (text.length > 20) {
          reviews.push({
            id: reviewId,
            text: text,
            rating: rating,
            element: container
          });
        }
      } else {
        console.debug('[AI-Detector] No text found in TA container', reviewId);
      }
    });

    if (reviews.length > 0) {
      analyzeReviews(reviews);
    }

    observeNewReviews(scanTripAdvisorPage);
  }

  // ========== ANALYSIS ==========
  async function analyzeReviews(reviews) {
    // Mark as processing
    reviews.forEach(r => {
      scannedReviews.set(r.id, { status: 'pending' });
      addBadge(r.element, 'pending');
    });

    // Batch API calls
    for (let i = 0; i < reviews.length; i += CONFIG.BATCH_SIZE) {
      const batch = reviews.slice(i, i + CONFIG.BATCH_SIZE);
      
      try {
        const results = await Promise.all(
          batch.map(review => analyzeReview(review.text))
        );

        results.forEach((result, idx) => {
          const review = batch[idx];
          const isFlagged = result.flagged;
          
          scannedReviews.set(review.id, {
            status: 'done',
            flagged: isFlagged,
            confidence: result.confidence,
            rating: review.rating
          });

          stats.total++;
          if (isFlagged) stats.flagged++;

          updateBadge(review.element, isFlagged, result.confidence);
        });

        updateWidget();
      } catch (error) {
        console.error('AI Review Detector: API error', error);
        batch.forEach(review => {
          updateBadge(review.element, 'error');
        });
      }
    }
  }

  function analyzeReview(text) {
    return new Promise((resolve, reject) => {
      chrome.runtime.sendMessage(
        { action: 'detectReview', apiUrl: CONFIG.API_URL, text },
        (response) => {
          if (chrome.runtime.lastError) {
            reject(new Error(chrome.runtime.lastError.message));
          } else if (!response.success) {
            reject(new Error(response.error));
          } else {
            resolve(response.data);
          }
        }
      );
    });
  }

  // ========== UI ==========
  function addBadge(element, status) {
    // Remove existing badge
    const existing = element.querySelector('.ai-review-badge');
    if (existing) existing.remove();

    const badge = document.createElement('div');
    badge.className = 'ai-review-badge ai-review-badge--pending';
    badge.innerHTML = `
      <span class="ai-badge-icon">⏳</span>
      <span class="ai-badge-text">Scanning...</span>
    `;
    
    // Insert badge directly before the text element if available, else top of container
    const textEl = element.querySelector('[lang="en"]') || element.firstChild;
    if (textEl && textEl.parentNode) {
      textEl.parentNode.insertBefore(badge, textEl);
    } else {
      element.insertBefore(badge, element.firstChild);
    }
  }

  function updateBadge(element, flagged, confidence) {
    const badge = element.querySelector('.ai-review-badge');
    if (!badge) return;

    badge.className = 'ai-review-badge';
    
    if (flagged === 'error') {
      badge.classList.add('ai-review-badge--error');
      badge.innerHTML = `
        <span class="ai-badge-icon">⚠️</span>
        <span class="ai-badge-text">Error</span>
      `;
    } else if (flagged) {
      badge.classList.add('ai-review-badge--flagged');
      const pct = (confidence * 100).toFixed(0);
      badge.innerHTML = `
        <span class="ai-badge-icon">🚩</span>
        <span class="ai-badge-text">Likely AI-generated (${pct}%)</span>
      `;
    } else {
      badge.classList.add('ai-review-badge--real');
      badge.innerHTML = `
        <span class="ai-badge-icon">✓</span>
        <span class="ai-badge-text">Likely authentic</span>
      `;
    }
  }

  function createWidget() {
    // Remove existing widget
    const existing = document.querySelector('.ai-review-widget');
    if (existing) existing.remove();

    const widget = document.createElement('div');
    widget.className = 'ai-review-widget';
    widget.innerHTML = `
      <div class="ai-widget-header">
        <span class="ai-widget-icon">🔍</span>
        <span class="ai-widget-title">AI Review Detector</span>
        <button class="ai-widget-minimize">−</button>
      </div>
      <div class="ai-widget-body">
        <div class="ai-widget-stat">
          <span class="ai-widget-label">Reviews scanned:</span>
          <span class="ai-widget-value" id="ai-total">0</span>
        </div>
        <div class="ai-widget-stat">
          <span class="ai-widget-label">Flagged as AI:</span>
          <span class="ai-widget-value ai-widget-flagged" id="ai-flagged">0</span>
        </div>
        <div class="ai-widget-divider"></div>
        <div class="ai-widget-stat">
          <span class="ai-widget-label">Original rating:</span>
          <span class="ai-widget-value" id="ai-original">--</span>
        </div>
        <div class="ai-widget-stat">
          <span class="ai-widget-label">Adjusted rating:</span>
          <span class="ai-widget-value ai-widget-adjusted" id="ai-adjusted">--</span>
        </div>
      </div>
    `;

    document.body.appendChild(widget);

    // Minimize functionality
    const minimizeBtn = widget.querySelector('.ai-widget-minimize');
    const body = widget.querySelector('.ai-widget-body');
    minimizeBtn.addEventListener('click', () => {
      body.classList.toggle('ai-widget-hidden');
      minimizeBtn.textContent = body.classList.contains('ai-widget-hidden') ? '+' : '−';
    });

    updateWidget();
  }

  function updateWidget() {
    const totalEl = document.getElementById('ai-total');
    const flaggedEl = document.getElementById('ai-flagged');
    const originalEl = document.getElementById('ai-original');
    const adjustedEl = document.getElementById('ai-adjusted');

    if (totalEl) totalEl.textContent = stats.total;
    if (flaggedEl) flaggedEl.textContent = stats.flagged;
    
    if (stats.originalRating && originalEl) {
      originalEl.textContent = stats.originalRating.toFixed(1) + ' ★';
    }

    // Calculate adjusted rating
    if (stats.total > 0 && adjustedEl) {
      const realReviews = [];
      scannedReviews.forEach(data => {
        if (data.status === 'done' && !data.flagged && data.rating) {
          realReviews.push(data.rating);
        }
      });

      if (realReviews.length > 0) {
        const avg = realReviews.reduce((a, b) => a + b, 0) / realReviews.length;
        adjustedEl.textContent = avg.toFixed(2) + ' ★';
        
        // Color based on difference
        if (stats.originalRating) {
          const diff = stats.originalRating - avg;
          if (diff > 0.5) {
            adjustedEl.style.color = '#e53935';
          } else if (diff > 0.2) {
            adjustedEl.style.color = '#fb8c00';
          } else {
            adjustedEl.style.color = '#43a047';
          }
        }
      }
    }
  }

  // ========== OBSERVERS ==========
  let observerDebounce;
  function observeNewReviews(scanFunction) {
    const observer = new MutationObserver(() => {
      clearTimeout(observerDebounce);
      observerDebounce = setTimeout(scanFunction, 500);
    });

    observer.observe(document.body, {
      childList: true,
      subtree: true
    });
  }

})();
