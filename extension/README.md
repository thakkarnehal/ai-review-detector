# AI Review Detector Chrome Extension

A Chrome extension that detects AI-generated restaurant reviews on Yelp and TripAdvisor.

## Features

- **Real-time detection**: Automatically scans reviews as you browse
- **Visual badges**: Green checkmark for authentic reviews, red flag for likely AI-generated
- **Floating widget**: Shows scan progress and adjusted rating (excluding flagged reviews)
- **Toggle on/off**: Enable or disable via popup
- **Configurable API**: Point to localhost for testing or your deployed API

## Installation

### Load as unpacked extension (development)

1. Open Chrome and go to `chrome://extensions/`
2. Enable "Developer mode" (toggle in top right)
3. Click "Load unpacked"
4. Select this `extension` folder
5. The extension icon should appear in your toolbar

### Configure API endpoint

1. Click the extension icon
2. Enter your API URL (default: `http://localhost:8080`)
3. Check the connection status indicator

## Usage

1. Navigate to a restaurant page on Yelp (`yelp.com/biz/*`) or TripAdvisor (`tripadvisor.com/Restaurant*`)
2. The extension will automatically scan visible reviews
3. Each review will display a badge:
   - ✓ **Green**: Likely authentic
   - 🚩 **Red**: Likely AI-generated
   - ⏳ **Gray**: Scanning...
4. The floating widget shows:
   - Total reviews scanned
   - Number flagged as AI
   - Original rating vs. adjusted rating

## Development

### API Requirements

The extension expects a backend API with:

```
POST /detect
Body: { "review_text": "..." }
Response: { "label": "Real|Fake", "confidence": 0.95, "flagged": true|false }

GET /health
Response: { "status": "ok", "model_loaded": true }
```

### Testing locally

1. Start the API server:
   ```bash
   docker run -p 8080:8080 review-detector
   ```

2. Load the extension in Chrome
3. Visit a Yelp or TripAdvisor restaurant page

## Files

- `manifest.json` - Extension configuration (Manifest V3)
- `content.js` - Content script that runs on Yelp/TripAdvisor pages
- `styles.css` - Badge and widget styles
- `popup.html` - Extension popup UI
- `popup.js` - Popup logic and settings
- `icons/` - Extension icons (16, 48, 128px)
