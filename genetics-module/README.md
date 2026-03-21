# Claude API Budget Tracker

A complete, production-ready solution for using Claude API with built-in cost tracking, budget limits, and real-time monitoring.

## ✨ Features

- **💰 Budget Protection** - Set daily and monthly spending limits
- **📊 Real-time Tracking** - Monitor costs as you go
- **🚨 Automatic Alerts** - Warnings at 80% budget usage
- **📈 Usage Reports** - Detailed breakdowns by model and time period
- **🔒 Rate Limiting** - Prevent abuse and unexpected costs
- **🎯 Smart Model Selection** - Automatically use cheaper models for simple tasks
- **💾 Persistent Storage** - Track usage across restarts
- **🌐 Ready-to-use API** - Drop-in Express.js backend
- **🎨 Demo Frontend** - Beautiful UI to test and monitor

## 📦 What's Included

```
.
├── claude-budget-tracker.js   # Core budget tracking class
├── server.js                  # Express.js API server
├── index.html                 # Demo frontend
├── package.json               # Dependencies
├── .env.example              # Environment variables template
└── claude_api_cost_guide.md  # Comprehensive cost guide
```

## 🚀 Quick Start

### 1. Get Your API Key

1. Go to https://console.anthropic.com
2. Sign up and verify your email
3. Navigate to **Settings → API Keys**
4. Click **Create Key** and copy it

### 2. Install Dependencies

```bash
# Install Node.js (if not already installed)
# Download from: https://nodejs.org (v18 or higher)

# Install project dependencies
npm install
```

### 3. Configure Environment

```bash
# Copy the example env file
cp .env.example .env

# Edit .env and add your API key
# ANTHROPIC_API_KEY=sk-ant-your-key-here
```

### 4. Start the Server

```bash
# Development mode (auto-restart on changes)
npm run dev

# Production mode
npm start
```

Server will start on http://localhost:3000

### 5. Open the Demo

Open `index.html` in your browser or visit:
```
http://localhost:3000
```

(You'll need to serve the HTML file - just double-click it or use a simple HTTP server)

## 📚 Usage Examples

### Basic Usage (Standalone)

```javascript
const ClaudeBudgetTracker = require('./claude-budget-tracker');

const claude = new ClaudeBudgetTracker({
  apiKey: 'your-api-key',
  dailyLimit: 2.00,
  monthlyLimit: 40.00,
});

// Simple chat
const response = await claude.chat('What is 2 + 2?');
console.log(response.text);  // "4"
console.log(response.usage); // { inputTokens: 15, outputTokens: 8, cost: 0.0002 }

// Get current budget status
const status = claude.getStatus();
console.log(status);
// {
//   current: { dailySpent: 0.0002, monthlySpent: 0.0002 },
//   limits: { daily: 2.00, monthly: 40.00 },
//   remaining: { daily: 1.9998, monthly: 39.9998 },
//   percentage: { daily: '0.0', monthly: '0.0' }
// }

// Generate usage report
claude.printReport(7); // Last 7 days
```

### API Endpoints

#### POST /api/chat
Send a message to Claude

**Request:**
```json
{
  "message": "Explain quantum computing",
  "model": "claude-sonnet-4-5-20250929",
  "maxTokens": 1024
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "reply": "Quantum computing is...",
    "usage": {
      "inputTokens": 42,
      "outputTokens": 150,
      "cost": 0.0024
    },
    "spending": {
      "daily": 0.0024,
      "monthly": 0.0024
    }
  }
}
```

#### POST /api/chat/smart
Smart model selection (automatically chooses cheapest model)

**Request:**
```json
{
  "message": "What is 2 + 2?"
}
```

**Response:**
Uses Haiku for simple queries, Sonnet for complex ones.

#### GET /api/budget/status
Get current budget status

**Response:**
```json
{
  "success": true,
  "data": {
    "current": { "dailySpent": 1.23, "monthlySpent": 15.67 },
    "limits": { "daily": 2.00, "monthly": 40.00 },
    "remaining": { "daily": 0.77, "monthly": 24.33 },
    "percentage": { "daily": "61.5", "monthly": "39.2" }
  }
}
```

#### GET /api/budget/report?days=7
Get usage report for last N days

**Response:**
```json
{
  "success": true,
  "data": {
    "period": "Last 7 days",
    "totalCost": 12.45,
    "totalCalls": 327,
    "avgCostPerCall": 0.038,
    "modelBreakdown": {
      "claude-sonnet-4-5-20250929": { "calls": 300, "cost": 11.40 },
      "claude-haiku-4-5-20251001": { "calls": 27, "cost": 1.05 }
    }
  }
}
```

## ⚙️ Configuration

### Budget Limits

Set in `claude-budget-tracker.js` constructor:

```javascript
const claude = new ClaudeBudgetTracker({
  dailyLimit: 2.00,    // $2 per day
  monthlyLimit: 40.00,  // $40 per month
  logFile: './claude_usage.json', // Where to store usage data
});
```

### Model Pricing

Current pricing (per million tokens):

| Model | Input | Output | Best For |
|-------|-------|--------|----------|
| Haiku 4.5 | $1 | $5 | Simple, fast tasks |
| Sonnet 4.5 | $3 | $15 | Balanced - most use cases |
| Opus 4.5 | $5 | $25 | Complex reasoning |

### Rate Limiting

Modify in `server.js`:

```javascript
const limiter = rateLimit({
  windowMs: 15 * 60 * 1000, // 15 minutes
  max: 20, // 20 requests per window
});
```

## 🛡️ Security Best Practices

### 1. Protect Your API Key

```bash
# ✅ Good - environment variable
ANTHROPIC_API_KEY=sk-ant-...

# ❌ Bad - hardcoded in code
const apiKey = "sk-ant-...";

# ❌ Bad - committed to Git
git add .env
```

### 2. Never Expose API Key to Frontend

```javascript
// ✅ Good - server-side only
// server.js handles API calls

// ❌ Bad - frontend JavaScript
fetch('https://api.anthropic.com/...', {
  headers: { 'x-api-key': 'sk-ant-...' }
});
```

### 3. Add Authentication

```javascript
// Protect admin endpoints
app.post('/api/admin/reset-budget', authenticateAdmin, (req, res) => {
  // Only allow authenticated admins
});
```

### 4. Set Hard Limits

```javascript
// Prevent budget overruns
const HARD_DAILY_CAP = 5.00;
const HARD_MONTHLY_CAP = 100.00;

if (spent >= HARD_DAILY_CAP) {
  throw new Error('Hard daily cap reached');
}
```

## 📊 Monitoring & Optimization

### Track Token Efficiency

```javascript
// Monitor average tokens per call
const report = claude.getReport(30);
console.log(`Avg tokens/call: ${report.avgTokensPerCall}`);

// If too high, optimize prompts:
// - Be more concise
// - Use shorter system prompts
// - Request briefer responses
```

### Use Prompt Caching

For repeated context (90% savings):

```javascript
// First call: Full price
await claude.chat(longContext + question);

// Subsequent calls: 90% off on cached context
await claude.chat(longContext + anotherQuestion);
```

### Batch Processing

For non-urgent tasks (50% discount):

```javascript
// Use Anthropic's Batch API
// See: https://docs.anthropic.com/en/api/batch-api
```

## 🎯 Cost Optimization Tips

1. **Use Haiku for Simple Tasks**
   - Quick questions
   - Simple classifications
   - Basic summaries
   - 60% cheaper than Sonnet!

2. **Set Appropriate max_tokens**
   ```javascript
   // Don't let responses run wild
   maxTokens: 500 // vs 4096 default
   ```

3. **Cache Responses**
   ```javascript
   // Store common questions
   const cache = new Map();
   if (cache.has(question)) return cache.get(question);
   ```

4. **Smart Model Selection**
   ```javascript
   // Already built into /api/chat/smart endpoint
   // Uses word count and complexity to choose model
   ```

5. **Monitor and Adjust**
   ```bash
   # Check weekly
   npm run report
   
   # Adjust limits as needed
   ```

## 🚨 Common Issues

### "Budget exceeded" Error

**Cause:** Daily or monthly limit reached

**Solution:**
```javascript
// Option 1: Wait for reset (5 hours for daily)
// Option 2: Increase limits
claude.dailyLimit = 5.00;

// Option 3: Reset manually (admin only)
claude.resetBudget();
```

### "API key invalid" Error

**Cause:** Wrong or expired API key

**Solution:**
```bash
# Get new key from console.anthropic.com
# Update .env file
ANTHROPIC_API_KEY=sk-ant-new-key-here

# Restart server
npm start
```

### High Token Usage

**Cause:** Verbose prompts or responses

**Solution:**
```javascript
// Be concise in prompts
const response = await claude.chat(
  "Summarize in 2 sentences: " + longText
);

// Set lower max_tokens
{ maxTokens: 200 }
```

## 📈 Scaling to Production

### 1. Database Storage

Replace file-based logging with a database:

```javascript
// Instead of JSON file
async function logUsage(data) {
  await db.query('INSERT INTO usage ...', data);
}
```

### 2. Multi-User Support

Track budget per user:

```javascript
const userBudgets = new Map();

app.post('/api/chat', async (req, res) => {
  const userId = req.user.id;
  const userClaude = getUserBudgetTracker(userId);
  // ...
});
```

### 3. Caching Layer

Use Redis for response caching:

```javascript
const redis = require('redis');
const cache = redis.createClient();

// Check cache before API call
const cached = await cache.get(messageHash);
if (cached) return cached;
```

### 4. Load Balancing

Multiple API keys for higher throughput:

```javascript
const keys = [key1, key2, key3];
const claude = new ClaudeBudgetTracker({
  apiKey: keys[Math.floor(Math.random() * keys.length)]
});
```

## 📖 Additional Resources

- [Anthropic API Docs](https://docs.anthropic.com)
- [Claude Pricing](https://anthropic.com/pricing)
- [API Console](https://console.anthropic.com)
- [Community Discord](https://anthropic.com/discord)

## 🤝 Support

Issues? Questions?
1. Check `claude_api_cost_guide.md` for detailed explanations
2. Review the code comments
3. Test with free credits first
4. Monitor usage closely for first week

## 📝 License

MIT - Use freely in your projects!

## 🎉 You're Ready!

Start with free credits, monitor closely, and scale as needed. The budget tracker will keep you safe while you build amazing AI-powered applications!

```bash
# Start building!
npm start
```

Good luck! 🚀
