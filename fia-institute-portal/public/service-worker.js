// VivaSense Service Worker
// Handles offline support, caching, and background sync

const CACHE_NAME = 'vivasense-v2';
const STATIC_ASSETS = [
  '/',
  '/index.html'
];

// Install event: cache static assets
self.addEventListener('install', (event) => {
  console.log('[ServiceWorker] Installing...');
  event.waitUntil(
    caches.open(CACHE_NAME).then((cache) => {
      console.log('[ServiceWorker] Caching static assets');
      return cache.addAll(STATIC_ASSETS).catch((err) => {
        console.warn('[ServiceWorker] Some assets failed to cache:', err);
        // Don't fail the install if some assets can't be cached
        return Promise.resolve();
      });
    })
  );
  self.skipWaiting();
});

// Activate event: clean up old caches
self.addEventListener('activate', (event) => {
  console.log('[ServiceWorker] Activating...');
  event.waitUntil(
    caches.keys().then((cacheNames) => {
      return Promise.all(
        cacheNames
          .filter((name) => name !== CACHE_NAME)
          .map((name) => {
            console.log('[ServiceWorker] Deleting old cache:', name);
            return caches.delete(name);
          })
      );
    })
  );
  self.clients.claim();
});

// Fetch event: network-first for API and JS/CSS assets, cache-first for HTML pages
self.addEventListener('fetch', (event) => {
  const { request } = event;
  const url = new URL(request.url);

  // Skip unsupported schemes (e.g., chrome-extension://)
  if (url.protocol !== 'http:' && url.protocol !== 'https:') {
    return;
  }

  // Skip non-GET requests
  if (request.method !== 'GET') {
    return;
  }

  // For API calls: network-first, no caching
  if (url.pathname.startsWith('/api/')) {
    event.respondWith(
      fetch(request).catch(() => caches.match(request))
    );
    return;
  }

  // For JS/CSS/font/image assets (hashed filenames): network-first to prevent stale cache
  if (url.pathname.startsWith('/assets/') || /\.(js|css|woff2?|ttf|eot|png|jpg|jpeg|svg|ico|webp)$/.test(url.pathname)) {
    event.respondWith(
      fetch(request)
        .then((response) => {
          if (response && response.status === 200) {
              const responseToCache = response.clone();
              caches.open(CACHE_NAME).then((c) => c.put(request, responseToCache));
          }
          return response;
        })
        .catch(() => caches.match(request))
    );
    return;
  }

  // For HTML navigation: network-first, fall back to cached index.html
  event.respondWith(
    fetch(request)
      .then((response) => {
        if (response && response.status === 200) {
            const responseToCache = response.clone();
            caches.open(CACHE_NAME).then((c) => c.put(request, responseToCache));
        }
        return response;
      })
      .catch(() => {
        return caches.match(request).then((cached) => cached || caches.match('/index.html'));
      })
  );
});

// Handle messages from clients
self.addEventListener('message', (event) => {
  if (event.data && event.data.type === 'SKIP_WAITING') {
    self.skipWaiting();
  }
});
