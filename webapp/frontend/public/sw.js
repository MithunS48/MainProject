/**
 * AquaScan Service Worker
 * ────────────────────────
 * Lightweight cache-first strategy for static assets.
 * Network-first for API calls.
 */

const CACHE_NAME = "aquascan-v1";
const STATIC_ASSETS = ["/", "/index.html"];

// Install — cache static shell
self.addEventListener("install", (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME).then((cache) => cache.addAll(STATIC_ASSETS))
  );
  self.skipWaiting();
});

// Activate — clean old caches
self.addEventListener("activate", (event) => {
  event.waitUntil(
    caches.keys().then((keys) =>
      Promise.all(keys.filter((k) => k !== CACHE_NAME).map((k) => caches.delete(k)))
    )
  );
  self.clients.claim();
});

// Fetch
self.addEventListener("fetch", (event) => {
  const { request } = event;
  const url = new URL(request.url);

  // Skip non-GET and chrome-extension requests
  if (request.method !== "GET" || url.protocol === "chrome-extension:") return;

  // API — network first, fallback to cache
  if (url.pathname.startsWith("/api/") || url.pathname.startsWith("/static/")) {
    event.respondWith(
      fetch(request)
        .then((res) => {
          const clone = res.clone();
          caches.open(CACHE_NAME).then((c) => c.put(request, clone));
          return res;
        })
        .catch(() => caches.match(request))
    );
    return;
  }

  // Static assets — cache first
  event.respondWith(
    caches.match(request).then(
      (cached) => cached || fetch(request).then((res) => {
        const clone = res.clone();
        caches.open(CACHE_NAME).then((c) => c.put(request, clone));
        return res;
      })
    )
  );
});
