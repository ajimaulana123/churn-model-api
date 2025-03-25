import http from 'k6/http';
import { check, sleep } from 'k6';

// Konfigurasi test
export const options = {
  stages: [
    { duration: '30s', target: 20 },   // Ramp-up perlahan
    { duration: '1m', target: 100 },   // Pertahankan traffic tinggi
    { duration: '20s', target: 0 },    // Ramp-down
  ],
  thresholds: {
    http_req_duration: ['p(95)<500'],  // 95% request harus <500ms
    http_req_failed: ['rate<0.01'],    // Error rate <1%
  },
};

// Payload contoh dari API Anda
const payload = JSON.stringify({
  features: [0, 0, 1, 0, 45, 1, 0, 2, 0, 0, 0, 0, 0, 0, 1, 1, 2, 89.85, 4034.45]
});

const headers = { 'Content-Type': 'application/json' };

export default function () {
  const res = http.post(
    'http://localhost:8000/predict', // Ganti dengan URL API Anda
    payload,
    { headers }
  );

  // Verifikasi response
  check(res, {
    'status 200': (r) => r.status === 200,
    'response valid': (r) => JSON.parse(r.body).hasOwnProperty('prediction'),
  });

  sleep(1); // Jeda 1 detik antara request
}