/**
 * API Service for Mango Leaf Disease Prediction Backend
 */

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

/**
 * Converts a data URL (e.g. SVG or base64) or remote URL into a File/Blob object
 */
async function urlToFileBlob(url, filename = 'specimen.jpg') {
  if (url.startsWith('data:image/svg+xml')) {
    // For SVG data URLs, create an SVG blob
    const svgContent = decodeURIComponent(url.split(',')[1]);
    return new Blob([svgContent], { type: 'image/svg+xml' });
  }
  
  const res = await fetch(url);
  const blob = await res.blob();
  return blob;
}

/**
 * Sends image to backend API for full diagnostic analysis
 * @param {Object} imageData - Object containing file or previewUrl
 * @returns {Promise<Object>} Diagnostic result payload
 */
export async function predictMangoLeaf(imageData) {
  const formData = new FormData();
  
  if (imageData.file) {
    formData.append('file', imageData.file, imageData.fileName || 'uploaded_leaf.jpg');
  } else if (imageData.previewUrl) {
    const blob = await urlToFileBlob(imageData.previewUrl, imageData.fileName || 'sample_leaf.jpg');
    formData.append('file', blob, imageData.fileName || 'sample_leaf.jpg');
  } else {
    throw new Error('No image file or preview URL provided for analysis.');
  }

  const response = await fetch(`${API_BASE_URL}/predict`, {
    method: 'POST',
    body: formData
  });

  if (!response.ok) {
    const errData = await response.json().catch(() => ({}));
    throw new Error(errData.error || `Server responded with status ${response.status}`);
  }

  const data = await response.json();
  return data;
}

/**
 * Checks backend health and loaded models
 */
export async function checkBackendHealth() {
  try {
    const response = await fetch(`${API_BASE_URL}/health`, { method: 'GET' });
    if (!response.ok) return { online: false };
    const data = await response.json();
    return { online: true, ...data };
  } catch (err) {
    return { online: false, error: err.message };
  }
}
