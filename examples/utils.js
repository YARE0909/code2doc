function delay(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

function generateId() {
  return Math.random().toString(36).substr(2, 9);
}

module.exports = { delay, generateId };
