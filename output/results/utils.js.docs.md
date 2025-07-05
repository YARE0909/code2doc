# Documentation for `utils.js`

```javascript
function delay(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

function generateId() {
  return Math.random().toString(36).substr(2, 9);
}
```

## Generated Documentation

Delay a delay. @param {number} ms @return {Promise}
