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

Delay a delay in ms.

@public
@param {number} ms The number of milliseconds to delay.
@return {Promise} A promise that will be resolved after the delay.
