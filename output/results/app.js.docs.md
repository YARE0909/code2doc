# Documentation for `app.js`

```javascript
function gcd(a, b) {
  if (b === 0) {
    return a;
  }
  return gcd(b, a % b);
}

function isPalindrome(str) {
  const cleaned = str.replace(/[\W_]/g, '').toLowerCase();
  return cleaned === cleaned.split('').reverse().join('');
}


```

## Generated Documentation

Calculates the gcd between two numbers. @param {number} a @param {number} b @return {number}
