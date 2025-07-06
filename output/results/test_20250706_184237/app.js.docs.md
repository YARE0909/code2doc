# Documentation for `app.js`

```javascript
function gcd(a, b) {
  if (b === 0) {
    return a;
  }
  return gcd(b, a % b);
}
```

## Generated Documentation

Returns the distance between two numbers.
