# Documentation for `stringUtils.js`

```javascript
// stringUtils.js

function capitalize(str) {
  if (!str) return '';
  return str[0].toUpperCase() + str.slice(1);
}

function reverse(str) {
  return str.split('').reverse().join('');
}
```

## Generated Documentation

Turn a string into a camel case.

@param {string} str
@returns {string}
