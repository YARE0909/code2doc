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

module.exports = { capitalize, reverse };

```

## Generated Documentation

/*  @param String str @return String @return String
