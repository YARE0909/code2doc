# Documentation for `stringUtils.js`

```javascript
function capitalize(str) {
    if (!str) return '';
    return str.charAt(0).toUpperCase() + str.slice(1).toLowerCase();
}

function reverse(str) {
    return str.split('').reverse().join('');
}
```

## Generated Documentation

Convert a string to lower case @param {String} str @return {String}
