// utils.js

/**
 * Compute the greatest common divisor (GCD) of two non-negative integers
 * using the Euclidean algorithm.
 *
 * @param {number} a  – first integer
 * @param {number} b  – second integer
 * @returns {number}  – the GCD of a and b
 */
function gcd(a, b) {
  if (b === 0) {
    return a;
  }
  return gcd(b, a % b);
}

/**
 * Check whether a given string is a palindrome.
 *
 * @param {string} str – the string to test
 * @returns {boolean}  – true if str reads the same forwards and backwards
 */
function isPalindrome(str) {
  const cleaned = str.replace(/[\W_]/g, '').toLowerCase();
  return cleaned === cleaned.split('').reverse().join('');
}

module.exports = { gcd, isPalindrome };
