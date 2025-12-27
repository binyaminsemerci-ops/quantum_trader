// ***********************************************************
// This support file is processed before test files
// ***********************************************************

// Import commands.js using ES2015 syntax:
import "./commands";

// Prevent Cypress from failing on uncaught exceptions
// We want to explicitly test for console errors instead
Cypress.on("uncaught:exception", (err, runnable) => {
  // Return false to prevent the error from failing the test
  return false;
});
