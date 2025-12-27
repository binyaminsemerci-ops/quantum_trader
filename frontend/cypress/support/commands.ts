/// <reference types="cypress" />

// Custom commands for Cypress
// Example: Cypress.Commands.add('login', (email, password) => { ... })

declare global {
  namespace Cypress {
    interface Chainable {
      // Add custom command type definitions here
      // login(email: string, password: string): Chainable<void>
    }
  }
}

export {};
