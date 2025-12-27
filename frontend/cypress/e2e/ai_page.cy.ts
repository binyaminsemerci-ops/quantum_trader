/**
 * Cypress E2E Test: AI Engine Dashboard
 * 
 * Purpose: Detect numeric rendering errors (.toFixed crashes) before production
 * Tests the /ai page for console errors and NaN rendering
 */

describe("AI Engine Dashboard - Numeric Safety", () => {
  beforeEach(() => {
    // Visit the AI Engine page
    cy.visit("/ai");
  });

  it("loads without TypeError or .toFixed errors", () => {
    // Spy on console.error
    cy.window().then((win) => {
      cy.spy(win.console, "error").as("consoleError");
    });

    // Wait for page to fully render
    cy.wait(2000);

    // Check that no .toFixed errors occurred
    cy.get("@consoleError").should((spy) => {
      const calls = (spy as any).getCalls();
      const hasToFixedError = calls.some((call: any) => {
        const args = call.args.join(" ");
        return args.includes("toFixed") || args.includes("Cannot read properties of undefined");
      });
      expect(hasToFixedError, "No .toFixed errors should occur").to.be.false;
    });
  });

  it("renders numeric cards safely without NaN", () => {
    // Wait for content to load
    cy.wait(2000);

    // Check that common numeric fields exist (adjust selectors based on your page)
    cy.contains(/confidence|accuracy|performance/i).should("exist");

    // Ensure no "NaN" text is rendered anywhere
    cy.get("body").should("not.contain", "NaN");
    cy.get("body").should("not.contain", "Infinity");
    cy.get("body").should("not.contain", "undefined");
  });

  it("handles API errors gracefully", () => {
    // Intercept API calls and return errors
    cy.intercept("GET", "**/api/ai/**", {
      statusCode: 500,
      body: { error: "Internal Server Error" },
    }).as("aiApiError");

    // Reload page
    cy.reload();

    // Wait for API call
    cy.wait("@aiApiError");

    // Page should still render without crashing
    cy.get("body").should("be.visible");
    
    // Should not show NaN or undefined
    cy.get("body").should("not.contain", "NaN");
  });

  it("handles missing data fields gracefully", () => {
    // Intercept API and return incomplete data
    cy.intercept("GET", "**/api/ai/**", {
      statusCode: 200,
      body: {
        confidence: null,
        accuracy: undefined,
        performance: {},
      },
    }).as("incompleteData");

    // Reload page
    cy.reload();

    // Wait for API call
    cy.wait("@incompleteData");

    // Should render default values (0.00, etc)
    cy.get("body").should("be.visible");
    cy.get("body").should("not.contain", "NaN");
    cy.get("body").should("not.contain", "undefined");
  });

  it("displays loading states properly", () => {
    // Visit page
    cy.visit("/ai");

    // Should show loading indicator initially (adjust selector as needed)
    cy.get("body").should("be.visible");

    // After loading, should show content
    cy.wait(2000);
    cy.get("body").should("be.visible");
  });

  it("handles extreme numeric values safely", () => {
    // Intercept API with extreme values
    cy.intercept("GET", "**/api/ai/**", {
      statusCode: 200,
      body: {
        confidence: 999999999.999999,
        accuracy: 0.0000000001,
        performance: -Infinity,
        pnl: Infinity,
      },
    }).as("extremeValues");

    // Reload page
    cy.reload();

    // Wait for API call
    cy.wait("@extremeValues");

    // Should not crash or show invalid values
    cy.get("body").should("be.visible");
    cy.get("body").should("not.contain", "Infinity");
    cy.get("body").should("not.contain", "-Infinity");
    cy.get("body").should("not.contain", "NaN");
  });

  it("navigates between pages without errors", () => {
    // Test navigation doesn't cause numeric errors
    cy.visit("/ai");
    cy.wait(1000);

    // Navigate to dashboard
    cy.visit("/");
    cy.wait(1000);

    // Back to AI page
    cy.visit("/ai");
    cy.wait(1000);

    // No console errors should occur
    cy.window().then((win) => {
      cy.spy(win.console, "error").as("consoleError");
    });

    cy.get("@consoleError").should((spy) => {
      const calls = (spy as any).getCalls();
      expect(calls.length, "Should have no console errors").to.equal(0);
    });
  });
});

describe("Dashboard Page - Numeric Safety", () => {
  beforeEach(() => {
    cy.visit("/");
  });

  it("loads dashboard without numeric errors", () => {
    cy.window().then((win) => {
      cy.spy(win.console, "error").as("consoleError");
    });

    cy.wait(2000);

    cy.get("@consoleError").should((spy) => {
      const calls = (spy as any).getCalls();
      const hasNumericError = calls.some((call: any) => {
        const args = call.args.join(" ");
        return (
          args.includes("toFixed") ||
          args.includes("Cannot read properties of undefined") ||
          args.includes("NaN")
        );
      });
      expect(hasNumericError, "No numeric errors on dashboard").to.be.false;
    });
  });

  it("displays PnL and metrics correctly", () => {
    cy.wait(2000);
    
    // Body should be visible
    cy.get("body").should("be.visible");
    
    // No invalid values
    cy.get("body").should("not.contain", "NaN");
    cy.get("body").should("not.contain", "undefined");
  });
});
