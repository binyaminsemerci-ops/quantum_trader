// Compatibility re-export: original project used TradeLogs (plural) in some pages.
// The canonical component is `TradeLog` (singular). Re-export default to preserve
// imports while we keep the single-file source.
export { default } from "./TradeLog";
