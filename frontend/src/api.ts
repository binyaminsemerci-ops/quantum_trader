import axios from "axios";

// Export a minimal, migration-friendly axios instance typed as `any` so
// legacy callsites like `api.get(...).then(res => res.data)` keep working
// during the gradual TypeScript conversion.
const api = axios.create({
  baseURL: "/api",
});

export default (api as any);
