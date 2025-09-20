export async function apiGet(path: string) {
  const res = await fetch(path);
  if (!res.ok) throw new Error(`GET ${path} failed`);
  return res.json();
}

export async function apiPost(path: string, body: object) {
  const res = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(`POST ${path} failed`);
  return res.json();
}

export async function apiDelete(path: string) {
  const res = await fetch(path, { method: "DELETE" });
  if (!res.ok) throw new Error(`DELETE ${path} failed`);
  return res.json();
}

const api = {
  get: apiGet,
  post: apiPost,
  delete: apiDelete
};

export default api;
