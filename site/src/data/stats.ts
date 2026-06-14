// Build-time fetch of live-ish stats with safe fallbacks. Runs once during
// `astro build`. If an API is flaky, the fallback keeps the build green.
// Numbers refresh on each deploy (not live in the browser).
import { statsFallback, site } from './site';

function compact(n: number): string {
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1).replace(/\.0$/, '')}M`;
  if (n >= 1_000) return `${Math.round(n / 1000)}k`;
  return `${n}`;
}

async function safeJson(u: string, headers: Record<string, string> = {}): Promise<any | null> {
  try {
    const res = await fetch(u, { headers: { 'User-Agent': 'apdtflow-site', ...headers } });
    if (!res.ok) return null;
    return await res.json();
  } catch {
    return null;
  }
}

export async function fetchStats(): Promise<{ downloads: string; stars: string }> {
  const token = process.env.GITHUB_TOKEN;
  const ghHeaders = token ? { Authorization: `Bearer ${token}` } : {};

  const [pepy, gh] = await Promise.all([
    // Note: the api.pepy.tech v2 endpoint now requires an API key; the bare
    // pepy.tech endpoint still serves total_downloads without one.
    safeJson('https://pepy.tech/api/v2/projects/apdtflow'),
    safeJson('https://api.github.com/repos/yotambraun/APDTFlow', ghHeaders),
  ]);

  let downloads = statsFallback.downloads;
  if (pepy && typeof pepy.total_downloads === 'number') {
    downloads = `${compact(pepy.total_downloads)}+`;
  }

  let stars = statsFallback.stars;
  if (gh && typeof gh.stargazers_count === 'number') {
    stars = `${compact(gh.stargazers_count)}+`;
  }

  return { downloads, stars };
}

export const versionLabel = `v${site.version}`;
