import type { Env } from "./kv";

export async function sendResendEmail(
  env: Env,
  opts: {
    to: string;
    subject: string;
    html: string;
    text: string;
  },
): Promise<{ id?: string }> {
  const from = env.FROM_NAME
    ? `${env.FROM_NAME.replace(/"/g, "")} <${env.FROM_EMAIL}>`
    : env.FROM_EMAIL;

  const res = await fetch("https://api.resend.com/emails", {
    method: "POST",
    headers: {
      Authorization: `Bearer ${env.RESEND_API_KEY}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      from,
      to: [opts.to],
      subject: opts.subject,
      html: opts.html,
      text: opts.text,
    }),
  });

  const bodyText = await res.text();
  if (!res.ok) {
    throw new Error(`Resend HTTP ${res.status}: ${bodyText.slice(0, 400)}`);
  }
  try {
    const parsed = JSON.parse(bodyText) as { id?: string };
    return { id: typeof parsed.id === "string" ? parsed.id : undefined };
  } catch {
    return {};
  }
}
