/**
 * Records what the renderer actually requested, so "the disabled sidecar sent
 * nothing" is an observation rather than an inference from the setting value.
 */
export async function createRequestLog(client) {
  const requests = [];
  client.on("Network.requestWillBeSent", (params) => {
    const url = params?.request?.url;
    if (typeof url === "string") requests.push(url);
  });
  await client.send("Network.enable");
  return {
    urls: () => [...requests],
    // Obsidian serves its own assets over app://, and data:/blob: URIs never
    // leave the process. Counting them would make "no sidecar traffic" read as
    // if requests had been sent.
    networkUrls: () =>
      requests.filter(
        (url) => !url.startsWith("app://") && !url.startsWith("data:") && !url.startsWith("blob:"),
      ),
  };
}
