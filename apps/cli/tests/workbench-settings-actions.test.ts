import { afterEach, describe, expect, it, vi } from "vitest";
import * as fs from "node:fs/promises";
import * as os from "node:os";
import * as path from "node:path";
import { buildNodeHostAdapters } from "@arxiv-daily/node-runtime";
import { readWorkbenchSettings, saveWorkbenchSettings } from "../src/workbench/settings";
import { inspectWorkbenchLibrary, performSettingsAction } from "../src/workbench/settings-actions";
const roots: string[] = [];
afterEach(async () => { await Promise.all(roots.splice(0).map(root => fs.rm(root, { recursive: true, force: true }))); });
async function fixture() {
 const root = await fs.mkdtemp(path.join(os.tmpdir(), "settings-actions-")); roots.push(root);
 const configPath = path.join(root, "config.toml");
 const view = await readWorkbenchSettings(configPath);
 const config = await saveWorkbenchSettings(configPath, {revision: null, values: {...view.values, vaultRoot: path.join(root, "vault"), baseUrl:"https://model.test/v1", apiKey:"secret-api-key", model:"m", categories:["cs.AI"], topics:[{id:"a",name:"AI",tag:"ai",description:"AI",detail:false}]}});
 const libraryPath = path.join(root, "library"); await fs.mkdir(libraryPath);
 return {config, libraryPath};
}
const io = {stdout:{write:vi.fn()},stderr:{write:vi.fn()}};
const signal = () => new AbortController().signal;
describe("settings actions", () => {
 it("connects without authorizing or processing, requires the disclosed fingerprint before build, and revokes", async () => {
  const {config,libraryPath} = await fixture(); const runLibrary = vi.fn(async () => 0);
  const connected = await performSettingsAction(config,{action:"library-connect",path:libraryPath},io,signal(),{runLibrary});
  expect(connected.library?.status.kind).not.toBe("authorized"); expect(runLibrary).not.toHaveBeenCalled();
  for(const fingerprint of [undefined,"stale"]) await expect(performSettingsAction(connected.config!,{action:"library-build",...(fingerprint?{fingerprint}:{})},io,signal(),{runLibrary})).rejects.toThrow(/fingerprint/i);
  expect(runLibrary).not.toHaveBeenCalled();
  const built = await performSettingsAction(connected.config!,{action:"library-build",fingerprint:connected.library!.disclosure!.authorizationFingerprint},io,signal(),{runLibrary});
  expect(built.library?.status.kind).toBe("authorized"); expect(runLibrary.mock.calls.map(call => call[1])).toEqual([["prepare"],["scan"],["index"]]);
  const revoked = await performSettingsAction(built.config!,{action:"library-revoke"},io,signal());
  expect(inspectWorkbenchLibrary(revoked.config!).status.kind).not.toBe("authorized");
 });
 it("lists models using the saved endpoint and rejects unrecognized request fields", async () => {
  const {config} = await fixture(); const request = vi.fn(async () => ({status:200,bodyText:JSON.stringify({data:[{id:"model-z"},{id:"model-a"}]}),headers:{}}));
  expect((await performSettingsAction(config,{action:"models"},io,signal(),{http:{request}})).models).toEqual(["model-z","model-a"]);
  expect(request.mock.calls[0]![0]).toMatchObject({url:"https://model.test/v1/models",method:"GET"});
  for(const body of [null,[],{action:"nope"},{action:"models",apiKey:"other"},{action:"library-connect",path:5}]) await expect(performSettingsAction(config,body,io,signal(),{http:{request}})).rejects.toThrow();
 });
 it("stops a build on failure and refuses already-cancelled actions", async () => {
  const {config,libraryPath} = await fixture(); const connected = await performSettingsAction(config,{action:"library-connect",path:libraryPath},io,signal());
  const runLibrary = vi.fn(async () => 1);
  await expect(performSettingsAction(connected.config!,{action:"library-build",fingerprint:connected.library!.disclosure!.authorizationFingerprint},io,signal(),{runLibrary})).rejects.toThrow(/prepare/);
  expect(runLibrary).toHaveBeenCalledTimes(1);
  const controller = new AbortController(); controller.abort();
  await expect(performSettingsAction(config,{action:"models"},io,controller.signal)).rejects.toThrow();
 });
 it("sends email only through explicit actions and disposes the temporary runtime on success and failure", async () => {
  const {config} = await fixture();
  config.settings.email = {...config.settings.email, mode:"self", to:"user@example.test", apiKey:"mail-secret", from:"papers@example.test"};
  const dispose = vi.fn();
  const host = buildNodeHostAdapters({rootDir:config.vaultRoot});
  const buildRuntime = vi.fn(async () => ({host,dispose}));
  const request = vi.fn(async () => ({status:200,headers:{},bodyText:JSON.stringify({id:"test-message"})}));
  await expect(performSettingsAction(config,{action:"email-test"},io,signal(),{http:{request},buildRuntime})).resolves.toMatchObject({message:"测试邮件已发送"});
  expect(dispose).toHaveBeenCalledTimes(1); expect(request).toHaveBeenCalledTimes(1);
  const failure = {request: vi.fn(async () => {throw new Error("mail-secret secret-api-key rejected");})};
  await expect(performSettingsAction(config,{action:"email-verify"},io,signal(),{http:failure,buildRuntime})).rejects.toThrow("验证邮件发送失败");
  expect(dispose).toHaveBeenCalledTimes(2);
  expect(io.stderr.write.mock.calls.flat().join(" ")).not.toContain("mail-secret");
  expect(io.stderr.write.mock.calls.flat().join(" ")).not.toContain("secret-api-key");
 });

 it("does not report a completed build after cancellation during the final operation", async () => {
  const {config,libraryPath} = await fixture();
  const connected = await performSettingsAction(config,{action:"library-connect",path:libraryPath},io,signal());
  const controller = new AbortController();
  const runLibrary = vi.fn(async (_config, args: string[]) => {if(args[0] === "index") controller.abort(new Error("cancelled-final-index")); return 0;});
  await expect(performSettingsAction(connected.config!,{action:"library-build",fingerprint:connected.library!.disclosure!.authorizationFingerprint},io,controller.signal,{runLibrary})).rejects.toThrow("cancelled-final-index");
 });
 it("cancels the default HTTP transport and does not try another model endpoint", async () => {
  const {config} = await fixture(); const controller = new AbortController();
  const fetch = vi.fn((_url, options) => new Promise((_resolve, reject) => {
   options.signal.addEventListener("abort", () => reject(options.signal.reason), {once:true});
   controller.abort(new Error("cancelled-model-request"));
  }));
  vi.stubGlobal("fetch", fetch);
  try {
   await expect(performSettingsAction(config,{action:"models"},io,controller.signal)).rejects.toThrow(/cancelled/);
   expect(fetch).toHaveBeenCalledTimes(1);
  } finally {vi.unstubAllGlobals();}
 });

 it("checks cancellation after email completion and still disposes the runtime", async () => {
  const {config} = await fixture(); config.settings.email = {...config.settings.email,mode:"self",to:"user@example.test",apiKey:"test-mail-key"};
  const controller = new AbortController(); const dispose = vi.fn();
  const host = buildNodeHostAdapters({rootDir:config.vaultRoot});
  const request = vi.fn(async () => { controller.abort(new Error("cancelled-mail-request")); return {status:200,headers:{},bodyText:JSON.stringify({id:"test-message"})}; });
  await expect(performSettingsAction(config,{action:"email-test"},io,controller.signal,{http:{request},buildRuntime:async()=>({host,dispose})})).rejects.toThrow("cancelled-mail-request");
  expect(dispose).toHaveBeenCalledOnce();
 });

});
