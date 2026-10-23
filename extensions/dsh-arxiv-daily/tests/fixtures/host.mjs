// Plays the DSH host for process.test.mjs: owns one real workbench child, reports it, and then idles until the
// test kills this process without letting it dispose anything.
import { spawn } from 'node:child_process';
import { WorkbenchProcess } from '../../src/workbench-process.mjs';
let pid;
const manager = new WorkbenchProcess({ cliPath: process.argv[2], env: process.env, spawn: (...args) => { const child = spawn(...args); pid = child.pid; return child; } });
const url = await manager.open();
process.stdout.write(`${JSON.stringify({ pid, url })}\n`);
setInterval(() => {}, 1000);
