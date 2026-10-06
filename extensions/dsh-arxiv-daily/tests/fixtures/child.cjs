const mode = process.env.DSH_TEST_CHILD;
if (mode === 'missing') { process.stderr.write('CLI config not found: fixture-secret\n'); process.exit(1); }
if (mode !== 'silent') {
  process.stdout.write('Work');
  setTimeout(() => process.stdout.write('bench: http://127.0.0.1:8123/aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa/\n'), 20);
}
setInterval(() => {}, 1000);
