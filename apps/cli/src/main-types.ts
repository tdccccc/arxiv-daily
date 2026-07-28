export interface WritableTextStream {
  write(chunk: string): unknown;
}

export interface CliIo {
  stdout: WritableTextStream;
  stderr: WritableTextStream;
}
