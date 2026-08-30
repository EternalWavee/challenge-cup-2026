import crypto from "crypto";
import fs from "fs";
import path from "path";
import url from "url";

const { createHash } = crypto;
const { readdir, readFile, writeFile } = fs.promises;
const { dirname, join, relative, resolve } = path;
const { fileURLToPath } = url;

const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const trackedRoots = ["src", "public"];
const trackedFiles = ["index.html", "package.json", "pnpm-lock.yaml"];

async function walk(directory) {
  const entries = await readdir(directory, { withFileTypes: true });
  const paths = [];
  for (const entry of entries) {
    const path = join(directory, entry.name);
    if (entry.isDirectory()) paths.push(...(await walk(path)));
    else if (entry.isFile()) paths.push(path);
  }
  return paths;
}

async function main() {
  const walkedRoots = await Promise.all(
    trackedRoots.map((path) => walk(join(root, path))),
  );
  const sourcePaths = [
    ...trackedFiles.map((path) => join(root, path)),
    ...walkedRoots.flat(),
  ].sort();

  const sources = {};
  for (const path of sourcePaths) {
    const digest = createHash("sha256")
      .update(await readFile(path))
      .digest("hex");
    sources[relative(root, path)] = digest;
  }

  await writeFile(
    join(root, "dist", "build-info.json"),
    `${JSON.stringify({ schema_version: 1, sources }, null, 2)}\n`,
    "utf8",
  );
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
