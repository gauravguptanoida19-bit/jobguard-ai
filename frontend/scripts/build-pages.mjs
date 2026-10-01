import { execSync } from "child_process";
import fs from "fs";
import path from "path";

console.log("🚀 Building JobGuard AI for GitHub Pages...");

try {
  // 1. Run Next.js build with static export flags
  console.log("📦 Compiling static export with basePath: /jobguard-ai ...");
  execSync("npx next build", {
    stdio: "inherit",
    env: {
      ...process.env,
      OUTPUT_EXPORT: "true",
      GITHUB_ACTIONS: "true",
    },
  });

  // 2. Create 404.html SPA fallback
  const outDir = path.resolve("out");
  const indexHtml = path.join(outDir, "index.html");
  const fallbackHtml = path.join(outDir, "404.html");

  if (fs.existsSync(indexHtml)) {
    fs.copyFileSync(indexHtml, fallbackHtml);
    console.log("✅ Created out/404.html SPA fallback for GitHub Pages");
  }

  // 3. Create .nojekyll so GitHub Pages does not ignore underscore files like _next
  fs.writeFileSync(path.join(outDir, ".nojekyll"), "");
  console.log("✅ Created out/.nojekyll");

  console.log("🎉 GitHub Pages build complete in frontend/out/");
} catch (err) {
  console.error("❌ Build failed:", err);
  process.exit(1);
}
