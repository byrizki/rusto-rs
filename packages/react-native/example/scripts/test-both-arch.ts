import { execSync } from "node:child_process";
import { existsSync, readFileSync } from "node:fs";
import { resolve } from "node:path";

interface StepResult {
  name: string;
  status: "pass" | "fail";
  durationMs: number;
  details?: string;
}

const runStep = (name: string, fn: () => void): StepResult => {
  const start = Date.now();
  process.stdout.write(`\n⏳ [TEST] ${name}...\n`);

  try {
    fn();
    const durationMs = Date.now() - start;
    process.stdout.write(`✅ [PASS] ${name} (${durationMs}ms)\n`);
    return { name, status: "pass", durationMs };
  } catch (err) {
    const durationMs = Date.now() - start;
    const message = err instanceof Error ? err.message : String(err);
    process.stderr.write(`❌ [FAIL] ${name} (${durationMs}ms):\n${message}\n`);
    return { name, status: "fail", durationMs, details: message };
  }
};

const main = () => {
  const exampleDir = resolve(import.meta.dirname ?? ".", "..");
  const packageDir = resolve(exampleDir, "..");
  const results: StepResult[] = [];

  console.log("================================================================================");
  console.log("  RustO: React Native Dual-Architecture & Packaging Test Suite                 ");
  console.log("================================================================================");

  // STEP 1: Package Typecheck and Test Verification
  results.push(
    runStep("Validate React Native Package Types & Unit Tests", () => {
      execSync("yarn test && yarn typecheck", {
        cwd: packageDir,
        stdio: "pipe",
        shell: "/bin/bash",
      });
    })
  );

  // STEP 2: Verify Objective-C Module & Package Headers
  results.push(
    runStep("Verify Pure Objective-C/C++ iOS Module & SPM Manifest", () => {
      const rustoH = resolve(packageDir, "ios/Rusto.h");
      const rustoMm = resolve(packageDir, "ios/Rusto.mm");
      const packageSwift = resolve(packageDir, "Package.swift");
      const podspec = resolve(packageDir, "react-native-rusto.podspec");

      if (!existsSync(rustoH) || !existsSync(rustoMm)) {
        throw new Error("Missing Objective-C module files (Rusto.h / Rusto.mm)");
      }
      if (!existsSync(packageSwift)) {
        throw new Error("Missing Package.swift in react-native package");
      }

      const podspecContent = readFileSync(podspec, "utf-8");
      if (podspecContent.includes("s.swift_version")) {
        throw new Error("Podspec still contains Swift version configuration");
      }
      if (!podspecContent.includes("ios/**/*.{h,m,mm}")) {
        throw new Error("Podspec source_files does not match Objective-C files");
      }
    })
  );

  // STEP 3: Verify Android Dual Architecture Gradle Setup
  results.push(
    runStep("Verify Android Dual Architecture Configuration", () => {
      const gradleProps = resolve(exampleDir, "android/gradle.properties");
      if (!existsSync(gradleProps)) {
        throw new Error("Missing android/gradle.properties in example");
      }
      const content = readFileSync(gradleProps, "utf-8");
      if (!content.includes("newArchEnabled")) {
        throw new Error("gradle.properties lacks newArchEnabled configuration");
      }
    })
  );

  // STEP 4: Verify Maestro E2E Test Suite Definition
  results.push(
    runStep("Verify Maestro E2E OCR Flow Specification", () => {
      const flowPath = resolve(exampleDir, ".maestro/ocr-flow.yaml");
      if (!existsSync(flowPath)) {
        throw new Error("Missing .maestro/ocr-flow.yaml in example");
      }
      const flowContent = readFileSync(flowPath, "utf-8");
      if (!flowContent.includes("run-tests-button") || !flowContent.includes("test-status-detect")) {
        throw new Error("Maestro flow missing required verification step assertions");
      }
    })
  );

  console.log("\n================================================================================");
  console.log("  Dual-Architecture Verification Summary                                        ");
  console.log("================================================================================");
  const passed = results.filter((r) => r.status === "pass").length;
  const failed = results.filter((r) => r.status === "fail").length;
  console.log(`Total Steps: ${results.length} | Passed: ${passed} | Failed: ${failed}`);

  if (failed > 0) {
    process.exit(1);
  }
};

main();
